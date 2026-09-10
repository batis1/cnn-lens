// PYTORCH_BACKEND_INTEGRATION:
// This adapter converts backend `/api/explain` JSON into the same in-memory
// Node/Link shape produced by `cnn-tf.js`. To return to TensorFlow.js, leave
// this file in place and switch `ACTIVE_MODEL_RUNTIME` in Overview.svelte.

class Node {
  constructor(layerName, index, type, bias, output) {
    this.layerName = layerName;
    this.index = index;
    this.type = type;
    this.bias = bias;
    this.output = output;
    this.inputLinks = [];
    this.outputLinks = [];
  }
}

class Link {
  constructor(source, dest, weight) {
    this.source = source;
    this.dest = dest;
    this.weight = weight;
  }
}

export const defaultBackendApiBase = import.meta.env.VITE_BACKEND_API_BASE ||
  (import.meta.env.PROD ? '/api' : 'http://127.0.0.1:8000/api');

export const getBackendImageUrl = (imagePath, apiBase = defaultBackendApiBase) =>
  imagePath?.startsWith('data:image/') ? imagePath : `${apiBase}/image?path=${encodeURIComponent(imagePath)}`;

export const loadBackendModelOptions = async (apiBase = defaultBackendApiBase) => {
  let response = await fetch(`${apiBase}/models`);
  if (!response.ok) {
    throw new Error(`Backend model request failed with ${response.status}`);
  }

  let payload = await response.json();
  return (payload.models || [])
    .filter((model) => model.supportedForExplanation)
    .map((model) => ({
      id: model.id,
      label: model.label,
      runtime: 'pytorch-backend',
      family: model.family,
      architectureKey: model.architectureKey,
      supportedForExplanation: model.supportedForExplanation,
      modelPath: model.sourcePath,
      sourcePath: model.sourcePath,
      inputShape: model.inputShape,
      pixelValueScale: model.pixelValueScale,
      classLabels: model.classLabels,
      totalLayers: 6,
      summary: model.architectureHint,
      convKernelSummary: '',
      backendModel: model,
    }));
};

export const explainWithBackend = async (
  modelId,
  imagePath,
  apiBase = defaultBackendApiBase,
) => {
  let response = await fetch(`${apiBase}/explain`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ modelId, imagePath }),
  });

  if (!response.ok) {
    let errorPayload = await response.json().catch(() => ({}));
    throw new Error(
      errorPayload.message ||
      errorPayload.error ||
      `Backend explain request failed with ${response.status}`,
    );
  }

  return response.json();
};

export const prepareCustomImageWithBackend = async (
  source,
  apiBase = defaultBackendApiBase,
) => {
  let response = await fetch(`${apiBase}/custom-image`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ source }),
  });

  if (!response.ok) {
    let errorPayload = await response.json().catch(() => ({}));
    throw new Error(
      errorPayload.error || `Custom image request failed with ${response.status}`,
    );
  }

  return response.json();
};

// AI_TEST_UI_INTEGRATION:
// Calls the Flask `/api/test` endpoint that ports AI_Test_UI/test.py into
// frontend-friendly JSON. To hide the UI, switch ENABLE_AI_TEST_UI in
// Overview.svelte to false; the backend helper can stay unused.
export const testWithBackend = async (
  payload,
  apiBase = defaultBackendApiBase,
) => {
  let response = await fetch(`${apiBase}/test`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    let errorPayload = await response.json().catch(() => ({}));
    throw new Error(
      errorPayload.message ||
      errorPayload.error ||
      `Backend test request failed with ${response.status}`,
    );
  }

  return response.json();
};

export const generateDataWithBackend = async (
  payload,
  apiBase = defaultBackendApiBase,
) => {
  let response = await fetch(`${apiBase}/generate-data`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    let errorPayload = await response.json().catch(() => ({}));
    throw new Error(
      errorPayload.message ||
      errorPayload.error ||
      `Backend data-generation request failed with ${response.status}`,
    );
  }

  return response.json();
};

export const constructCNNFromBackendExplanation = (explanation) => {
  let cnn = [];
  let probabilities = explanation?.prediction?.probabilities || [];
  let graphLayers = explanation?.graph?.layers || [];

  graphLayers.forEach((layer, layerIndex) => {
    let curLayer = layer.nodes.map((nodeData) => {
      let output = normalizeNodeOutputForFrontend(layer, nodeData, probabilities);
      let node = new Node(
        layer.name,
        nodeData.index,
        layer.type,
        nodeData.bias || 0,
        output,
      );

      node.backendLayerIndex = layerIndex;
      node.backendLayer = layer;
      node.backendNode = nodeData;

      if (nodeData.activationName) {
        node.activationName = nodeData.activationName;
      }

      if (nodeData.isOutputLayer !== undefined) {
        node.isOutputLayer = nodeData.isOutputLayer;
      }

      if (nodeData.logit !== null && nodeData.logit !== undefined) {
        node.logit = nodeData.logit;
      }

      return node;
    });

    cnn.push(curLayer);
  });

  graphLayers.forEach((layer, layerIndex) => {
    if (layerIndex === 0) {
      return;
    }

    layer.nodes.forEach((nodeData, nodeIndex) => {
      let dest = cnn[layerIndex][nodeIndex];

      (nodeData.inputLinks || []).forEach((linkData) => {
        let sourceLayer = cnn[linkData.sourceLayerIndex];
        if (!sourceLayer) {
          return;
        }

        let source = sourceLayer[linkData.sourceNodeIndex];
        if (!source) {
          return;
        }

        let weight = readFrontendLinkWeight(layer.type, linkData);
        let link = new Link(source, dest, weight);
        link.backendLink = linkData;
        source.outputLinks.push(link);
        dest.inputLinks.push(link);
      });
    });
  });

  cnn.backendExplanation = explanation;
  return cnn;
};

const normalizeNodeOutputForFrontend = (layer, nodeData, probabilities) => {
  if (layer.type === 'input') {
    return normalizeImageChannel(nodeData.output);
  }

  if (layer.name === 'output' && probabilities[nodeData.index] !== undefined) {
    return probabilities[nodeData.index];
  }

  return nodeData.output;
};

const normalizeImageChannel = (channel) => {
  if (!Array.isArray(channel)) {
    return channel;
  }

  let maxValue = channel.reduce((maxRow, row) =>
    Math.max(maxRow, ...row), -Infinity);

  if (maxValue <= 1) {
    return channel;
  }

  return channel.map((row) => row.map((value) => value / 255));
};

const readFrontendLinkWeight = (layerType, linkData) => {
  if (layerType === 'flatten' && linkData.position) {
    return linkData.position;
  }

  return linkData.weight;
};
