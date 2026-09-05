/* global tf */

// Network input image size
const networkInputSize = 64;

// Enum of node types
const nodeType = {
  INPUT: 'input',
  CONV: 'conv',
  POOL: 'pool',
  RELU: 'relu',
  FC: 'fc',
  FLATTEN: 'flatten'
}

class Node {
  /**
   * Class structure for each neuron node.
   * 
   * @param {string} layerName Name of the node's layer.
   * @param {int} index Index of this node in its layer.
   * @param {string} type Node type {input, conv, pool, relu, fc}. 
   * @param {number} bias The bias assocated to this node.
   * @param {number[]} output Output of this node.
   */
  constructor(layerName, index, type, bias, output) {
    this.layerName = layerName;
    this.index = index;
    this.type = type;
    this.bias = bias;
    this.output = output;

    // Weights are stored in the links
    this.inputLinks = [];
    this.outputLinks = [];
  }
}

class Link {
  /**
   * Class structure for each link between two nodes.
   * 
   * @param {Node} source Source node.
   * @param {Node} dest Target node.
   * @param {number} weight Weight associated to this link. It can be a number,
   *  1D array, or 2D array.
   */
  constructor(source, dest, weight) {
    this.source = source;
    this.dest = dest;
    this.weight = weight;
  }
}

const getLayerClassName = (layer) => {
  let className = undefined;
  if (layer.getClassName) {
    className = layer.getClassName();
  } else if (layer.className) {
    className = layer.className;
  } else if (layer.constructor && layer.constructor.className) {
    className = layer.constructor.className;
  }

  return className;
}

const isInputLayer = (layer) => {
  let className = getLayerClassName(layer);

  return className === 'InputLayer' || layer.name.includes('input_layer');
}

// Task 3 at lines 70-80: newer trained exports expose layer tensors differently
// from the legacy model, so this helper reads weights and biases through either API.
const readLayerTensor = (layerTensor) => {
  if (layerTensor === undefined || layerTensor === null) {
    return undefined;
  }

  if (layerTensor.val) {
    return layerTensor.val;
  }

  if (typeof layerTensor.read === 'function') {
    return layerTensor.read();
  }

  return layerTensor;
}

/**
 * Construct a CNN with given extracted outputs from every layer.
 * 
 * @param {number[][]} allOutputs Array of outputs for each layer.
 *  allOutputs[i][j] is the output for layer i node j.
 * @param {Model} model Loaded tf.js model.
 * @param {Tensor} inputImageTensor Loaded input image tensor.
 */
const constructCNNFromOutputs = (allOutputs, model, inputImageTensor) => {
  let cnn = [];
  let modelLayers = model.layers.filter(layer => !isInputLayer(layer));

  // Add the first layer (input layer)
  let inputLayer = [];
  let inputSourceLayer = model.layers.find(layer =>
    layer.batchInputShape !== undefined) || modelLayers[0];
  let inputShape = inputSourceLayer.batchInputShape.slice(1);
  let inputImageArray = inputImageTensor.transpose([2, 0, 1]).arraySync();

  // First layer's three nodes' outputs are the channels of inputImageArray
  for (let i = 0; i < inputShape[2]; i++) {
    let node = new Node('input', i, nodeType.INPUT, 0, inputImageArray[i]);
    inputLayer.push(node);
  }

  cnn.push(inputLayer);
  let curLayerIndex = 1;

  for (let l = 0; l < modelLayers.length; l++) {
    let layer = modelLayers[l];
    let layerClassName = getLayerClassName(layer);
    // Get the current output
    let outputs = allOutputs[l].squeeze();
    outputs = outputs.arraySync();

    let curLayerNodes = [];
    let curLayerType;
    // Task 6.2
    // Identify layer type based on the layer name
    if (layer.name.includes('conv')) {
      curLayerType = nodeType.CONV;
    } else if (layer.name.includes('pool')) {
      curLayerType = nodeType.POOL;
    } else if (layer.name.includes('relu') || layer.name.includes('sigmoid')) {
      curLayerType = nodeType.RELU;
    } else if (
      layer.name.includes('output') ||
      layer.name.includes('dense') ||
      layer.name.includes('fc') ||
      layerClassName === 'Dense'
    ) {
      curLayerType = nodeType.FC;
    } else if (layer.name.includes('flatten')) {
      curLayerType = nodeType.FLATTEN;
    } else {
      console.log('Find unknown type');
    }

    // Construct this layer based on its layer type
    switch (curLayerType) {
      case nodeType.CONV: {
        let biasTensor = readLayerTensor(layer.bias);
        let kernelTensor = readLayerTensor(layer.kernel);
        let biases = biasTensor.arraySync();
        // The new order is [output_depth, input_depth, height, width]
        let weights = kernelTensor.transpose([3, 2, 0, 1]).arraySync();

        // Add nodes into this layer
        for (let i = 0; i < outputs.length; i++) {
          let node = new Node(layer.name, i, curLayerType, biases[i],
            outputs[i]);

          // Connect this node to all previous nodes (create links)
          // CONV layers have weights in links. Links are one-to-multiple.
          for (let j = 0; j < cnn[curLayerIndex - 1].length; j++) {
            let preNode = cnn[curLayerIndex - 1][j];
            let curLink = new Link(preNode, node, weights[i][j]);
            preNode.outputLinks.push(curLink);
            node.inputLinks.push(curLink);
          }
          curLayerNodes.push(node);
        }
        break;
      }
      case nodeType.FC: {
        let biasTensor = readLayerTensor(layer.bias);
        let kernelTensor = readLayerTensor(layer.kernel);
        let biases = biasTensor.arraySync();
        // The new order is [output_depth, input_depth]
        let weights = kernelTensor.transpose([1, 0]).arraySync();

        // Add nodes into this layer
        for (let i = 0; i < outputs.length; i++) {
          let node = new Node(layer.name, i, curLayerType, biases[i],
            outputs[i]);
          node.activationName = layer.activation?.name || 'linear';
          node.isOutputLayer = layer.name.includes('output');

          // Connect this node to all previous nodes (create links)
          // FC layers have weights in links. Links are one-to-multiple.

          // Since we are visualizing the logit values, we need to track
          // the raw value before softmax
          let curLogit = 0;
          for (let j = 0; j < cnn[curLayerIndex - 1].length; j++) {
            let preNode = cnn[curLayerIndex - 1][j];
            let curLink = new Link(preNode, node, weights[i][j]);
            preNode.outputLinks.push(curLink);
            node.inputLinks.push(curLink);
            curLogit += preNode.output * weights[i][j];
          }
          curLogit += biases[i];
          node.logit = curLogit;
          curLayerNodes.push(node);
        }

        if (cnn[curLayerIndex - 1][0]?.type === nodeType.FLATTEN) {
          // Sort flatten layer based on the node TF index
          cnn[curLayerIndex - 1].sort((a, b) => a.realIndex - b.realIndex);
        }
        break;
      }
      case nodeType.RELU:
      case nodeType.POOL: {
        // RELU and POOL have no bias nor weight
        let bias = 0;
        let weight = null;

        // Add nodes into this layer
        for (let i = 0; i < outputs.length; i++) {
          let node = new Node(layer.name, i, curLayerType, bias, outputs[i]);

          // RELU and POOL layers have no weights. Links are one-to-one
          let preNode = cnn[curLayerIndex - 1][i];
          let link = new Link(preNode, node, weight);
          preNode.outputLinks.push(link);
          node.inputLinks.push(link);

          curLayerNodes.push(node);
        }
        break;
      }
      case nodeType.FLATTEN: {
        // Flatten layer has no bias nor weights.
        let bias = 0;

        for (let i = 0; i < outputs.length; i++) {
          // Flatten layer has no weights. Links are multiple-to-one.
          // Use dummy weights to store the corresponding entry in the previsou
          // node as (row, column)
          // The flatten() in tf2.keras has order: channel -> row -> column
          let preNodeWidth = cnn[curLayerIndex - 1][0].output.length,
            preNodeNum = cnn[curLayerIndex - 1].length,
            preNodeIndex = i % preNodeNum,
            preNodeRow = Math.floor(Math.floor(i / preNodeNum) / preNodeWidth),
            preNodeCol = Math.floor(i / preNodeNum) % preNodeWidth,
            // Use channel, row, colume to compute the real index with order
            // row -> column -> channel
            curNodeRealIndex = preNodeIndex * (preNodeWidth * preNodeWidth) +
              preNodeRow * preNodeWidth + preNodeCol;

          let node = new Node(layer.name, i, curLayerType,
            bias, outputs[i]);

          // TF uses the (i) index for computation, but the real order should
          // be (curNodeRealIndex). We will sort the nodes using the real order
          // after we compute the logits in the output layer.
          node.realIndex = curNodeRealIndex;

          let link = new Link(cnn[curLayerIndex - 1][preNodeIndex],
            node, [preNodeRow, preNodeCol]);

          cnn[curLayerIndex - 1][preNodeIndex].outputLinks.push(link);
          node.inputLinks.push(link);

          curLayerNodes.push(node);
        }

        // Sort flatten layer based on the node TF index
        curLayerNodes.sort((a, b) => a.index - b.index);
        break;
      }
      default:
        console.error('Encounter unknown layer type');
        break;
    }

    // Add current layer to the NN
    cnn.push(curLayerNodes);
    curLayerIndex++;
  }

  return cnn;
}

/**
 * Construct a CNN with given model and input.
 * 
 * @param {string} inputImageFile filename of input image.
 * @param {Model} model Loaded tf.js model.
 */
export const constructCNN = async (inputImageFile, model) => {
  console.log('[constructCNN] input image:', inputImageFile);
  console.log('[constructCNN] model layer names:',
    model.layers.map(layer => layer.name));

  // Load the image file
  let inputImageTensor = await getInputImageArray(inputImageFile, true);

  // Need to feed the model with a batch
  let inputImageTensorBatch = tf.stack([inputImageTensor]);

  // To get intermediate layer outputs, we will iterate through all layers in
  // the model, and sequencially apply transformations.
  let preTensor = inputImageTensorBatch;
  let outputs = [];

  // Iterate through all layers, and build one model with that layer as output
  for (let l = 0; l < model.layers.length; l++) {
    let layer = model.layers[l];

    if (isInputLayer(layer)) {
      console.log('[constructCNN] skipping input layer:', layer.name);
      continue;
    }

    let curTensor = layer.apply(preTensor);

    // Record the output tensor
    // Because there is only one element in the batch, we use squeeze()
    // We also want to use CHW order here
    let output = curTensor.squeeze();
    if (output.shape.length === 3) {
      output = output.transpose([2, 0, 1]);
    }
    outputs.push(output);

    // Update preTensor for next nesting iteration
    preTensor = curTensor;
  }

  let cnn = constructCNNFromOutputs(outputs, model, inputImageTensor);
  console.log('[constructCNN] constructed visible layers:', cnn.map(layer => layer[0].layerName));
  return cnn;
}

// Helper functions

/**
 * Convert canvas image data into a 3D tensor with dimension [height, width, 3].
 * Recall that tensorflow uses NHWC order (batch, height, width, channel).
 * Each pixel is in 0-255 scale.
 * 
 * @param {[int8]} imageData Canvas image data
 * @param {int} width Canvas image width
 * @param {int} height Canvas image height
 */
const imageDataTo3DTensor = (imageData, width, height, normalize = true) => {
  // Create array placeholder for the 3d array
  let imageArray = tf.fill([width, height, 3], 0).arraySync();

  // Iterate through the data to fill out channel arrays above
  for (let i = 0; i < imageData.length; i++) {
    let pixelIndex = Math.floor(i / 4),
      channelIndex = i % 4,
      row = width === height ? Math.floor(pixelIndex / width)
        : pixelIndex % width,
      column = width === height ? pixelIndex % width
        : Math.floor(pixelIndex / width);

    if (channelIndex < 3) {
      let curEntry = imageData[i];
      // Normalize the original pixel value from [0, 255] to [0, 1]
      if (normalize) {
        curEntry /= 255;
      }
      imageArray[row][column][channelIndex] = curEntry;
    }
  }

  let tensor = tf.tensor3d(imageArray);
  return tensor;
}

/**
 * Get the 3D pixel value array of the given image file.
 * 
 * @param {string} imgFile File path to the image file
 * @returns A promise with the corresponding 3D array
 */
const getInputImageArray = (imgFile, normalize = true) => {
  let canvas = document.createElement('canvas');
  canvas.style.cssText = 'display:none;';
  document.getElementsByTagName('body')[0].appendChild(canvas);
  let context = canvas.getContext('2d');

  return new Promise((resolve, reject) => {
    let inputImage = new Image();
    inputImage.crossOrigin = "Anonymous";
    inputImage.src = imgFile;
    inputImage.onload = () => {
      // Task 3 at lines 381-385: match the Colab training pipeline by resizing
      // every browser input directly to 64x64 before running inference.
      // Match the Colab training pipeline: always resize directly to 64x64
      // and normalize to [0, 1], without extra browser-side crop/rotate logic.
      canvas.width = networkInputSize;
      canvas.height = networkInputSize;
      context.clearRect(0, 0, networkInputSize, networkInputSize);
      context.drawImage(inputImage, 0, 0, networkInputSize, networkInputSize);
      let canvasImage = context.getImageData(0, 0, networkInputSize, networkInputSize);

      // Get image data and convert it to a 3D array
      let imageData = canvasImage.data;
      let imageWidth = canvasImage.width;
      let imageHeight = canvasImage.height;

      // Remove this newly created canvas element
      canvas.parentNode.removeChild(canvas);

      resolve(imageDataTo3DTensor(imageData, imageWidth, imageHeight, normalize));
    }
    inputImage.onerror = reject;
  })
}

/**
 * Wrapper to load a model.
 * 
 * @param {string} modelFile Filename of converted (through tensorflowjs.py)
 *  model json file.
 */
export const loadTrainedModel = (modelFile) => {
  console.log('[loadTrainedModel] loading:', modelFile);
  // Task 3 at lines 411-439 and 452-487: normalize the raw Colab TensorFlow.js
  // export in memory, then fetch the matching weight shard(s), so the website
  // can switch among 7-layer, 12-layer, and 17-layer models without manual edits.
  const sanitizeModelJson = (json) => {
    let layers =
      json?.modelTopology?.model_config?.config?.layers;

    if (Array.isArray(layers) && layers.length > 0 &&
      layers[0].class_name === 'InputLayer') {
      let inputConfig = layers[0].config || {};
      let firstRealLayer = layers[1];
      if (firstRealLayer && firstRealLayer.config &&
        firstRealLayer.config.batch_input_shape === undefined) {
        firstRealLayer.config.batch_input_shape =
          inputConfig.batch_shape || inputConfig.batch_input_shape;
      }

      layers.shift();
    }

    if (json?.modelTopology?.model_config?.config) {
      json.modelTopology.model_config.config.name = 'sequential';
    }

    for (let manifest of json.weightsManifest || []) {
      for (let weight of manifest.weights || []) {
        weight.name = weight.name.replace(/^sequential(_1)?\//, '');
      }
    }

    return json;
  };

  const concatArrayBuffers = (buffers) => {
    let totalLength = buffers.reduce((sum, buffer) => sum + buffer.byteLength, 0);
    let temp = new Uint8Array(totalLength);
    let offset = 0;
    buffers.forEach((buffer) => {
      temp.set(new Uint8Array(buffer), offset);
      offset += buffer.byteLength;
    });
    return temp.buffer;
  };

  return fetch(modelFile)
    .then((response) => {
      if (!response.ok) {
        throw new Error(`Failed to fetch model json: ${modelFile}`);
      }
      return response.json();
    })
    .then(async (rawJson) => {
      let json = sanitizeModelJson(rawJson);
      let modelUrl = new URL(modelFile, window.location.href);
      let baseUrl = new URL('.', modelUrl);
      let weightSpecs = [];
      let weightBuffers = [];

      for (let manifest of json.weightsManifest || []) {
        weightSpecs.push(...(manifest.weights || []));

        for (let relPath of manifest.paths || []) {
          let weightUrl = new URL(relPath, baseUrl);
          let response = await fetch(weightUrl.toString());
          if (!response.ok) {
            throw new Error(`Failed to fetch weight shard: ${weightUrl.toString()}`);
          }
          weightBuffers.push(await response.arrayBuffer());
        }
      }

      let weightData = concatArrayBuffers(weightBuffers);

      return tf.loadLayersModel(
        tf.io.fromMemory(
          json.modelTopology,
          weightSpecs,
          weightData,
          json.trainingConfig,
        ),
      );
    })
    .then((model) => {
      console.log('[loadTrainedModel] loaded layers:',
        model.layers.map(layer => layer.name));
      return model;
    });
}
