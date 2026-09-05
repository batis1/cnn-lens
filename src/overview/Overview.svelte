<script>
  // Svelte functions
  import { onDestroy, onMount, tick } from "svelte";
  import {
    cnnStore,
    svgStore,
    vSpaceAroundGapStore,
    hSpaceAroundGapStore,
    nodeCoordinateStore,
    selectedScaleLevelStore,
    layerDisplayOrderStore,
    cnnLayerRangesStore,
    needRedrawStore,
    cnnLayerMinMaxStore,
    detailedModeStore,
    shouldIntermediateAnimateStore,
    isInSoftmaxStore,
    softmaxDetailViewStore,
    denseDetailViewStore,
    hoverInfoStore,
    allowsSoftmaxAnimationStore,
    modalStore,
    intermediateLayerPositionStore,
    manualLayerRevealStore,
  } from "../stores.js";

  // Svelte views
  import ConvolutionView from "../detail-view/Convolutionview.svelte";
  import ActivationView from "../detail-view/Activationview.svelte";
  import PoolView from "../detail-view/Poolview.svelte";
  import SoftmaxView from "../detail-view/Softmaxview.svelte";
  import DenseView from "../detail-view/Denseview.svelte";
  import Modal from "./Modal.svelte";
  import Article from "../article/Article.svelte";
  import AiTestPanel from "../ai-test-ui/AiTestPanel.svelte";

  // Overview functions
  import { loadTrainedModel, constructCNN } from "../utils/cnn-tf.js";
  import {
    constructCNNFromBackendExplanation,
    defaultBackendApiBase,
    explainWithBackend,
    getBackendImageUrl,
    loadBackendModelOptions,
  } from "../utils/cnn-backend.js";
  import { overviewConfig } from "../config.js";

  import {
    addOverlayRect,
    drawConv1,
    drawConv2,
    drawConv3,
    drawConv4,
    drawConv5,
    drawConv6,
  } from "./intermediate-draw.js";

  import { moveLayerX, addOverlayGradient } from "./intermediate-utils.js";

  import {
    drawFlatten,
    closeDenseNeuronDetail,
    resetDenseNeuronDetailLayout,
    softmaxDetailViewMouseOverHandler,
    softmaxDetailViewMouseLeaveHandler,
  } from "./flatten-draw.js";

  import {
    drawOutput,
    drawCNN,
    updateCNN,
    updateCNNLayerRanges,
    drawCustomImage,
    canRevealNextOverviewLayer,
    revealNextOverviewLayer,
    applyCurrentEdgeVisibility,
    resetOverviewLayerReveal,
    revealOverviewThroughLayer,
  } from "./overview-draw.js";

  // View bindings
  let overviewComponent;
  let scaleLevelSet = new Set(["local", "module", "global"]);
  let selectedScaleLevel = "local";
  selectedScaleLevelStore.set(selectedScaleLevel);
  let selectedLayerOrder = "model";
  layerDisplayOrderStore.set(selectedLayerOrder);
  let previousSelectedScaleLevel = selectedScaleLevel;
  let wholeSvg = undefined;
  let svg = undefined;

  $: selectedScaleLevel, selectedScaleLevelChanged();

  // Configs
  const layerColorScales = overviewConfig.layerColorScales;
  const nodeLength = overviewConfig.nodeLength;
  const plusSymbolRadius = overviewConfig.plusSymbolRadius;
  const numLayers = overviewConfig.numLayers;
  const edgeOpacity = overviewConfig.edgeOpacity;
  const edgeInitColor = overviewConfig.edgeInitColor;
  const edgeHoverColor = overviewConfig.edgeHoverColor;
  const edgeHoverOuting = overviewConfig.edgeHoverOuting;
  const edgeStrokeWidth = overviewConfig.edgeStrokeWidth;
  const intermediateColor = overviewConfig.intermediateColor;
  const kernelRectLength = overviewConfig.kernelRectLength;
  const svgPaddings = overviewConfig.svgPaddings;
  const gapRatio = overviewConfig.gapRatio;
  const overlayRectOffset = overviewConfig.overlayRectOffset;
  const classLists = overviewConfig.classLists;
  const logicalSvgWidth = 1680;
  const logicalSvgHeight = 760;
  // PYTORCH_BACKEND_INTEGRATION:
  // Active path for the new Flask/PyTorch backend. To return to the old
  // TensorFlow.js-only app, change this value to RUNTIME_TENSORFLOW_JS.
  // The original TensorFlow.js code path is kept below as `loadTensorFlowModelManifest`
  // and the TensorFlow branch in `loadModelById`.
  const RUNTIME_PYTORCH_BACKEND = "pytorch-backend";
  const RUNTIME_TENSORFLOW_JS = "tensorflow-js";
  // AI_TEST_UI_INTEGRATION_GUARD:
  // Keep this false to hide the AI_Test_UI-style testing panel while leaving
  // the existing CNN visualization unchanged. The panel is also guarded by
  // ACTIVE_MODEL_RUNTIME === RUNTIME_PYTORCH_BACKEND in the markup.
  const ENABLE_AI_TEST_UI = false;
  // AI_TEST_UI_STYLE_EXPERIMENT:
  // Set this to false to return to the original AI_Test_UI blue dashboard
  // styling without changing runtime/model behavior.
  const Style_Test = false;
  const DEFAULT_AI_TEST_UI_MODEL_ID = "cnn-net-28-ori";
  const DEFAULT_AI_VISUALIZATION_MODEL_ID = "tiny-vgg-12-pytorch";
  let isAiVisualizationSelected = false;
  $: isAiTestUiMode =
    ACTIVE_MODEL_RUNTIME === RUNTIME_PYTORCH_BACKEND && ENABLE_AI_TEST_UI;
  // ACTIVE_RUNTIME_ROLLBACK:
  // TensorFlow.js is active now. To return to the Flask/PyTorch backend, switch
  // this value back to RUNTIME_PYTORCH_BACKEND; do not delete either branch.
  // Keep both runtime choices here. Comment/uncomment these two lines to switch.
  const ACTIVE_MODEL_RUNTIME = RUNTIME_PYTORCH_BACKEND;
  // const ACTIVE_MODEL_RUNTIME = RUNTIME_TENSORFLOW_JS;
  const PYTORCH_BACKEND_API_BASE = defaultBackendApiBase;
  const tensorflowClassLabels = [...overviewConfig.classLists];
  const syncClassLabels = (labels) => {
    overviewConfig.classLists.splice(
      0,
      overviewConfig.classLists.length,
      ...(labels && labels.length ? labels : tensorflowClassLabels),
    );
  };

  // AI_TEST_UI_INTEGRATION:
  // The AI_Test_UI panel moves the CNN SVG lower on the page. Keep detail
  // panels anchored to the real SVG position instead of the old fixed 100px
  // page offset so TensorFlow and PyTorch views share the same click behavior.
  const getSvgPageOffset = () => {
    let svgElement = document.getElementById("cnn-svg");
    if (!svgElement) {
      return { top: 100, left: 0 };
    }

    let rect = svgElement.getBoundingClientRect();
    return {
      top: rect.top + window.scrollY,
      left: rect.left + window.scrollX,
    };
  };

  const positionFloatingDetailView = (topWithinSvg, leftWithinSvg) => {
    const detailview = document.getElementById("detailview");
    if (!detailview) {
      return;
    }

    let svgOffset = getSvgPageOffset();
    detailview.style.top = `${svgOffset.top + topWithinSvg}px`;
    detailview.style.left = `${svgOffset.left + leftWithinSvg}px`;
    detailview.style.position = "absolute";
    detailview.style.zIndex = "30";
  };
  const getOutputLabelBaseColor = () => (isAiTestUiMode ? "#ffffff" : "black");
  const formatToolbarLayerName = (name) => {
    if (!name) {
      return "";
    }

    let normalized = String(name).toLowerCase();
    if (normalized.includes("flatten")) {
      return "Flatten";
    }
    if (normalized.includes("avg_pool")) {
      return "Avg Pool";
    }
    if (normalized.includes("max_pool") || normalized.includes("pool")) {
      return "Pool";
    }
    if (normalized.includes("sigmoid")) {
      return "Sigmoid";
    }
    if (normalized.includes("relu")) {
      return "ReLU";
    }
    if (normalized.includes("dense")) {
      let suffix = normalized.match(/dense[_-]?(\d+)/)?.[1];
      return suffix ? `Dense ${suffix}` : "Dense";
    }
    if (normalized.includes("conv")) {
      let suffix = normalized.match(/conv[_-]?(\d+)(?:[_-]?(\d+))?/) || [];
      return suffix[1]
        ? `Conv ${suffix.slice(1).filter(Boolean).join(".")}`
        : "Conv";
    }
    if (normalized === "output") {
      return "Output";
    }

    return String(name).replace(/_/g, " ");
  };

  // Shared properties
  let needRedraw = [undefined, undefined];
  needRedrawStore.subscribe((value) => {
    needRedraw = value;
  });

  let nodeCoordinate = undefined;
  nodeCoordinateStore.subscribe((value) => {
    nodeCoordinate = value;
  });

  let cnnLayerRanges = undefined;
  cnnLayerRangesStore.subscribe((value) => {
    cnnLayerRanges = value;
  });

  let cnnLayerMinMax = undefined;
  cnnLayerMinMaxStore.subscribe((value) => {
    cnnLayerMinMax = value;
  });

  let detailedMode = undefined;
  detailedModeStore.subscribe((value) => {
    detailedMode = value;
  });

  let shouldIntermediateAnimate = undefined;
  shouldIntermediateAnimateStore.subscribe((value) => {
    shouldIntermediateAnimate = value;
  });

  let vSpaceAroundGap = undefined;
  vSpaceAroundGapStore.subscribe((value) => {
    vSpaceAroundGap = value;
  });

  let hSpaceAroundGap = undefined;
  hSpaceAroundGapStore.subscribe((value) => {
    hSpaceAroundGap = value;
  });

  let isInSoftmax = undefined;
  isInSoftmaxStore.subscribe((value) => {
    isInSoftmax = value;
  });

  let softmaxDetailViewInfo = undefined;
  softmaxDetailViewStore.subscribe((value) => {
    softmaxDetailViewInfo = value;
  });

  let denseDetailViewInfo = {};
  denseDetailViewStore.subscribe((value) => {
    denseDetailViewInfo = value;
  });

  let modalInfo = undefined;
  modalStore.subscribe((value) => {
    modalInfo = value;
  });

  let hoverInfo = undefined;
  hoverInfoStore.subscribe((value) => {
    hoverInfo = value;
  });

  let manualRevealInfo = {
    hasMore: false,
    nextLabel: "",
    progressText: "",
  };
  let autoRevealTimer = undefined;
  let isAutoRevealRunning = false;
  manualLayerRevealStore.subscribe((value) => {
    manualRevealInfo = value;
  });

  let intermediateLayerPosition = undefined;
  intermediateLayerPositionStore.subscribe((value) => {
    intermediateLayerPosition = value;
  });

  let width = undefined;
  let height = undefined;
  let model = undefined;
  let selectedNode = { layerName: "", index: -1, data: null };
  const layerFocusRadius = 2;
  let isLayerFocusActive = false;
  let focusedLayerIndex = 0;
  let layerSearchValue = "";
  let layerFocusAnnouncement = "";
  let isInIntermediateView = false;
  let isInActPoolDetailView = false;
  let actPoolDetailViewNodeIndex = -1;
  let actPoolDetailViewLayerIndex = -1;
  let detailedViewNum = undefined;
  let disableControl = false;

  // Wait to load
  let cnn = undefined;

  let detailedViewAbsCoords = {
    1: [600, 270, 490, 290],
    2: [500, 270, 490, 290],
    3: [700, 270, 490, 290],
    4: [600, 270, 490, 290],
    5: [650, 270, 490, 290],
    6: [775, 270, 490, 290],
    7: [100, 270, 490, 290],
    8: [60, 270, 490, 290],
    9: [200, 270, 490, 290],
    10: [300, 270, 490, 290],
  };

  let layerIndexDict = {};
  let layerLegendDict = {};
  let convLayerIndices = [];
  let visibleLayerCount = 0;

  const buildLayerLegendDict = (layerCount) => {
    let dict = {
      0: {
        local: "input-legend",
        module: "input-legend",
        global: "input-legend",
      },
    };

    for (let layerIndex = 1; layerIndex < layerCount - 1; layerIndex++) {
      let featureIndex = layerIndex - 1;
      let componentIndex = Math.floor(featureIndex / 5);
      let localSlot = featureIndex % 5 <= 1 ? 1 : 2;
      dict[layerIndex] = {
        local: `local-legend-${componentIndex}-${localSlot}`,
        module: `module-legend-${componentIndex}`,
        global: "global-legend",
      };
    }

    dict[layerCount - 1] = {
      local: "output-legend",
      module: "output-legend",
      global: "output-legend",
    };

    return dict;
  };

  const getConvDrawHandler = (curLayerIndex) => {
    let convOrder = convLayerIndices.indexOf(curLayerIndex);
    let handlers = [
      drawConv1,
      drawConv2,
      drawConv3,
      drawConv4,
      drawConv5,
      drawConv6,
    ];
    return handlers[Math.max(convOrder, 0)] || handlers[0];
  };

  $: if (cnn && cnn.length) {
    visibleLayerCount = cnn.length;
    layerIndexDict = Object.fromEntries(
      cnn.map((layer, index) => [layer[0].layerName, index]),
    );
    layerLegendDict = buildLayerLegendDict(visibleLayerCount);
    convLayerIndices = cnn
      .map((layer, index) => ({ type: layer[0].type, index }))
      .filter((entry) => entry.type === "conv")
      .map((entry) => entry.index);
  }

  // PYTORCH_BACKEND_INTEGRATION:
  // Old TensorFlow.js image list kept as a named fallback. To restore it, set
  // ACTIVE_MODEL_RUNTIME to RUNTIME_TENSORFLOW_JS above.
  /*
  Previous active TensorFlow.js image list:
  let imageOptions = [
    { file: "boat_1.jpeg", class: "lifeboat" },
    { file: "bug_1.jpeg", class: "ladybug" },
    { file: "pizza_1.jpeg", class: "pizza" },
    { file: "pepper_1.jpeg", class: "bell pepper" },
    { file: "bus_1.jpeg", class: "bus" },
    { file: "koala_1.jpeg", class: "koala" },
    { file: "espresso_1.jpeg", class: "espresso" },
    { file: "panda_1.jpeg", class: "red panda" },
    { file: "orange_1.jpeg", class: "orange" },
    { file: "car_1.jpeg", class: "sport car" },
  ];
  let selectedImage = imageOptions[6].file;
  */
  const tensorflowImageOptions = [
    { file: "boat_1.jpeg", class: "lifeboat" },
    { file: "bug_1.jpeg", class: "ladybug" },
    { file: "pizza_1.jpeg", class: "pizza" },
    { file: "pepper_1.jpeg", class: "bell pepper" },
    { file: "bus_1.jpeg", class: "bus" },
    { file: "koala_1.jpeg", class: "koala" },
    { file: "espresso_1.jpeg", class: "espresso" },
    { file: "panda_1.jpeg", class: "red panda" },
    { file: "orange_1.jpeg", class: "orange" },
    { file: "car_1.jpeg", class: "sport car" },
  ];
  const tinyVggImageOptions = tensorflowImageOptions.map((image) => ({
    ...image,
    backendPath: `assets/img/${image.file}`,
    src: `${import.meta.env.BASE_URL}assets/img/${image.file}`,
  }));
  const DEFAULT_TINY_VGG_IMAGE_FILE = "espresso_1.jpeg";
  const pytorchDigitImageOptions = [
    {
      file: "test_original/1_7.jpg",
      class: "digit 7",
      src: getBackendImageUrl(
        "test_original/1_7.jpg",
        PYTORCH_BACKEND_API_BASE,
      ),
    },
    {
      file: "test_original/11_0.jpg",
      class: "digit 0",
      src: getBackendImageUrl(
        "test_original/11_0.jpg",
        PYTORCH_BACKEND_API_BASE,
      ),
    },
    {
      file: "test_original/108_1.jpg",
      class: "digit 1",
      src: getBackendImageUrl(
        "test_original/108_1.jpg",
        PYTORCH_BACKEND_API_BASE,
      ),
    },
    {
      file: "test_original/107_2.jpg",
      class: "digit 2",
      src: getBackendImageUrl(
        "test_original/107_2.jpg",
        PYTORCH_BACKEND_API_BASE,
      ),
    },
    {
      file: "test_original/113_3.jpg",
      class: "digit 3",
      src: getBackendImageUrl(
        "test_original/113_3.jpg",
        PYTORCH_BACKEND_API_BASE,
      ),
    },
    {
      file: "test_original/104_4.jpg",
      class: "digit 4",
      src: getBackendImageUrl(
        "test_original/104_4.jpg",
        PYTORCH_BACKEND_API_BASE,
      ),
    },
    {
      file: "test_original/103_5.jpg",
      class: "digit 5",
      src: getBackendImageUrl(
        "test_original/103_5.jpg",
        PYTORCH_BACKEND_API_BASE,
      ),
    },
    {
      file: "test_original/101_6.jpg",
      class: "digit 6",
      src: getBackendImageUrl(
        "test_original/101_6.jpg",
        PYTORCH_BACKEND_API_BASE,
      ),
    },
    {
      file: "test_original/129_8.jpg",
      class: "digit 8",
      src: getBackendImageUrl(
        "test_original/129_8.jpg",
        PYTORCH_BACKEND_API_BASE,
      ),
    },
    {
      file: "test_original/100_9.jpg",
      class: "digit 9",
      src: getBackendImageUrl(
        "test_original/100_9.jpg",
        PYTORCH_BACKEND_API_BASE,
      ),
    },
  ];
  const getImageOptionsForBackendModel = (modelSpec) => {
    if (
      modelSpec?.family === "tiny-vgg" ||
      modelSpec?.id?.startsWith("tiny-vgg-")
    ) {
      return tinyVggImageOptions;
    }

    return pytorchDigitImageOptions;
  };

  const setBackendImageOptionsForModel = (modelSpec) => {
    let nextImageOptions = getImageOptionsForBackendModel(modelSpec);
    imageOptions = nextImageOptions;

    if (
      imageOptions.length &&
      !imageOptions.some(
        (image) =>
          image.file === selectedImage || image.backendPath === selectedImage,
      )
    ) {
      selectedImage = imageOptions[0].file;
    }
  };

  let imageOptions =
    ACTIVE_MODEL_RUNTIME === RUNTIME_PYTORCH_BACKEND
      ? ENABLE_AI_TEST_UI
        ? tinyVggImageOptions
        : pytorchDigitImageOptions
      : tensorflowImageOptions;
  let selectedImage =
    ACTIVE_MODEL_RUNTIME === RUNTIME_PYTORCH_BACKEND
      ? DEFAULT_TINY_VGG_IMAGE_FILE
      : imageOptions[6].file;
  // Task 3 at lines 260-263: model selector state for switching among the
  // trained 7-layer, 12-layer, and 17-layer demonstrations in the same UI.
  let modelOptions = [];
  let selectedModelId = "";
  let selectedModelSpec = undefined;
  let isModelLoading = false;

  let nodeData;
  let selectedNodeIndex = -1;
  let isExitedFromDetailedView = true;
  let isExitedFromCollapse = true;
  let customImageURL = null;
  let hasRenderedOverview = false;

  // Task 3 at lines 274-296: bundled fallback model list so the selector still
  // works even if the external model manifest is missing or out of date.
  //Task 4 at lines 275-290: update the fallback model list to include the metadata needed for dynamic legend generation and layer display, so that even in the fallback scenario users can benefit from the dynamic features instead of seeing a static architecture diagram.
  const fallbackModelOptions = [
    {
      id: "compact-7",
      label: "7-layer",
      modelPath: "assets/data/models/compact-7/model.json",
    },
    {
      id: "balanced-12",
      label: "12-layer",
      modelPath: "assets/data/models/balanced-12/model.json",
    },
    {
      id: "deep-17",
      label: "17-layer",
      modelPath: "assets/data/models/deep-17/model.json",
    },
  ];

  const getCurrentInputSource = () => {
    if (ACTIVE_MODEL_RUNTIME === RUNTIME_PYTORCH_BACKEND) {
      let selectedOption = imageOptions.find(
        (image) => image.file === selectedImage,
      );
      return (
        selectedOption?.backendPath || selectedOption?.file || selectedImage
      );
    }

    if (selectedImage === "custom" && customImageURL) {
      return customImageURL;
    }

    return `assets/img/${selectedImage}`;
  };

  const normalizeVisualizationImageName = (imageName) => {
    if (!imageName) {
      return "";
    }

    let matchingOption = imageOptions.find(
      (image) => image.file === imageName || image.backendPath === imageName,
    );
    if (matchingOption) {
      return matchingOption.file;
    }

    let normalizedName = imageName.replace(/^\.?\//, "");
    matchingOption = imageOptions.find(
      (image) =>
        image.file === normalizedName || image.backendPath === normalizedName,
    );
    if (matchingOption) {
      return matchingOption.file;
    }

    if (normalizedName.startsWith("assets/img/")) {
      let fileName = normalizedName.slice("assets/img/".length);
      matchingOption = imageOptions.find((image) => image.file === fileName);
      if (matchingOption) {
        return matchingOption.file;
      }
    }

    return imageName;
  };

  const syncCustomImagePreview = () => {
    if (ACTIVE_MODEL_RUNTIME === RUNTIME_PYTORCH_BACKEND) {
      return;
    }

    if (selectedImage !== "custom" || !customImageURL || !cnn) {
      return;
    }

    let customImageSlot = d3
      .select(overviewComponent)
      .select(".custom-image")
      .node();

    if (customImageSlot) {
      drawCustomImage(customImageSlot, cnn[0]);
    }
  };

  const getDenseWeightMatrix = (layerNodes) => {
    return layerNodes.map((node) => node.inputLinks.map((link) => link.weight));
  };

  const getDenseBiasVector = (layerNodes) => {
    return layerNodes.map((node) => node.bias || 0);
  };

  const getDenseActivationMask = (layerNodes) => {
    let activationName = layerNodes[0]?.activationName || "linear";
    if (activationName === "relu") {
      return layerNodes.map((node) => (node.output > 0 ? 1 : 0));
    }

    return layerNodes.map(() => 1);
  };

  const multiplyMatrixByMaskedRows = (leftMatrix, rightMatrix, rowMask) => {
    let result = Array.from({ length: leftMatrix.length }, () =>
      Array(rightMatrix[0].length).fill(0),
    );

    for (let row = 0; row < leftMatrix.length; row++) {
      for (let inner = 0; inner < rightMatrix.length; inner++) {
        let leftValue = leftMatrix[row][inner] * rowMask[inner];
        if (leftValue === 0) {
          continue;
        }

        for (let col = 0; col < rightMatrix[0].length; col++) {
          result[row][col] += leftValue * rightMatrix[inner][col];
        }
      }
    }

    return result;
  };

  const multiplyMatrixVectorWithMask = (matrix, vector, mask) => {
    return matrix.map((row) =>
      row.reduce(
        (sum, value, index) => sum + value * vector[index] * mask[index],
        0,
      ),
    );
  };

  const buildClassifierHeadSummary = (constructedCnn) => {
    let flattenIndex = constructedCnn.findIndex(
      (layer) => layer[0]?.type === "flatten",
    );
    if (flattenIndex === -1) {
      return undefined;
    }

    let outputIndex = constructedCnn.length - 1;
    let flattenLayer = constructedCnn[flattenIndex];
    let denseLayers = constructedCnn.slice(flattenIndex + 1, outputIndex);
    let outputLayer = constructedCnn[outputIndex];

    let effectiveWeights = getDenseWeightMatrix(outputLayer);
    let effectiveBias = getDenseBiasVector(outputLayer);

    for (
      let denseIndex = denseLayers.length - 1;
      denseIndex >= 0;
      denseIndex--
    ) {
      let denseLayer = denseLayers[denseIndex];
      let denseWeights = getDenseWeightMatrix(denseLayer);
      let denseBias = getDenseBiasVector(denseLayer);
      let denseMask = getDenseActivationMask(denseLayer);

      let nextBiasContribution = multiplyMatrixVectorWithMask(
        effectiveWeights,
        denseBias,
        denseMask,
      );

      effectiveBias = effectiveBias.map(
        (value, index) => value + nextBiasContribution[index],
      );
      effectiveWeights = multiplyMatrixByMaskedRows(
        effectiveWeights,
        denseWeights,
        denseMask,
      );
    }

    return {
      flatten: flattenLayer,
      denseLayers,
      output: outputLayer,
      effectiveWeights,
      effectiveBias,
    };
  };

  const buildOverviewOutputLayer = (
    outputLayer,
    prevFeatureLayer,
    headSummary,
  ) => {
    let flattenLength = Math.max(
      1,
      Math.floor(headSummary.flatten.length / prevFeatureLayer.length),
    );

    return outputLayer.map((outputNode, outputIndex) => {
      let overviewNode = {
        ...outputNode,
        bias: headSummary.effectiveBias[outputIndex],
        inputLinks: [],
        outputLinks: [],
      };

      prevFeatureLayer.forEach((sourceNode, featureIndex) => {
        let start = featureIndex * flattenLength;
        let end = Math.min(start + flattenLength, headSummary.flatten.length);
        let channelWeights = headSummary.effectiveWeights[outputIndex].slice(
          start,
          end,
        );
        let averageAbsWeight =
          channelWeights.reduce((sum, value) => sum + Math.abs(value), 0) /
          Math.max(channelWeights.length, 1);

        overviewNode.inputLinks.push({
          source: sourceNode,
          dest: overviewNode,
          weight: averageAbsWeight,
          isOverviewSynthetic: true,
        });
      });

      return overviewNode;
    });
  };

  const buildFlattenViewLayer = (headSummary, overviewOutputLayer) => {
    return headSummary.flatten.map((flattenNode, flattenIndex) => {
      let viewNode = {
        ...flattenNode,
        inputLinks: [...flattenNode.inputLinks],
        outputLinks: [],
      };

      overviewOutputLayer.forEach((outputNode, outputIndex) => {
        viewNode.outputLinks.push({
          source: viewNode,
          dest: outputNode,
          weight: headSummary.effectiveWeights[outputIndex][flattenIndex],
          isClassifierSynthetic: true,
        });
      });

      return viewNode;
    });
  };

  const updateVisibleCnn = (constructedCnn) => {
    let headSummary = buildClassifierHeadSummary(constructedCnn);

    if (!headSummary || !headSummary.denseLayers.length) {
      let visibleCnn = [...constructedCnn];
      let flatten = visibleCnn[visibleCnn.length - 2];
      visibleCnn.splice(visibleCnn.length - 2, 1);
      visibleCnn.flatten = flatten;

      cnn = visibleCnn;
      cnnStore.set(cnn);
      updateCNNLayerRanges();
      return;
    }

    let flattenIndex = constructedCnn.findIndex(
      (layer) => layer[0]?.type === "flatten",
    );
    let prevFeatureLayer = constructedCnn[flattenIndex - 1];
    let overviewOutputLayer = buildOverviewOutputLayer(
      headSummary.output,
      prevFeatureLayer,
      headSummary,
    );
    let flattenViewLayer = buildFlattenViewLayer(
      headSummary,
      overviewOutputLayer,
    );
    let visibleCnn = [
      ...constructedCnn.slice(0, flattenIndex),
      overviewOutputLayer,
    ];
    visibleCnn.flatten = flattenViewLayer;
    visibleCnn.classifierHead = {
      ...headSummary,
      flattenView: flattenViewLayer,
      outputView: overviewOutputLayer,
    };

    cnn = visibleCnn;
    cnnStore.set(cnn);
    updateCNNLayerRanges();
  };

  const reconstructCnnForCurrentInput = async ({ redraw = true } = {}) => {
    // Task 3 at lines 335-340: one shared refresh path for normal image
    // changes, custom-image changes, and model switches, so every redraw uses
    // the currently selected model architecture.
    // PYTORCH_BACKEND_INTEGRATION:
    // Active path calls Flask `/api/explain` and adapts the response to the
    // original frontend CNN shape. The original TensorFlow.js call is still
    // below in the fallback branch.
    let constructedCnn;
    if (ACTIVE_MODEL_RUNTIME === RUNTIME_PYTORCH_BACKEND) {
      let explanation = await explainWithBackend(
        selectedModelId,
        getCurrentInputSource(),
        PYTORCH_BACKEND_API_BASE,
      );
      constructedCnn = constructCNNFromBackendExplanation(explanation);
      if (selectedModelSpec) {
        selectedModelSpec = {
          ...selectedModelSpec,
          lastPrediction: explanation.prediction,
        };
      }
    } else {
      // TensorFlow.js fallback:
      // To return to browser-only TensorFlow.js, set ACTIVE_MODEL_RUNTIME to
      // RUNTIME_TENSORFLOW_JS and this original code path will run again.
      constructedCnn = await constructCNN(getCurrentInputSource(), model);
    }
    updateVisibleCnn(constructedCnn);
    syncCustomImagePreview();

    if (svg && redraw) {
      redrawOverviewLayout();
    }
  };
  //Task 4 at lines 340-418: derive the metadata needed for dynamic legend generation and layer display from the model JSON, so that the overview can adapt to different architectures without hardcoded assumptions.
  const deriveModelMetadata = (modelJson) => {
    let rawLayers =
      modelJson?.modelTopology?.model_config?.config?.layers || [];

    let visibleLayers = rawLayers.filter(
      (layer) => layer.class_name !== "InputLayer",
    );

    let counts = visibleLayers.reduce(
      (acc, layer) => {
        let className = layer.class_name;
        if (className === "Conv2D") acc.conv += 1;
        if (className === "Dense") acc.dense += 1;
        if (className === "Activation" && layer.config?.activation === "relu")
          acc.relu += 1; // Task 6.2
        if (
          className === "Activation" &&
          layer.config?.activation === "sigmoid"
        )
          acc.sigmoid += 1;
        if (className === "MaxPooling2D") acc.maxPool += 1;
        if (className === "AveragePooling2D") acc.avgPool += 1;
        return acc;
      },
      { conv: 0, dense: 0, relu: 0, sigmoid: 0, maxPool: 0, avgPool: 0 },
    );

    let summaryParts = [];
    if (counts.conv) summaryParts.push(`${counts.conv} conv`);
    if (counts.dense)
      summaryParts.push(`${Math.max(counts.dense - 1, 0)} hidden dense`);
    if (counts.relu) summaryParts.push(`${counts.relu} relu`);
    // Task 6.2
    if (counts.sigmoid) summaryParts.push(`${counts.sigmoid} sigmoid`);
    if (counts.maxPool) summaryParts.push(`${counts.maxPool} max-pool`);
    if (counts.avgPool) summaryParts.push(`${counts.avgPool} avg-pool`);
    // Task 6.1
    let convFilters = [
      ...new Set(
        visibleLayers
          .filter((layer) => layer.class_name === "Conv2D")
          .map((layer) => layer.config?.filters)
          .filter((filters) => filters !== undefined),
      ),
    ];

    let stageBoundaries = [];
    visibleLayers.forEach((layer, index) => {
      if (
        layer.class_name === "MaxPooling2D" ||
        layer.class_name === "AveragePooling2D"
      ) {
        stageBoundaries.push(index);
      }
    });

    return {
      totalLayers: visibleLayers.length,
      visibleLayers: visibleLayers.map(
        (layer) => layer.config?.name || layer.class_name,
      ),
      stageBoundaries,
      counts,
      convFilters,
      convKernelSummary: convFilters.length
        ? `${convFilters.join("/")} kernels`
        : "",
      summary: summaryParts.join(", "),
    };
  };

  const enrichModelOption = async (option) => {
    try {
      let response = await fetch(option.modelPath);
      if (!response.ok) {
        throw new Error(
          `Model metadata request failed with ${response.status}`,
        );
      }

      let modelJson = await response.json();
      return {
        ...option,
        ...deriveModelMetadata(modelJson),
      };
    } catch (error) {
      console.warn(
        `[Overview] Failed to derive metadata for ${option.modelPath}`,
        error,
      );
      return {
        ...option,
        totalLayers: undefined,
        visibleLayers: [],
        stageBoundaries: [],
        // Task 6.2
        counts: {
          conv: 0,
          dense: 0,
          relu: 0,
          sigmoid: 0,
          maxPool: 0,
          avgPool: 0,
        },
        convFilters: [],
        convKernelSummary: "",
        summary: "",
      };
    }
  };

  const loadTensorFlowModelManifest = async () => {
    try {
      let response = await fetch("assets/data/model-index.json");
      if (!response.ok) {
        throw new Error(`Manifest request failed with ${response.status}`);
      }

      let loadedOptions = await response.json();
      let baseOptions = loadedOptions.length
        ? loadedOptions
        : fallbackModelOptions;
      return Promise.all(baseOptions.map(enrichModelOption));
    } catch (error) {
      console.warn("[Overview] Falling back to bundled model list", error);
      return Promise.all(fallbackModelOptions.map(enrichModelOption));
    }
  };

  const loadBackendModelManifest = async () => {
    try {
      return await loadBackendModelOptions(PYTORCH_BACKEND_API_BASE);
    } catch (error) {
      console.warn(
        "[Overview] Failed to load PyTorch backend model list",
        error,
      );
      throw error;
    }
  };

  const loadModelManifest = async () => {
    // PYTORCH_BACKEND_INTEGRATION:
    // Active backend path. To restore TensorFlow.js, switch ACTIVE_MODEL_RUNTIME
    // at the top of this file; do not delete either branch.
    if (ACTIVE_MODEL_RUNTIME === RUNTIME_PYTORCH_BACKEND) {
      return loadBackendModelManifest();
    }

    return loadTensorFlowModelManifest();
  };

  const getDefaultAiVisualizationModelId = () =>
    modelOptions.find(
      (option) => option.id === DEFAULT_AI_VISUALIZATION_MODEL_ID,
    )?.id ||
    modelOptions.find((option) => option.family === "tiny-vgg")?.id ||
    modelOptions.find((option) => option.id === DEFAULT_AI_TEST_UI_MODEL_ID)
      ?.id ||
    modelOptions[0]?.id;

  const loadModelById = async (modelId, { redraw = true } = {}) => {
    let nextModel = modelOptions.find((option) => option.id === modelId);
    if (!nextModel) {
      return;
    }

    isModelLoading = true;
    disableControl = true;

    try {
      if (model && typeof model.dispose === "function") {
        model.dispose();
      }

      // Task 3 at lines 375-378: load the selected trained model and
      // immediately rebuild the visible CNN graph for the current input image.
      //   model = await loadTrainedModel(nextModel.modelPath);
      //   selectedModelId = nextModel.id;
      //   selectedModelSpec = nextModel;
      //   await reconstructCnnForCurrentInput({ redraw });
      // } finally {
      //   isModelLoading = false;
      //   disableControl = false;
      // Task 5 at line 454-463
      if (ACTIVE_MODEL_RUNTIME === RUNTIME_PYTORCH_BACKEND) {
        // PYTORCH_BACKEND_INTEGRATION:
        // Backend models are loaded by Flask/PyTorch, so the frontend keeps no
        // TensorFlow.js model object. Comment this branch and uncomment/use the
        // TensorFlow.js branch below only if you switch ACTIVE_MODEL_RUNTIME.
        model = null;
        selectedModelId = nextModel.id;
        selectedModelSpec = nextModel;
        syncClassLabels(nextModel.classLabels);
        setBackendImageOptionsForModel(nextModel);
        selectedLayerOrder = "model";
        layerDisplayOrderStore.set(selectedLayerOrder);
        await reconstructCnnForCurrentInput({ redraw });
      } else {
        // TensorFlow.js fallback:
        // This is the original browser-side model loading path.
        model = await loadTrainedModel(nextModel.modelPath); // loads assets/data/models/balanced-12/model.json  This calls the loader in cnn-tf.js  That loader: reads model.json and fetches .bin weights
        selectedModelId = nextModel.id; //selectedModelId = "balanced-12" //
        selectedModelSpec = nextModel;
        syncClassLabels(tensorflowClassLabels);
        selectedLayerOrder = "model";
        layerDisplayOrderStore.set(selectedLayerOrder);
        await reconstructCnnForCurrentInput({ redraw });
      }
    } finally {
      isModelLoading = false;
      disableControl = false;
    }
  };

  // Helper functions
  const selectedScaleLevelChanged = () => {
    if (svg !== undefined) {
      if (!scaleLevelSet.add(selectedScaleLevel)) {
        console.error("Encounter unknown scale level!");
      }

      // Update nodes and legends
      if (selectedScaleLevel != previousSelectedScaleLevel) {
        // We can simply redraw all nodes using the new color scale, or we can
        // make it faster by only redraw certian nodes
        let updatingLayerIndex = Array.from(
          { length: visibleLayerCount - 2 },
          (_, index) => index + 1,
        );

        updatingLayerIndex.forEach((l) => {
          let range = cnnLayerRanges[selectedScaleLevel][l];
          svg
            .select(`#cnn-layer-group-${l}`)
            .selectAll(".node-image")
            .each((d, i, g) => drawOutput(d, i, g, range));
        });

        // Hide previous legend
        svg
          .selectAll(`.${previousSelectedScaleLevel}-legend`)
          .classed("hidden", true);

        // Show selected legends
        svg
          .selectAll(`.${selectedScaleLevel}-legend`)
          .classed("hidden", !detailedMode);
      }
      previousSelectedScaleLevel = selectedScaleLevel;
      selectedScaleLevelStore.set(selectedScaleLevel);
    }
  };

  const intermediateNodeMouseOverHandler = (d, i, g) => {
    if (detailedViewNum !== undefined) {
      return;
    }
    svg.select(`rect#underneath-gateway-${d.index}`).style("opacity", 1);
  };

  const intermediateNodeMouseLeaveHandler = (d, i, g) => {
    // screenshot
    // return;
    if (detailedViewNum !== undefined) {
      return;
    }
    svg.select(`rect#underneath-gateway-${d.index}`).style("opacity", 0);
  };

  const intermediateNodeClicked = (d, i, g, selectedI, curLayerIndex) => {
    d3.event.stopPropagation();
    isExitedFromCollapse = false;
    // Use this event to trigger the detailed view
    if (detailedViewNum === d.index) {
      // Setting this for testing purposes currently.
      selectedNodeIndex = -1;
      // User clicks this node again -> rewind
      detailedViewNum = undefined;
      svg.select(`rect#underneath-gateway-${d.index}`).style("opacity", 0);
    }
    // We need to show a new detailed view (two cases: if we need to close the
    // old detailed view or not)
    else {
      // Setting this for testing purposes currently.
      selectedNodeIndex = d.index;
      let inputMatrix = d.output;
      let kernelMatrix = d.outputLinks[selectedI].weight;
      // let interMatrix = singleConv(inputMatrix, kernelMatrix);
      let colorScale = layerColorScales.conv;

      // Compute the color range
      let rangePre = cnnLayerRanges[selectedScaleLevel][curLayerIndex - 1];
      let rangeCur = cnnLayerRanges[selectedScaleLevel][curLayerIndex];
      let range = Math.max(rangePre, rangeCur);

      // User triggers a different detailed view
      if (detailedViewNum !== undefined) {
        // Change the underneath highlight
        svg
          .select(`rect#underneath-gateway-${detailedViewNum}`)
          .style("opacity", 0);
        svg.select(`rect#underneath-gateway-${d.index}`).style("opacity", 1);
      }

      // Dynamically position the detail view
      let wholeSvg = d3.select("#cnn-svg");
      let svgYMid = +wholeSvg.style("height").replace("px", "") / 2;
      let svgWidth = +wholeSvg.style("width").replace("px", "");
      let detailViewTop = svgYMid - 250 / 2;
      let positionX =
        intermediateLayerPosition[cnn[curLayerIndex][0].layerName];

      let posX = 0;
      if (curLayerIndex > Math.floor((visibleLayerCount - 1) / 2)) {
        posX = (positionX - svgPaddings.left) / 2;
        posX = svgPaddings.left + posX - 486 / 2;
      } else {
        posX = (svgWidth + svgPaddings.right - positionX) / 2;
        posX = positionX + posX - 486 / 2;
      }

      positionFloatingDetailView(detailViewTop, posX);

      detailedViewNum = d.index;

      // Send the currently used color range to detailed view
      nodeData.colorRange = range;
      nodeData.inputIsInputLayer = curLayerIndex <= 1;
    }
  };

  // The order of the if/else statements in this function is very critical
  const emptySpaceClicked = () => {
    if (closeDenseNeuronDetail(true)) {
      return;
    }

    // If detail view -> rewind to intermediate view
    if (detailedViewNum !== undefined) {
      // Setting this for testing purposes currently.
      selectedNodeIndex = -1;
      // User clicks this node again -> rewind
      svg
        .select(`rect#underneath-gateway-${detailedViewNum}`)
        .style("opacity", 0);
      detailedViewNum = undefined;
    }

    // If softmax view -> rewind to flatten layer view
    else if (isInSoftmax) {
      svg.select(".softmax-symbol").dispatch("click");
    }

    // If intermediate view -> rewind to overview
    else if (isInIntermediateView) {
      let curLayerIndex = layerIndexDict[selectedNode.layerName];
      quitIntermediateView(curLayerIndex, selectedNode.domG, selectedNode.domI);
      d3.select(selectedNode.domG[selectedNode.domI]).dispatch("mouseleave");
    }

    // If pool/act detail view -> rewind to overview
    else if (isInActPoolDetailView) {
      quitActPoolDetailView();
    }
  };

  const prepareToEnterIntermediateView = (d, g, i, curLayerIndex) => {
    isInIntermediateView = true;
    applyCurrentEdgeVisibility(() => false);
    // Hide all legends
    svg.selectAll(`.${selectedScaleLevel}-legend`).classed("hidden", true);
    svg.selectAll(".input-legend").classed("hidden", true);
    svg.selectAll(".output-legend").classed("hidden", true);

    // Hide the input annotation
    svg.select(".input-annotation").classed("hidden", true);

    // Highlight the previous layer and this node
    svg
      .select(`g#cnn-layer-group-${curLayerIndex - 1}`)
      .selectAll("rect.bounding")
      .style("stroke-width", 2);

    d3.select(g[i]).select("rect.bounding").style("stroke-width", 2);

    // Disable control panel UI
    // d3.select('#level-select').property('disabled', true);
    // d3.selectAll('.image-container')
    //   .style('cursor', 'not-allowed')
    //   .on('mouseclick', () => {});
    disableControl = true;

    // Allow infinite animation loop
    shouldIntermediateAnimateStore.set(true);

    // Highlight the labels
    svg
      .selectAll(
        `g#layer-label-${curLayerIndex - 1},
      g#layer-detailed-label-${curLayerIndex - 1},
      g#layer-label-${curLayerIndex},
      g#layer-detailed-label-${curLayerIndex}`,
      )
      .style("font-weight", "800");

    // Register a handler on the svg element so user can click empty space to quit
    // the intermediate view
    d3.select("#cnn-svg").on("click", emptySpaceClicked);
  };

  const quitActPoolDetailView = () => {
    isInActPoolDetailView = false;
    actPoolDetailViewNodeIndex = -1;

    let layerIndex = layerIndexDict[selectedNode.layerName];
    let nodeIndex = selectedNode.index;
    svg
      .select(`g#layer-${layerIndex}-node-${nodeIndex}`)
      .select("rect.bounding")
      .classed("hidden", true);

    selectedNode.data.inputLinks.forEach((link) => {
      let layerIndex = layerIndexDict[link.source.layerName];
      let nodeIndex = link.source.index;
      svg
        .select(`g#layer-${layerIndex}-node-${nodeIndex}`)
        .select("rect.bounding")
        .classed("hidden", true);
    });

    // Clean up the underneath rects
    svg.select("g.underneath").selectAll("rect").remove();

    applyCurrentEdgeVisibility();

    // Recover control UI
    disableControl = false;

    // Show legends if in detailed mode
    svg
      .selectAll(`.${selectedScaleLevel}-legend`)
      .classed("hidden", !detailedMode);
    svg.selectAll(".input-legend").classed("hidden", !detailedMode);
    svg.selectAll(".output-legend").classed("hidden", !detailedMode);

    // Also dehighlight the edge
    let edgeGroup = svg.select("g.cnn-group").select("g.edge-group");
    edgeGroup
      .selectAll(`path.edge-${layerIndex}-${nodeIndex}`)
      .transition()
      .ease(d3.easeCubicOut)
      .duration(200)
      .style("stroke", edgeInitColor)
      .style("stroke-width", edgeStrokeWidth)
      .style("opacity", edgeOpacity);

    // Remove the overlay rect
    svg
      .selectAll(
        "g.intermediate-layer-overlay, g.intermediate-layer-annotation",
      )
      .transition("remove")
      .duration(500)
      .ease(d3.easeCubicInOut)
      .style("opacity", 0)
      .on("end", (d, i, g) => {
        svg
          .selectAll(
            "g.intermediate-layer-overlay, g.intermediate-layer-annotation",
          )
          .remove();
        svg.selectAll("defs.overlay-gradient").remove();
        svg.select(".input-annotation").classed("hidden", false);
      });

    // Turn the fade out nodes back
    svg
      .select(`g#cnn-layer-group-${layerIndex}`)
      .selectAll("g.node-group")
      .each((sd, si, sg) => {
        d3.select(sg[si]).style("pointer-events", "all");
      });

    svg
      .select(`g#cnn-layer-group-${layerIndex - 1}`)
      .selectAll("g.node-group")
      .each((sd, si, sg) => {
        // Recover the old events
        d3.select(sg[si])
          .style("pointer-events", "all")
          .on("mouseover", nodeMouseOverHandler)
          .on("mouseleave", nodeMouseLeaveHandler)
          .on("click", nodeClickHandler);
      });

    // Deselect the node
    selectedNode.layerName = "";
    selectedNode.index = -1;
    selectedNode.data = null;

    actPoolDetailViewLayerIndex = -1;
  };

  const actPoolDetailViewPreNodeMouseOverHandler = (d, i, g) => {
    // Highlight the edges
    let layerIndex = layerIndexDict[d.layerName];
    let nodeIndex = d.index;
    let edgeGroup = svg.select("g.cnn-group").select("g.edge-group");

    edgeGroup
      .selectAll(`path.edge-${actPoolDetailViewLayerIndex}-${nodeIndex}`)
      .raise()
      .transition()
      .ease(d3.easeCubicInOut)
      .duration(400)
      .style("stroke", edgeHoverColor)
      .style("stroke-width", "1")
      .style("opacity", 1);

    // Highlight its border
    d3.select(g[i]).select("rect.bounding").classed("hidden", false);

    // Highlight node's pair
    let associatedLayerIndex = layerIndex - 1;
    if (layerIndex === actPoolDetailViewLayerIndex - 1) {
      associatedLayerIndex = layerIndex + 1;
    }

    svg
      .select(`g#layer-${associatedLayerIndex}-node-${nodeIndex}`)
      .select("rect.bounding")
      .classed("hidden", false);
  };

  const actPoolDetailViewPreNodeMouseLeaveHandler = (d, i, g) => {
    // De-highlight the edges
    let layerIndex = layerIndexDict[d.layerName];
    let nodeIndex = d.index;
    let edgeGroup = svg.select("g.cnn-group").select("g.edge-group");

    edgeGroup
      .selectAll(`path.edge-${actPoolDetailViewLayerIndex}-${nodeIndex}`)
      .transition()
      .ease(d3.easeCubicOut)
      .duration(200)
      .style("stroke", edgeInitColor)
      .style("stroke-width", edgeStrokeWidth)
      .style("opacity", edgeOpacity);

    // De-highlight its border
    d3.select(g[i]).select("rect.bounding").classed("hidden", true);

    // De-highlight node's pair
    let associatedLayerIndex = layerIndex - 1;
    if (layerIndex === actPoolDetailViewLayerIndex - 1) {
      associatedLayerIndex = layerIndex + 1;
    }

    svg
      .select(`g#layer-${associatedLayerIndex}-node-${nodeIndex}`)
      .select("rect.bounding")
      .classed("hidden", true);
  };

  const actPoolDetailViewPreNodeClickHandler = (d, i, g) => {
    let layerIndex = layerIndexDict[d.layerName];
    let nodeIndex = d.index;

    // Click the pre-layer node in detail view has the same effect as clicking
    // the cur-layer node, which is to open a new detail view window
    svg
      .select(`g#layer-${layerIndex + 1}-node-${nodeIndex}`)
      .node()
      .dispatchEvent(new Event("click"));
  };

  const enterDetailView = (curLayerIndex, i) => {
    isInActPoolDetailView = true;
    actPoolDetailViewNodeIndex = i;
    actPoolDetailViewLayerIndex = curLayerIndex;

    // Dynamically position the detail view
    let wholeSvg = d3.select("#cnn-svg");
    let svgYMid = +wholeSvg.style("height").replace("px", "") / 2;
    let svgWidth = +wholeSvg.style("width").replace("px", "");
    let detailViewTop = svgYMid - 260 / 2;

    let posX = 0;
    if (curLayerIndex > Math.floor((visibleLayerCount - 1) / 2)) {
      posX = nodeCoordinate[curLayerIndex - 1][0].x + 50;
      posX = posX / 2 - 500 / 2;
    } else {
      posX = (svgWidth - nodeCoordinate[curLayerIndex][0].x - nodeLength) / 2;
      posX = nodeCoordinate[curLayerIndex][0].x + nodeLength + posX - 500 / 2;
    }

    positionFloatingDetailView(detailViewTop, posX);

    applyCurrentEdgeVisibility(
      (edgeDatum, isVisibleByReveal) =>
        isVisibleByReveal && edgeDatum.targetLayerIndex === curLayerIndex,
    );

    // Disable UI
    disableControl = true;

    // Hide input annotaitons
    svg.select(".input-annotation").classed("hidden", true);

    // Hide legends
    svg.selectAll(`.${selectedScaleLevel}-legend`).classed("hidden", true);
    svg.selectAll(".input-legend").classed("hidden", true);
    svg.selectAll(".output-legend").classed("hidden", true);
    svg
      .select(`#${layerLegendDict[curLayerIndex][selectedScaleLevel]}`)
      .classed("hidden", false);

    // Add overlay rects
    let leftX = nodeCoordinate[curLayerIndex - 1][i].x;
    // +5 to cover the detailed mode long label
    let rightStart = nodeCoordinate[curLayerIndex][i].x + nodeLength + 5;

    // Compute the left and right overlay rect width
    let rightWidth = width - rightStart - overlayRectOffset / 2;
    let leftWidth = leftX - nodeCoordinate[0][0].x;

    // The overlay rects should be symmetric
    if (rightWidth > leftWidth) {
      let stops = [
        { offset: "0%", color: "rgb(250, 250, 250)", opacity: 0.85 },
        { offset: "50%", color: "rgb(250, 250, 250)", opacity: 0.9 },
        { offset: "100%", color: "rgb(250, 250, 250)", opacity: 1 },
      ];
      addOverlayGradient("overlay-gradient-right", stops);

      let leftEndOpacity = 0.85 + (0.95 - 0.85) * (leftWidth / rightWidth);
      stops = [
        { offset: "0%", color: "rgb(250, 250, 250)", opacity: leftEndOpacity },
        { offset: "100%", color: "rgb(250, 250, 250)", opacity: 0.85 },
      ];
      addOverlayGradient("overlay-gradient-left", stops);
    } else {
      let stops = [
        { offset: "0%", color: "rgb(250, 250, 250)", opacity: 1 },
        { offset: "50%", color: "rgb(250, 250, 250)", opacity: 0.9 },
        { offset: "100%", color: "rgb(250, 250, 250)", opacity: 0.85 },
      ];
      addOverlayGradient("overlay-gradient-left", stops);

      let rightEndOpacity = 0.85 + (0.95 - 0.85) * (rightWidth / leftWidth);
      stops = [
        { offset: "0%", color: "rgb(250, 250, 250)", opacity: 0.85 },
        {
          offset: "100%",
          color: "rgb(250, 250, 250)",
          opacity: rightEndOpacity,
        },
      ];
      addOverlayGradient("overlay-gradient-right", stops);
    }

    addOverlayRect(
      "overlay-gradient-right",
      rightStart + overlayRectOffset / 2 + 0.5,
      0,
      rightWidth,
      height + svgPaddings.top,
    );

    addOverlayRect(
      "overlay-gradient-left",
      nodeCoordinate[0][0].x - overlayRectOffset / 2,
      0,
      leftWidth,
      height + svgPaddings.top,
    );

    svg.selectAll("rect.overlay").on("click", emptySpaceClicked);

    // Add underneath rectangles
    let underGroup = svg.select("g.underneath");
    let padding = 7;
    for (let n = 0; n < cnn[curLayerIndex - 1].length; n++) {
      underGroup
        .append("rect")
        .attr("class", "underneath-gateway")
        .attr("id", `underneath-gateway-${n}`)
        .attr("x", nodeCoordinate[curLayerIndex - 1][n].x - padding)
        .attr("y", nodeCoordinate[curLayerIndex - 1][n].y - padding)
        .attr("width", 2 * nodeLength + hSpaceAroundGap + 2 * padding)
        .attr("height", nodeLength + 2 * padding)
        .attr("rx", 10)
        .style("fill", "rgba(160, 160, 160, 0.3)")
        .style("opacity", 0);

      // Update the event functions for these two layers
      svg
        .select(`g#layer-${curLayerIndex - 1}-node-${n}`)
        .style("pointer-events", "all")
        .style("cursor", "pointer")
        .on("mouseover", actPoolDetailViewPreNodeMouseOverHandler)
        .on("mouseleave", actPoolDetailViewPreNodeMouseLeaveHandler)
        .on("click", actPoolDetailViewPreNodeClickHandler);
    }
    underGroup.lower();

    // Highlight the selcted pair
    underGroup.select(`#underneath-gateway-${i}`).style("opacity", 1);
  };

  const resetOverviewEdgeStyles = () => {
    svg
      .select("g.cnn-group")
      .select("g.edge-group")
      .selectAll("path.edge")
      .interrupt()
      .style("stroke", edgeInitColor)
      .style("stroke-width", edgeStrokeWidth)
      .style("stroke-dasharray", null)
      .style("stroke-dashoffset", null);

    applyCurrentEdgeVisibility();
  };

  const quitIntermediateView = (curLayerIndex, g, i) => {
    resetDenseNeuronDetailLayout();

    // If it is the softmax detail view, quit that view first
    if (isInSoftmax) {
      svg.select(".logit-layer").remove();
      svg.select(".logit-layer-lower").remove();
      svg.selectAll(".plus-symbol-clone").remove();

      // Instead of removing the paths, we hide them, so it is faster to load in
      // the future
      svg.select(".underneath").selectAll(".logit-lower").style("opacity", 0);

      softmaxDetailViewStore.set({
        show: false,
        logits: [],
      });

      allowsSoftmaxAnimationStore.set(false);
    }
    isInSoftmaxStore.set(false);
    isInIntermediateView = false;

    // Show the legend
    svg
      .selectAll(`.${selectedScaleLevel}-legend`)
      .classed("hidden", !detailedMode);
    svg.selectAll(".input-legend").classed("hidden", !detailedMode);
    svg.selectAll(".output-legend").classed("hidden", !detailedMode);

    // Recover control panel UI
    disableControl = false;

    // Recover the input layer node's event
    for (let n = 0; n < cnn[curLayerIndex - 1].length; n++) {
      svg
        .select(`g#layer-${curLayerIndex - 1}-node-${n}`)
        .on("mouseover", nodeMouseOverHandler)
        .on("mouseleave", nodeMouseLeaveHandler)
        .on("click", nodeClickHandler);
    }

    // Clean up the underneath rects
    svg.select("g.underneath").selectAll("rect").remove();
    detailedViewNum = undefined;

    // Highlight the previous layer and this node
    svg
      .select(`g#cnn-layer-group-${curLayerIndex - 1}`)
      .selectAll("rect.bounding")
      .style("stroke-width", 1);

    d3.select(g[i]).select("rect.bounding").style("stroke-width", 1);

    // Highlight the labels
    svg
      .selectAll(
        `g#layer-label-${curLayerIndex - 1},
      g#layer-detailed-label-${curLayerIndex - 1},
      g#layer-label-${curLayerIndex},
      g#layer-detailed-label-${curLayerIndex}`,
      )
      .style("font-weight", "normal");

    // Also unclick the node
    // Record the current clicked node
    selectedNode.layerName = "";
    selectedNode.index = -1;
    selectedNode.data = null;
    isExitedFromCollapse = true;

    // Remove the intermediate layer
    let intermediateLayer = svg.select("g.intermediate-layer");

    // Kill the infinite animation loop
    shouldIntermediateAnimateStore.set(false);

    intermediateLayer
      .transition("remove")
      .duration(500)
      .ease(d3.easeCubicInOut)
      .style("opacity", 0)
      .on("end", (d, i, g) => {
        d3.select(g[i]).remove();
      });

    // Remove the output node overlay mask
    svg.selectAll(".overlay-group").remove();

    // Remove the overlay rect
    svg
      .selectAll(
        "g.intermediate-layer-overlay, g.intermediate-layer-annotation",
      )
      .transition("remove")
      .duration(500)
      .ease(d3.easeCubicInOut)
      .style("opacity", 0)
      .on("end", (d, i, g) => {
        svg
          .selectAll(
            "g.intermediate-layer-overlay, g.intermediate-layer-annotation",
          )
          .remove();
        svg.selectAll("defs.overlay-gradient").remove();
      });

    // Recover the layer if we have drdrawn it
    if (needRedraw[0] !== undefined) {
      let redrawRange = cnnLayerRanges[selectedScaleLevel][needRedraw[0]];
      if (needRedraw[1] !== undefined) {
        svg
          .select(`g#layer-${needRedraw[0]}-node-${needRedraw[1]}`)
          .select("image.node-image")
          .each((d, i, g) => drawOutput(d, i, g, redrawRange));
      } else {
        svg
          .select(`g#cnn-layer-group-${needRedraw[0]}`)
          .selectAll("image.node-image")
          .each((d, i, g) => drawOutput(d, i, g, redrawRange));
      }
    }

    // Move all layers to their original place
    for (let i = 0; i < visibleLayerCount; i++) {
      moveLayerX({
        layerIndex: i,
        targetX: nodeCoordinate[i][0].x,
        disable: false,
        delay: 500,
        opacity: 1,
      });
    }

    moveLayerX({
      layerIndex: visibleLayerCount - 2,
      targetX: nodeCoordinate[visibleLayerCount - 2][0].x,
      opacity: 1,
      disable: false,
      delay: 500,
      onEndFunc: () => {
        resetOverviewEdgeStyles();

        // Recover the input annotation
        svg
          .select(".input-annotation")
          .classed("hidden", false)
          .style("opacity", 1);
      },
    });
  };

  const nodeClickHandler = (d, i, g) => {
    d3.event.stopPropagation();
    let nodeIndex = d.index;
    let curLayerIndex = layerIndexDict[d.layerName];

    // Match the original layer-view toggle: clicking the selected conv/output
    // node again exits and removes the intermediate/flatten view.
    if (
      (d.type === "conv" || d.layerName === "output") &&
      isInIntermediateView
    ) {
      quitIntermediateView(curLayerIndex, g, i);
      return;
    }

    // Record the current clicked node
    selectedNode.layerName = d.layerName;
    selectedNode.index = d.index;
    selectedNode.data = d;
    selectedNode.domI = i;
    selectedNode.domG = g;

    // Record data for detailed view.
    if (d.type === "conv" || d.type === "relu" || d.type === "pool") {
      let data = [];
      for (let j = 0; j < d.inputLinks.length; j++) {
        data.push({
          input: d.inputLinks[j].source.output,
          kernel: d.inputLinks[j].weight,
          output: d.inputLinks[j].dest.output,
        });
      }
      let curLayerIndex = layerIndexDict[d.layerName];
      data.colorRange = cnnLayerRanges[selectedScaleLevel][curLayerIndex];
      data.isInputInputLayer = curLayerIndex <= 1;
      nodeData = data;
    }

    if (d.type == "relu" || d.type == "pool") {
      isExitedFromDetailedView = false;
      if (!isInActPoolDetailView) {
        // Enter the act pool detail view
        enterDetailView(curLayerIndex, d.index);
      } else {
        if (d.index === actPoolDetailViewNodeIndex) {
          // Quit the act pool detail view
          quitActPoolDetailView();
        } else {
          // Switch the detail view input to the new clicked pair

          // Remove the previous selection effect
          svg
            .select(
              `g#layer-${curLayerIndex}-node-${actPoolDetailViewNodeIndex}`,
            )
            .select("rect.bounding")
            .classed("hidden", true);

          svg
            .select(
              `g#layer-${curLayerIndex - 1}-node-${actPoolDetailViewNodeIndex}`,
            )
            .select("rect.bounding")
            .classed("hidden", true);

          let edgeGroup = svg.select("g.cnn-group").select("g.edge-group");

          edgeGroup
            .selectAll(
              `path.edge-${curLayerIndex}-${actPoolDetailViewNodeIndex}`,
            )
            .transition()
            .ease(d3.easeCubicOut)
            .duration(200)
            .style("stroke", edgeInitColor)
            .style("stroke-width", edgeStrokeWidth)
            .style("opacity", edgeOpacity);

          let underGroup = svg.select("g.underneath");
          underGroup
            .select(`#underneath-gateway-${actPoolDetailViewNodeIndex}`)
            .style("opacity", 0);

          // Add selection effect on the new selected pair
          svg
            .select(`g#layer-${curLayerIndex}-node-${nodeIndex}`)
            .select("rect.bounding")
            .classed("hidden", false);

          svg
            .select(`g#layer-${curLayerIndex - 1}-node-${nodeIndex}`)
            .select("rect.bounding")
            .classed("hidden", false);

          edgeGroup
            .selectAll(`path.edge-${curLayerIndex}-${nodeIndex}`)
            .raise()
            .transition()
            .ease(d3.easeCubicInOut)
            .duration(400)
            .style("stroke", edgeHoverColor)
            .style("stroke-width", "1")
            .style("opacity", 1);

          underGroup
            .select(`#underneath-gateway-${nodeIndex}`)
            .style("opacity", 1);

          actPoolDetailViewNodeIndex = nodeIndex;
        }
      }
    }

    // Enter the second view (layer-view) when user clicks a conv node
    if (
      (d.type === "conv" || d.layerName === "output") &&
      !isInIntermediateView
    ) {
      prepareToEnterIntermediateView(d, g, nodeIndex, curLayerIndex);

      if (d.type === "conv") {
        getConvDrawHandler(curLayerIndex)(
          curLayerIndex,
          d,
          nodeIndex,
          width,
          height,
          intermediateNodeMouseOverHandler,
          intermediateNodeMouseLeaveHandler,
          intermediateNodeClicked,
        );
      } else if (d.layerName === "output") {
        drawFlatten(curLayerIndex, d, nodeIndex, width, height);
      }
    }
    // Quit the layerview
    else if (
      (d.type === "conv" || d.layerName === "output") &&
      isInIntermediateView
    ) {
      quitIntermediateView(curLayerIndex, g, i);
    }
  };

  const nodeMouseOverHandler = (d, i, g) => {
    // if (isInIntermediateView || isInActPoolDetailView) { return; }
    if (isInIntermediateView) {
      return;
    }

    // Highlight the edges
    let layerIndex = layerIndexDict[d.layerName];
    let nodeIndex = d.index;
    let edgeGroup = svg.select("g.cnn-group").select("g.edge-group");

    edgeGroup
      .selectAll(`path.edge-${layerIndex}-${nodeIndex}`)
      .raise()
      .transition()
      .ease(d3.easeCubicInOut)
      .duration(400)
      .style("stroke", edgeHoverColor)
      .style("stroke-width", "1")
      .style("opacity", 1);

    // Highlight its border
    d3.select(g[i]).select("rect.bounding").classed("hidden", false);

    // Highlight source's border
    if (d.inputLinks.length === 1) {
      let link = d.inputLinks[0];
      let layerIndex = layerIndexDict[link.source.layerName];
      let nodeIndex = link.source.index;
      svg
        .select(`g#layer-${layerIndex}-node-${nodeIndex}`)
        .select("rect.bounding")
        .classed("hidden", false);
    } else {
      svg
        .select(`g#cnn-layer-group-${layerIndex - 1}`)
        .selectAll("g.node-group")
        .selectAll("rect.bounding")
        .classed("hidden", false);
    }

    // Highlight the output text
    if (d.layerName === "output") {
      d3.select(g[i])
        .select(".output-text")
        .style("fill", getOutputLabelBaseColor())
        .style("opacity", 0.8)
        .style("text-decoration", "underline");
    }

    /* Use the following commented code if we have non-linear model
    d.inputLinks.forEach(link => {
      let layerIndex = layerIndexDict[link.source.layerName];
      let nodeIndex = link.source.index;
      svg.select(`g#layer-${layerIndex}-node-${nodeIndex}`)
        .select('rect.bounding')
        .classed('hidden', false);
    });
    */
  };

  const nodeMouseLeaveHandler = (d, i, g) => {
    // Screenshot
    // return;

    if (isInIntermediateView) {
      return;
    }

    // Keep the highlight if user has clicked
    if (
      isInActPoolDetailView ||
      d.layerName !== selectedNode.layerName ||
      d.index !== selectedNode.index
    ) {
      let layerIndex = layerIndexDict[d.layerName];
      let nodeIndex = d.index;
      let edgeGroup = svg.select("g.cnn-group").select("g.edge-group");

      edgeGroup
        .selectAll(`path.edge-${layerIndex}-${nodeIndex}`)
        .transition()
        .ease(d3.easeCubicOut)
        .duration(200)
        .style("stroke", edgeInitColor)
        .style("stroke-width", edgeStrokeWidth)
        .style("opacity", edgeOpacity);

      d3.select(g[i]).select("rect.bounding").classed("hidden", true);

      if (d.inputLinks.length === 1) {
        let link = d.inputLinks[0];
        let layerIndex = layerIndexDict[link.source.layerName];
        let nodeIndex = link.source.index;
        svg
          .select(`g#layer-${layerIndex}-node-${nodeIndex}`)
          .select("rect.bounding")
          .classed("hidden", true);
      } else {
        svg
          .select(`g#cnn-layer-group-${layerIndex - 1}`)
          .selectAll("g.node-group")
          .selectAll("rect.bounding")
          .classed(
            "hidden",
            (d) =>
              d.layerName !== selectedNode.layerName ||
              d.index !== selectedNode.index,
          );
      }

      // Dehighlight the output text
      if (d.layerName === "output") {
        d3.select(g[i])
          .select(".output-text")
          .style("fill", getOutputLabelBaseColor())
          .style("opacity", 0.5)
          .style("text-decoration", "none");
      }

      /* Use the following commented code if we have non-linear model
      d.inputLinks.forEach(link => {
        let layerIndex = layerIndexDict[link.source.layerName];
        let nodeIndex = link.source.index;
        svg.select(`g#layer-${layerIndex}-node-${nodeIndex}`)
          .select('rect.bounding')
          .classed('hidden', true);
      });
      */
    }
  };
  let logits = [-4.28, 2.96, -0.38, 5.24, -7.56, -3.43, 8.63, 2.63, 6.3, 0.68];
  let selectedI = 4;

  onMount(async () => {
    try {
      console.log("[Overview:onMount] start");

      // Create SVG
      wholeSvg = d3.select(overviewComponent).select("#cnn-svg");
      wholeSvg
        .attr("viewBox", `0 0 ${logicalSvgWidth} ${logicalSvgHeight}`)
        .attr("preserveAspectRatio", "xMidYMid meet");
      svg = wholeSvg
        .append("g")
        .attr("class", "main-svg")
        .attr("transform", `translate(${svgPaddings.left}, 0)`);
      svgStore.set(svg);

      width = logicalSvgWidth - svgPaddings.left - svgPaddings.right;
      height = logicalSvgHeight - svgPaddings.top - svgPaddings.bottom;

      console.log("[Overview:onMount] svg size", { width, height });

      let cnnGroup = svg.append("g").attr("class", "cnn-group");

      let underGroup = svg.append("g").attr("class", "underneath");

      let svgYMid = logicalSvgHeight / 2;
      detailedViewAbsCoords = {
        1: [600, 100 + svgYMid - 220 / 2, 490, 290],
        2: [500, 100 + svgYMid - 220 / 2, 490, 290],
        3: [700, 100 + svgYMid - 220 / 2, 490, 290],
        4: [600, 100 + svgYMid - 220 / 2, 490, 290],
        5: [650, 100 + svgYMid - 220 / 2, 490, 290],
        6: [850, 100 + svgYMid - 220 / 2, 490, 290],
        7: [100, 100 + svgYMid - 220 / 2, 490, 290],
        8: [60, 100 + svgYMid - 220 / 2, 490, 290],
        9: [200, 100 + svgYMid - 220 / 2, 490, 290],
        10: [300, 100 + svgYMid - 220 / 2, 490, 290],
      };

      // Define global arrow marker end
      svg
        .append("defs")
        .append("marker")
        .attr("id", "marker")
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 6)
        .attr("refY", 0)
        .attr("markerWidth", 6)
        .attr("markerHeight", 6)
        .attr("orient", "auto")
        .append("path")
        .style("stroke-width", 1.2)
        .style("fill", "gray")
        .style("stroke", "gray")
        .attr("d", "M0,-5L10,0L0,5");

      // Alternative arrow head style for non-interactive annotation
      svg
        .append("defs")
        .append("marker")
        .attr("id", "marker-alt")
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 6)
        .attr("refY", 0)
        .attr("markerWidth", 6)
        .attr("markerHeight", 6)
        .attr("orient", "auto")
        .append("path")
        .style("fill", "none")
        .style("stroke", "gray")
        .style("stroke-width", 2)
        .attr("d", "M-5,-10L10,0L-5,10");

      modelOptions = await loadModelManifest();
      if (!modelOptions.length) {
        throw new Error("No model architectures were found.");
      }

      selectedModelId = getDefaultAiVisualizationModelId();

      console.time("Construct cnn");
      await loadModelById(selectedModelId, { redraw: false });
      console.timeEnd("Construct cnn");
      console.log(
        "[Overview:onMount] visible cnn layers",
        cnn.map((layer) => layer[0].layerName),
      );

      // Create and draw the CNN view
      drawCNN(
        width,
        height,
        cnnGroup,
        nodeMouseOverHandler,
        nodeMouseLeaveHandler,
        nodeClickHandler,
      );
      hasRenderedOverview = true;
      console.log("[Overview:onMount] drawCNN finished");
    } catch (error) {
      console.error("[Overview:onMount] failed", error);
      throw error;
    }
  });

  const detailedButtonClicked = () => {
    detailedMode = !detailedMode;
    detailedModeStore.set(detailedMode);

    if (!isInIntermediateView) {
      // Show the legend
      svg
        .selectAll(`.${selectedScaleLevel}-legend`)
        .classed("hidden", !detailedMode);

      svg.selectAll(".input-legend").classed("hidden", !detailedMode);
      svg.selectAll(".output-legend").classed("hidden", !detailedMode);
    }

    // Switch the layer name
    svg.selectAll(".layer-detailed-label").classed("hidden", !detailedMode);

    svg.selectAll(".layer-label").classed("hidden", detailedMode);

    if (isLayerFocusActive) {
      renderLayerFocusView();
    }
  };

  const wait = (delayMs) =>
    new Promise((resolve) => {
      autoRevealTimer = window.setTimeout(resolve, delayMs);
    });

  const clearAutoRevealTimer = () => {
    if (autoRevealTimer !== undefined) {
      window.clearTimeout(autoRevealTimer);
      autoRevealTimer = undefined;
    }
    isAutoRevealRunning = false;
  };

  const openFlattenViewForCurrentPrediction = () => {
    if (!svg || !cnn || isInIntermediateView) {
      return;
    }

    let outputLayerIndex = cnn.length - 1;
    let predictedIndex = selectedModelSpec?.lastPrediction?.index;
    if (!Number.isInteger(predictedIndex)) {
      let outputLayer = cnn[outputLayerIndex] || [];
      predictedIndex = outputLayer.reduce(
        (bestIndex, node, nodeIndex) =>
          Number(node.output || 0) > Number(outputLayer[bestIndex]?.output || 0)
            ? nodeIndex
            : bestIndex,
        0,
      );
    }

    let outputNode = svg
      .select(`#layer-${outputLayerIndex}-node-${predictedIndex}`)
      .node();

    if (!outputNode) {
      return;
    }

    outputNode.dispatchEvent(
      new MouseEvent("click", { bubbles: true, cancelable: true }),
    );
  };

  const runAutoRevealFromInputToOutput = async ({
    showFlatten = true,
  } = {}) => {
    clearAutoRevealTimer();
    isAutoRevealRunning = true;

    if (isInIntermediateView && selectedNode.data) {
      quitIntermediateView(
        layerIndexDict[selectedNode.layerName],
        selectedNode.domG,
        selectedNode.domI,
      );
      await wait(900);
    }

    resetOverviewLayerReveal();
    await wait(650);

    while (isAutoRevealRunning && canRevealNextOverviewLayer()) {
      revealNextOverviewLayer();
      await wait(1550);
    }

    if (isAutoRevealRunning && showFlatten) {
      await wait(650);
      openFlattenViewForCurrentPrediction();
    }

    clearAutoRevealTimer();
  };

  const manualRevealButtonClicked = () => {
    clearAutoRevealTimer();
    revealNextOverviewLayer();
  };

  onDestroy(() => {
    clearAutoRevealTimer();
  });

  const imageOptionClicked = async (e) => {
    let newImageName = d3.select(e.target).attr("data-imageName");

    await selectVisualizationImage(newImageName);
  };

  const selectVisualizationImage = async (newImageName) => {
    let normalizedImageName = normalizeVisualizationImageName(newImageName);
    if (!normalizedImageName || normalizedImageName === selectedImage) {
      return;
    }

    clearAutoRevealTimer();
    selectedImage = normalizedImageName;
    await reconstructCnnForCurrentInput();
  };

  const handleAiVisualizationImageChange = async (event) => {
    await selectVisualizationImage(event.detail.imageFile);
  };

  const handleAiTestTypeChange = async (event) => {
    isAiVisualizationSelected = event.detail.testType === "visualization";

    if (isAiVisualizationSelected) {
      await tick();
      await reconstructCnnForCurrentInput();
    }
  };

  const handleAiVisualizationModelChange = async (event) => {
    let nextModelId = event.detail.modelId;
    if (!nextModelId || nextModelId === selectedModelId) {
      return;
    }

    clearAutoRevealTimer();
    selectedModelId = nextModelId;
    await loadModelById(nextModelId);
  };

  const handleAiVisualizationAutoReveal = async (event) => {
    await selectVisualizationImage(event.detail.imageFile);
    await runAutoRevealFromInputToOutput({ showFlatten: true });
  };

  const customImageClicked = () => {
    // Case 1: there is no custom image -> show the modal to get user input
    if (customImageURL === null) {
      modalInfo.show = true;
      modalInfo.preImage = selectedImage;
      modalStore.set(modalInfo);
    }

    // Case 2: there is an existing custom image, not the focus -> switch to this image
    else if (selectedImage !== "custom") {
      selectedImage = "custom";
      let fakeEvent = { detail: { url: customImageURL } };
      handleCustomImage(fakeEvent);
    }

    // Case 3: there is an existing custom image, and its the focus -> let user
    // upload a new image
    else {
      modalInfo.show = true;
      modalInfo.preImage = selectedImage;
      modalStore.set(modalInfo);
    }

    if (selectedImage !== "custom" && customImageURL === null) {
      selectedImage = "custom";
    }
  };

  const handleModalCanceled = (event) => {
    // User cancels the modal without a successful image, so we restore the
    // previous selected image as input
    selectedImage = event.detail.preImage;
  };

  const handleCustomImage = async (event) => {
    // User gives a valid image URL
    customImageURL = event.detail.url;
    selectedImage = "custom";
    await reconstructCnnForCurrentInput();
  };

  function handleExitFromDetiledConvView(event) {
    if (event.detail.text) {
      detailedViewNum = undefined;
      svg
        .select(`rect#underneath-gateway-${selectedNodeIndex}`)
        .style("opacity", 0);
      selectedNodeIndex = -1;
    }
  }

  function handleExitFromDetiledPoolView(event) {
    if (event.detail.text) {
      quitActPoolDetailView();
      isExitedFromDetailedView = true;
    }
  }

  function handleExitFromDetiledActivationView(event) {
    if (event.detail.text) {
      quitActPoolDetailView();
      isExitedFromDetailedView = true;
    }
  }

  function handleExitFromDetiledSoftmaxView(event) {
    softmaxDetailViewInfo.show = false;
    softmaxDetailViewStore.set(softmaxDetailViewInfo);
  }

  function handleExitFromDetiledDenseView(event) {
    denseDetailViewStore.set({ show: false });
  }

  function handleKeyActivate(event, handler) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      handler();
    }
  }

  const redrawOverviewLayout = () => {
    if (!svg || !cnn) {
      return;
    }

    svg.selectAll("*").remove();
    nodeCoordinate = [];
    nodeCoordinateStore.set(nodeCoordinate);

    let cnnGroup = svg.append("g").attr("class", "cnn-group");
    svg.append("g").attr("class", "underneath");

    drawCNN(
      width,
      height,
      cnnGroup,
      nodeMouseOverHandler,
      nodeMouseLeaveHandler,
      nodeClickHandler,
    );

    if (isLayerFocusActive) {
      renderLayerFocusView();
    }
  };

  const getLayerSearchLabel = (layerIndex) => {
    let layerName = cnn?.[layerIndex]?.[0]?.layerName || `Layer ${layerIndex + 1}`;
    return `${layerIndex + 1} - ${formatToolbarLayerName(layerName)}`;
  };

  const getLayerFocusWindow = (layerIndex) => {
    if (!cnn?.length) {
      return [];
    }

    let windowSize = Math.min(cnn.length, layerFocusRadius * 2 + 1);
    let start = Math.max(
      0,
      Math.min(layerIndex - layerFocusRadius, cnn.length - windowSize),
    );
    return Array.from({ length: windowSize }, (_, index) => start + index);
  };

  const parseLayerSearchValue = (value) => {
    if (!cnn?.length) {
      return -1;
    }

    let normalizedValue = String(value || "").trim().toLowerCase();
    let numberMatch = normalizedValue.match(/^(\d+)/);
    if (numberMatch) {
      let requestedIndex = Number(numberMatch[1]) - 1;
      if (requestedIndex >= 0 && requestedIndex < cnn.length) {
        return requestedIndex;
      }
    }

    return cnn.findIndex((layer, index) => {
      let layerName = String(layer[0]?.layerName || "").toLowerCase();
      let displayName = formatToolbarLayerName(layerName).toLowerCase();
      let optionLabel = getLayerSearchLabel(index).toLowerCase();
      return (
        normalizedValue === layerName ||
        normalizedValue === displayName ||
        normalizedValue === optionLabel
      );
    });
  };

  const resetLayerFocusPresentation = () => {
    if (!svg) {
      return;
    }

    svg.select("g.layer-focus-context").remove();
    svg
      .selectAll("g.cnn-layer-group")
      .each(function () {
        let layerGroup = d3.select(this);
        let originalOpacity = layerGroup.attr("data-layer-focus-opacity");
        if (originalOpacity !== null) {
          layerGroup
            .style("opacity", originalOpacity || null)
            .attr("data-layer-focus-opacity", null);
        }
      })
      .style("display", null)
      .attr("transform", null)
      .selectAll("g.node-group")
      .on("click", nodeClickHandler);

    svg.selectAll("g.layer-label, g.layer-detailed-label").each(function (
      d,
      selectionIndex,
    ) {
      let label = d3.select(this);
      let layerIndex = Number(String(this.id || "").match(/-(\d+)$/)?.[1]);
      let originalX = nodeCoordinate?.[layerIndex]?.[0]?.x + nodeLength / 2;
      let transform = label.attr("transform") || "";
      let yMatch = transform.match(/translate\([^,]+,\s*([^)]+)\)/);
      let originalY = yMatch?.[1] || 0;
      let originalOpacity = label.attr("data-layer-focus-opacity");
      label
        .style("display", null)
        .attr("transform", `translate(${originalX}, ${originalY})`);
      if (originalOpacity !== null) {
        label
          .style("opacity", originalOpacity || null)
          .attr("data-layer-focus-opacity", null);
      }
    });

    svg
      .selectAll(
        "g.stage-grouping, g.color-legend, g.input-annotation, g.edge-packet-group, g.node-pulse-group",
      )
      .style("display", null);

    let linkGenerator = d3
      .linkHorizontal()
      .x((point) => point.x)
      .y((point) => point.y);
    svg
      .select("g.edge-group")
      .selectAll("path.edge")
      .attr("d", (edge) =>
        linkGenerator({ source: edge.source, target: edge.target }),
      );

    svg
      .selectAll("rect.layer-hover-outline, rect.layer-label-highlight")
      .style("opacity", 0);
    svg
      .selectAll("rect.layer-hover-outline")
      .attr("width", nodeLength + 28);
    svg.selectAll("g.output-winner-overlay").each(function () {
      let overlay = d3.select(this);
      let originalTransform = overlay.attr("data-layer-focus-transform");
      if (originalTransform !== null) {
        overlay
          .attr("transform", originalTransform || null)
          .attr("data-layer-focus-transform", null);
      }
    });
    applyCurrentEdgeVisibility();
  };

  const renderLayerFocusView = () => {
    if (!isLayerFocusActive || !svg || !cnn?.length || !nodeCoordinate?.length) {
      return;
    }

    resetLayerFocusPresentation();
    focusedLayerIndex = Math.max(
      0,
      Math.min(focusedLayerIndex, cnn.length - 1),
    );

    let visibleLayers = getLayerFocusWindow(focusedLayerIndex);
    let visibleLayerSet = new Set(visibleLayers);
    let firstVisibleLayer = visibleLayers[0];
    let lastVisibleLayer = visibleLayers[visibleLayers.length - 1];
    let hasEarlierLayers = firstVisibleLayer > 0;
    let hasLaterLayers = lastVisibleLayer < cnn.length - 1;
    let includesOutputLayer = lastVisibleLayer === cnn.length - 1;
    let leftBoundary = hasEarlierLayers ? 190 : 60;
    let rightReserve = hasLaterLayers ? 190 : includesOutputLayer ? 190 : 60;
    let rightBoundary = width - rightReserve - nodeLength;
    let focusedTargetX = width / 2 - nodeLength / 2;
    let layersBeforeFocus = visibleLayers.filter(
      (layerIndex) => layerIndex < focusedLayerIndex,
    );
    let layersAfterFocus = visibleLayers.filter(
      (layerIndex) => layerIndex > focusedLayerIndex,
    );
    let targetXByLayer = new Map([[focusedLayerIndex, focusedTargetX]]);

    layersBeforeFocus.forEach((layerIndex, index) => {
      let step =
        (focusedTargetX - leftBoundary) / layersBeforeFocus.length;
      targetXByLayer.set(layerIndex, leftBoundary + step * index);
    });
    layersAfterFocus.forEach((layerIndex, index) => {
      let step =
        (rightBoundary - focusedTargetX) / layersAfterFocus.length;
      targetXByLayer.set(layerIndex, focusedTargetX + step * (index + 1));
    });

    let xScale = (layerIndex) => targetXByLayer.get(layerIndex);
    let layerShift = new Map();

    svg.selectAll("g.cnn-layer-group").each(function (d, layerIndex) {
      let layerGroup = d3.select(this);
      layerGroup.attr(
        "data-layer-focus-opacity",
        layerGroup.style("opacity") || "",
      );
      if (!visibleLayerSet.has(layerIndex)) {
        layerGroup.style("display", "none");
        return;
      }

      let originalX = nodeCoordinate[layerIndex][0].x;
      let targetX = xScale(layerIndex);
      layerShift.set(layerIndex, targetX - originalX);
      layerGroup
        .style("display", null)
        .style("opacity", 1)
        .attr("transform", `translate(${targetX - originalX}, 0)`)
        .selectAll("g.node-group")
        .on("click", (node, nodeIndex, groups) => {
          d3.event.stopPropagation();
          if (layerIndex !== focusedLayerIndex) {
            focusedLayerIndex = layerIndex;
            layerSearchValue = getLayerSearchLabel(layerIndex);
            renderLayerFocusView();
            return;
          }

          isLayerFocusActive = false;
          resetLayerFocusPresentation();
          revealOverviewThroughLayer(layerIndex);
          nodeClickHandler(node, nodeIndex, groups);
        });
    });

    svg.selectAll("g.layer-label, g.layer-detailed-label").each(function (
      d,
      selectionIndex,
    ) {
      let label = d3.select(this);
      let layerIndex = Number(String(this.id || "").match(/-(\d+)$/)?.[1]);
      label.attr("data-layer-focus-opacity", label.style("opacity") || "");
      if (!visibleLayerSet.has(layerIndex)) {
        label.style("display", "none");
        return;
      }

      let originalTransform = label.attr("transform") || "";
      let yMatch = originalTransform.match(/translate\([^,]+,\s*([^)]+)\)/);
      let labelY = yMatch?.[1] || 0;
      let shouldShowLabel =
        (detailedMode && label.classed("layer-detailed-label")) ||
        (!detailedMode && label.classed("layer-label"));
      label
        .style("display", shouldShowLabel ? null : "none")
        .style("opacity", 1)
        .attr(
          "transform",
          `translate(${xScale(layerIndex) + nodeLength / 2}, ${labelY})`,
        );
    });

    svg
      .selectAll(
        "g.stage-grouping, g.color-legend, g.input-annotation, g.edge-packet-group, g.node-pulse-group",
      )
      .style("display", "none");

    let linkGenerator = d3
      .linkHorizontal()
      .x((point) => point.x)
      .y((point) => point.y);
    svg
      .select("g.edge-group")
      .selectAll("path.edge")
      .each(function (edge) {
        let edgePath = d3.select(this);
        let sourceLayerIndex =
          edge.sourceLayerIndex ?? edge.targetLayerIndex - 1;
        let isVisible =
          visibleLayerSet.has(sourceLayerIndex) &&
          visibleLayerSet.has(edge.targetLayerIndex);

        if (!isVisible) {
          edgePath
            .style("visibility", "hidden")
            .style("opacity", 0)
            .style("pointer-events", "none");
          return;
        }

        let source = {
          ...edge.source,
          x: edge.source.x + (layerShift.get(sourceLayerIndex) || 0),
        };
        let target = {
          ...edge.target,
          x: edge.target.x + (layerShift.get(edge.targetLayerIndex) || 0),
        };
        edgePath
          .attr("d", linkGenerator({ source, target }))
          .style("visibility", "visible")
          .style("opacity", edgeOpacity)
          .style("pointer-events", "stroke");
      });

    let focusedOutline = svg
      .select(`#cnn-layer-group-${focusedLayerIndex}`)
      .select("rect.layer-hover-outline");
    focusedOutline
      .attr(
        "width",
        focusedLayerIndex === cnn.length - 1
          ? nodeLength + 120
          : nodeLength + 28,
      )
      .style("opacity", 1);

    if (visibleLayerSet.has(cnn.length - 1)) {
      let outputShift = layerShift.get(cnn.length - 1) || 0;
      svg.selectAll("g.output-winner-overlay").each(function () {
        let overlay = d3.select(this);
        let originalTransform = overlay.attr("transform") || "";
        overlay
          .attr("data-layer-focus-transform", originalTransform)
          .attr("transform", `translate(${outputShift}, 0)`);
      });
    }

    svg
      .selectAll(
        `#layer-label-${focusedLayerIndex} rect.layer-label-highlight,
        #layer-detailed-label-${focusedLayerIndex} rect.layer-label-highlight`,
      )
      .style("opacity", 1);

    let contextGroup = svg.append("g").attr("class", "layer-focus-context");
    let addRangeIndicator = (x, text, alignment = "start") => {
      let indicator = contextGroup
        .append("g")
        .attr("class", "layer-focus-range-indicator")
        .attr("transform", `translate(${x}, ${svgPaddings.top + height / 2})`);
      indicator
        .append("rect")
        .attr("x", alignment === "end" ? -130 : 0)
        .attr("y", -22)
        .attr("width", 130)
        .attr("height", 44)
        .attr("rx", 6)
        .attr("ry", 6);
      indicator
        .append("text")
        .attr("x", alignment === "end" ? -65 : 65)
        .attr("y", 1)
        .attr("text-anchor", "middle")
        .style("dominant-baseline", "middle")
        .text(text);
    };

    if (hasEarlierLayers) {
      addRangeIndicator(8, `Layers 1-${firstVisibleLayer}`);
    }
    if (hasLaterLayers) {
      addRangeIndicator(
        width - 8,
        `Layers ${lastVisibleLayer + 2}-${cnn.length}`,
        "end",
      );
    }

    layerFocusAnnouncement = `Showing layers ${firstVisibleLayer + 1} through ${
      lastVisibleLayer + 1
    }. Layer ${focusedLayerIndex + 1} is selected.`;
  };

  const focusLayer = (layerIndex) => {
    if (!cnn?.length || disableControl || isModelLoading) {
      return;
    }

    clearAutoRevealTimer();
    focusedLayerIndex = Math.max(0, Math.min(layerIndex, cnn.length - 1));
    layerSearchValue = getLayerSearchLabel(focusedLayerIndex);
    isLayerFocusActive = true;
    renderLayerFocusView();
  };

  const focusLayerFromSearch = () => {
    let layerIndex = parseLayerSearchValue(layerSearchValue);
    if (layerIndex >= 0) {
      focusLayer(layerIndex);
      return;
    }

    layerFocusAnnouncement = `No layer matches "${layerSearchValue}".`;
  };

  const handleLayerSearchKeydown = (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      focusLayerFromSearch();
    } else if (event.key === "Escape" && isLayerFocusActive) {
      exitLayerFocus();
    }
  };

  const moveLayerFocus = (direction) => {
    focusLayer(focusedLayerIndex + direction);
  };

  const exitLayerFocus = () => {
    if (!isLayerFocusActive) {
      return;
    }

    isLayerFocusActive = false;
    resetLayerFocusPresentation();
    layerFocusAnnouncement = "Full network view restored.";
  };

  // Task 5 at lines 1625-1631
  const modelChanged = async () => {
    if (selectedModelId === selectedModelSpec?.id) {
      //the same selected and already there
      return;
    }

    exitLayerFocus();
    await loadModelById(selectedModelId);
  };
</script>

{#if isAiTestUiMode}
  <AiTestPanel
    apiBase={PYTORCH_BACKEND_API_BASE}
    defaultModelId={DEFAULT_AI_TEST_UI_MODEL_ID}
    {imageOptions}
    models={modelOptions}
    {selectedImage}
    {selectedModelId}
    on:visualization-auto-reveal={handleAiVisualizationAutoReveal}
    on:visualization-image-change={handleAiVisualizationImageChange}
    on:visualization-model-change={handleAiVisualizationModelChange}
    on:test-type-change={handleAiTestTypeChange}
  />
{/if}

<div
  class:ai-test-visualization-mode={isAiTestUiMode}
  class:style-test-mode={isAiTestUiMode && Style_Test}
  class:ai-test-overview-hidden={isAiTestUiMode && !isAiVisualizationSelected}
  class="overview"
  bind:this={overviewComponent}
>
  <div class="control-container">
    <div class="control-summary">
      {#if selectedModelSpec}
        {selectedModelSpec.totalLayers} layers | {selectedModelSpec.summary}
        {#if selectedModelSpec.convKernelSummary}
          | {selectedModelSpec.convKernelSummary}
        {/if}
      {:else if isModelLoading}
        Loading model...
      {/if}
    </div>

    <div class="control-row">
      {#if !isAiTestUiMode}
        <div class="left-control">
          {#each imageOptions as image, i}
            <div
              class="image-container"
              on:click={disableControl || isModelLoading
                ? () => {}
                : imageOptionClicked}
              on:keydown={(event) =>
                handleKeyActivate(
                  event,
                  disableControl || isModelLoading
                    ? () => {}
                    : () => imageOptionClicked({ target: event.currentTarget }),
                )}
              class:inactive={selectedImage !== image.file}
              class:disabled={disableControl || isModelLoading}
              data-imageName={image.file}
              role="button"
              tabindex="0"
            >
              <img
                src={image.src || `assets/img/${image.file}`}
                alt="image option"
                title={image.class}
                data-imageName={image.file}
              />
            </div>
          {/each}

          <!--
            PYTORCH_BACKEND_INTEGRATION:
            Custom image URLs are part of the original TensorFlow.js browser-only
            flow. They remain below and become active again if ACTIVE_MODEL_RUNTIME
            is switched to RUNTIME_TENSORFLOW_JS.
          -->
          {#if ACTIVE_MODEL_RUNTIME !== RUNTIME_PYTORCH_BACKEND}
            <div
              class="image-container"
              class:inactive={selectedImage !== "custom"}
              class:disabled={disableControl || isModelLoading}
              data-imageName={"custom"}
              on:click={disableControl || isModelLoading
                ? () => {}
                : customImageClicked}
              on:keydown={(event) =>
                handleKeyActivate(
                  event,
                  disableControl || isModelLoading
                    ? () => {}
                    : customImageClicked,
                )}
              role="button"
              tabindex="0"
            >
              <img
                class="custom-image"
                src="assets/img/plus.svg"
                alt="plus button"
                title="Add new input image"
                data-imageName="custom"
              />

              <span
                class="fa-stack edit-icon"
                class:hidden={customImageURL === null}
              >
                <i class="fas fa-circle fa-stack-2x"></i>
                <i class="fas fa-pen fa-stack-1x fa-inverse"></i>
              </span>
            </div>
          {/if}

          <div
            class="hover-label badge"
            style="opacity:{hoverInfo.show ? 1 : 0}"
          >
            <span class="icon" style="margin-right: 5px;">
              <i class="fas fa-crosshairs"></i>
            </span>
            <span id="hover-label-text">
              {hoverInfo.text}
            </span>
          </div>
        </div>
      {/if}

      <div class="right-control" class:ai-test-right-control={isAiTestUiMode}>
        <div
          class="layer-focus-tools"
          role="group"
          aria-label="Layer focus controls"
        >
          <div class="select-wrapper layer-search-wrapper">
            <span class="icon is-left" aria-hidden="true">
              <i class="fas fa-search"></i>
            </span>
            <label class="sr-only" for="layer-search">Find a CNN layer</label>
            <input
              bind:value={layerSearchValue}
              class="custom-select layer-search-input"
              id="layer-search"
              list="layer-search-options"
              placeholder="Find layer"
              autocomplete="off"
              disabled={disableControl || isModelLoading || !cnn?.length}
              on:keydown={handleLayerSearchKeydown}
            />
            <datalist id="layer-search-options">
              {#each cnn || [] as layer, layerIndex}
                <option value={getLayerSearchLabel(layerIndex)}>
                  {layer[0]?.layerName}
                </option>
              {/each}
            </datalist>
          </div>

          <button
            class="custom-btn layer-focus-submit-btn"
            disabled={disableControl || isModelLoading || !cnn?.length}
            on:click={focusLayerFromSearch}
          >
            <span class="icon" aria-hidden="true">
              <i class="fas fa-crosshairs"></i>
            </span>
            <span>Focus</span>
          </button>

          {#if isLayerFocusActive}
            <button
              class="custom-btn layer-focus-icon-btn"
              title="Previous layer"
              aria-label="Previous layer"
              disabled={disableControl ||
                isModelLoading ||
                focusedLayerIndex <= 0}
              on:click={() => moveLayerFocus(-1)}
            >
              <span class="icon" aria-hidden="true">
                <i class="fas fa-chevron-left"></i>
              </span>
            </button>
            <span class="layer-focus-status" aria-live="polite">
              Layer {focusedLayerIndex + 1} of {cnn?.length || 0}
            </span>
            <button
              class="custom-btn layer-focus-icon-btn"
              title="Next layer"
              aria-label="Next layer"
              disabled={disableControl ||
                isModelLoading ||
                focusedLayerIndex >= (cnn?.length || 1) - 1}
              on:click={() => moveLayerFocus(1)}
            >
              <span class="icon" aria-hidden="true">
                <i class="fas fa-chevron-right"></i>
              </span>
            </button>
            <button
              class="custom-btn"
              disabled={disableControl || isModelLoading}
              on:click={exitLayerFocus}
            >
              <span class="icon" aria-hidden="true">
                <i class="fas fa-project-diagram"></i>
              </span>
              <span>Full network</span>
            </button>
          {/if}
          <span class="sr-only" aria-live="polite">
            {layerFocusAnnouncement}
          </span>
        </div>

        {#if !isLayerFocusActive}
          <button
            class="reveal-layer-btn"
            disabled={disableControl || isModelLoading}
            on:click={manualRevealButtonClicked}
          >
            <span class="reveal-icon">
              <i
                class={manualRevealInfo.hasMore
                  ? "fas fa-chevron-right"
                  : "fas fa-rotate-left"}
              ></i>
            </span>
            <span class="reveal-label">
              {manualRevealInfo.hasMore
                ? `Show ${
                    Style_Test
                      ? formatToolbarLayerName(manualRevealInfo.nextLabel)
                      : manualRevealInfo.nextLabel
                  }`
                : "Restart layers"}
            </span>
            <span class="reveal-progress">{manualRevealInfo.progressText}</span>
          </button>
        {/if}

        {#if !isAiTestUiMode && !isLayerFocusActive}
          <div class="select-wrapper" title="Switch trained architecture">
            <span class="icon is-left">
              <i class="fas fa-project-diagram"></i>
            </span>
            <select
              bind:value={selectedModelId}
              class="custom-select"
              disabled={disableControl ||
                isModelLoading ||
                !modelOptions.length}
              on:change={modelChanged}
            >
              {#each modelOptions as modelOption}
                <option value={modelOption.id}>{modelOption.label}</option>
              {/each}
            </select>
          </div>
        {/if}

        <button
          class="custom-btn"
          id="detailed-button"
          disabled={disableControl || isModelLoading}
          class:is-activated={detailedMode}
          on:click={detailedButtonClicked}
        >
          <span class="icon">
            <i class="fas fa-eye"></i>
          </span>
          <span id="hover-label-text"> Show detail </span>
        </button>

        <div class="select-wrapper" title="Change color scale range">
          <span class="icon is-left">
            <i class="fas fa-palette"></i>
          </span>
          <select
            bind:value={selectedScaleLevel}
            id="level-select"
            class="custom-select"
            disabled={disableControl || isModelLoading}
          >
            <option value="local">Unit</option>
            <option value="module">Module</option>
            <option value="global">Global</option>
          </select>
        </div>
      </div>
    </div>
  </div>

  <div class="cnn">
    <svg id="cnn-svg"></svg>
  </div>
</div>

{#if !isAiTestUiMode}
  <Article />
{/if}

<div id="detailview">
  {#if selectedNode.data && selectedNode.data.type === "conv" && selectedNodeIndex != -1}
    <ConvolutionView
      on:message={handleExitFromDetiledConvView}
      input={nodeData[selectedNodeIndex].input}
      kernel={nodeData[selectedNodeIndex].kernel}
      dataRange={nodeData.colorRange}
      colorScale={nodeData.inputIsInputLayer
        ? layerColorScales.input[0]
        : layerColorScales.conv}
      isInputInputLayer={nodeData.inputIsInputLayer}
      isExited={isExitedFromCollapse}
    />
  {:else if selectedNode.data && selectedNode.data.type === "relu"}
    <ActivationView
      on:message={handleExitFromDetiledActivationView}
      input={nodeData[0].input}
      output={nodeData[0].output}
      dataRange={nodeData.colorRange}
      activationType={selectedNode.data.layerName.includes("sigmoid") //Task 6.2
        ? "sigmoid"
        : "relu"}
      isExited={isExitedFromDetailedView}
    />
  {:else if selectedNode.data && selectedNode.data.type === "pool"}
    <PoolView
      on:message={handleExitFromDetiledPoolView}
      input={nodeData[0].input}
      output={nodeData[0].output}
      kernelLength={2}
      poolType={selectedNode.data.layerName.includes("avg_pool")
        ? "avg"
        : "max"}
      dataRange={nodeData.colorRange}
      isExited={isExitedFromDetailedView}
    />
  {:else if softmaxDetailViewInfo.show}
    <SoftmaxView
      logits={softmaxDetailViewInfo.logits}
      logitColors={softmaxDetailViewInfo.logitColors}
      selectedI={softmaxDetailViewInfo.selectedI}
      highlightI={softmaxDetailViewInfo.highlightI}
      outputName={softmaxDetailViewInfo.outputName}
      outputValue={softmaxDetailViewInfo.outputValue}
      startAnimation={softmaxDetailViewInfo.startAnimation}
      on:xClicked={handleExitFromDetiledSoftmaxView}
      on:mouseOver={softmaxDetailViewMouseOverHandler}
      on:mouseLeave={softmaxDetailViewMouseLeaveHandler}
    />
  {:else if denseDetailViewInfo.show}
    <DenseView
      layerName={denseDetailViewInfo.layerName}
      nodeIndex={denseDetailViewInfo.nodeIndex}
      terms={denseDetailViewInfo.terms}
      restTerm={denseDetailViewInfo.restTerm}
      bias={denseDetailViewInfo.bias}
      preActivation={denseDetailViewInfo.preActivation}
      output={denseDetailViewInfo.output}
      isActive={denseDetailViewInfo.isActive}
      on:xClicked={handleExitFromDetiledDenseView}
    />
  {/if}
</div>

<Modal on:xClicked={handleModalCanceled} on:urlTyped={handleCustomImage} />

<style>
  .overview {
    padding: 0;
    height: 100%;
    width: 100%;
    display: flex;
    position: relative;
    flex-direction: column;
    justify-content: flex-start;
    align-items: flex-start;
  }

  .control-container {
    padding: 14px 24px;
    display: flex;
    flex-direction: column;
    align-items: stretch;
    justify-content: flex-start;
    flex-wrap: wrap;
    gap: 12px;
    width: 100%;
    box-sizing: border-box;
    background-color: #ffffff;
  }

  .control-summary {
    color: #6b7280;
    font-size: 12px;
    font-weight: 500;
    white-space: normal;
  }

  .control-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: nowrap;
    gap: 16px;
    width: 100%;
  }

  .right-control {
    display: flex;
    align-items: center;
    flex-wrap: nowrap;
    gap: 12px;
    justify-content: flex-end;
    flex: 1 1 auto;
    min-width: 0;
  }

  .left-control {
    display: flex;
    align-items: center;
    flex-wrap: nowrap;
    gap: 8px;
    flex: 0 1 auto;
    min-width: 0;
  }

  .reveal-layer-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    min-width: 210px;
    padding: 8px 14px;
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    color: #4b5563;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .reveal-layer-btn:hover:not(:disabled) {
    background-color: #f9fafb;
    border-color: #d1d5db;
  }

  .reveal-layer-btn:disabled {
    cursor: not-allowed;
    opacity: 0.5;
  }

  .reveal-icon {
    color: #6b7280;
    font-size: 14px;
    font-weight: 700;
    line-height: 1;
  }

  .reveal-label {
    max-width: 124px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .reveal-progress {
    color: #9ca3af;
    font-size: 12px;
    font-weight: 500;
  }

  .cnn {
    width: 100%;
    flex: 1 1 auto;
    padding: 0;
    background: var(--light-gray);
    display: flex;
    align-items: flex-start;
    justify-content: center;
    overflow: hidden;
  }

  #cnn-svg {
    margin: 0 auto;
    display: block;
    width: 100%;
    max-width: 1680px;
    height: auto;
    max-height: calc(100vh - 180px);
    min-height: 420px;
    aspect-ratio: 1680 / 760;
  }

  .overview.ai-test-visualization-mode {
    --ai-vis-bg: #0b347d;
    --ai-vis-panel: rgba(12, 65, 156, 0.96);
    --ai-vis-cyan: #0efcff;
    --ai-vis-cyan-soft: rgba(14, 252, 255, 0.38);
    --ai-vis-blue: #008eff;
    --ai-vis-text: #e8fdff;
    --ai-vis-muted: #8edce7;
    background: var(--ai-vis-bg);
  }

  .overview.ai-test-overview-hidden {
    display: none;
  }

  .ai-test-visualization-mode .control-container {
    padding: 12px 24px 10px;
    background: linear-gradient(
      180deg,
      rgba(8, 33, 88, 0.98),
      rgba(11, 52, 125, 0.98)
    );
    border-top: 1px solid rgba(14, 252, 255, 0.55);
    border-bottom: 1px solid rgba(0, 142, 255, 0.85);
    box-shadow:
      inset 0 1px 0 rgba(14, 252, 255, 0.35),
      0 0 18px rgba(0, 142, 255, 0.28);
  }

  .ai-test-visualization-mode.style-test-mode .control-container {
    min-height: 86px;
    padding: 16px 26px 14px;
    background: #0b2f70;
    border-top: 0;
    border-bottom: 1px solid rgba(14, 252, 255, 0.42);
    box-shadow: none;
  }

  .ai-test-visualization-mode .control-summary {
    color: var(--ai-vis-text);
    text-shadow: 0 0 8px rgba(14, 252, 255, 0.4);
  }

  .ai-test-visualization-mode.style-test-mode .control-summary {
    color: rgba(232, 253, 255, 0.78);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0;
    line-height: 1.35;
    text-shadow: none;
  }

  .ai-test-visualization-mode.style-test-mode .control-row {
    align-items: center;
    gap: 18px;
  }

  .ai-test-visualization-mode.style-test-mode .right-control {
    gap: 10px;
  }

  .ai-test-visualization-mode .cnn {
    background: radial-gradient(
        circle at 50% 20%,
        rgba(0, 142, 255, 0.2),
        transparent 34%
      ),
      linear-gradient(180deg, #0b347d 0%, #0a2f73 100%);
    border-top: 1px solid rgba(14, 252, 255, 0.55);
    box-shadow:
      inset 0 0 24px rgba(0, 142, 255, 0.25),
      inset 0 1px 0 rgba(14, 252, 255, 0.22);
  }

  .ai-test-visualization-mode #cnn-svg {
    background: linear-gradient(rgba(14, 252, 255, 0.035) 1px, transparent 1px),
      linear-gradient(90deg, rgba(14, 252, 255, 0.035) 1px, transparent 1px),
      radial-gradient(
        circle at 50% 50%,
        rgba(10, 90, 185, 0.38),
        transparent 55%
      ),
      #0b347d;
    background-size:
      32px 32px,
      32px 32px,
      auto,
      auto;
  }

  .ai-test-visualization-mode.style-test-mode #cnn-svg {
    background: linear-gradient(rgba(164, 210, 232, 0.014) 1px, transparent 1px),
      linear-gradient(90deg, rgba(164, 210, 232, 0.014) 1px, transparent 1px),
      radial-gradient(
        circle at 48% 48%,
        rgba(21, 82, 151, 0.22),
        transparent 58%
      ),
      #0b347d;
    background-size:
      40px 40px,
      40px 40px,
      auto,
      auto;
  }

  .ai-test-visualization-mode .reveal-layer-btn,
  .ai-test-visualization-mode .custom-btn,
  .ai-test-visualization-mode #detailed-button {
    background: linear-gradient(180deg, #0c7ed0, #0557b6);
    border: 1px solid rgba(14, 252, 255, 0.86);
    border-radius: 3px;
    color: #ffffff;
    box-shadow:
      inset 0 0 8px rgba(14, 252, 255, 0.28),
      0 0 10px rgba(0, 142, 255, 0.32);
  }

  .ai-test-visualization-mode.style-test-mode .reveal-layer-btn,
  .ai-test-visualization-mode.style-test-mode .custom-btn,
  .ai-test-visualization-mode.style-test-mode #detailed-button,
  .ai-test-visualization-mode.style-test-mode .custom-select {
    min-height: 38px;
    background: rgba(7, 35, 84, 0.7);
    border: 1px solid rgba(164, 210, 232, 0.28);
    border-radius: 999px;
    color: rgba(232, 253, 255, 0.88);
    box-shadow: none;
    text-shadow: none;
  }

  .ai-test-visualization-mode.style-test-mode .reveal-layer-btn {
    min-width: 210px;
    padding: 8px 16px;
    background: rgba(8, 46, 106, 0.88);
    border-color: rgba(14, 252, 255, 0.48);
  }

  .ai-test-visualization-mode.style-test-mode .custom-btn,
  .ai-test-visualization-mode.style-test-mode #detailed-button {
    padding: 8px 16px;
  }

  .ai-test-visualization-mode .reveal-layer-btn:hover:not(:disabled),
  .ai-test-visualization-mode .custom-btn:hover:not(:disabled),
  .ai-test-visualization-mode #detailed-button:hover {
    background: linear-gradient(180deg, #08a6e2, #066bd0);
    border-color: #ffffff;
    color: #ffffff;
  }

  .ai-test-visualization-mode.style-test-mode
    .reveal-layer-btn:hover:not(:disabled),
  .ai-test-visualization-mode.style-test-mode .custom-btn:hover:not(:disabled),
  .ai-test-visualization-mode.style-test-mode #detailed-button:hover {
    background: rgba(10, 55, 124, 0.96);
    border-color: rgba(14, 252, 255, 0.56);
    color: #f8fdff;
  }

  .ai-test-visualization-mode.style-test-mode .reveal-label,
  .ai-test-visualization-mode.style-test-mode #hover-label-text {
    color: #f8fdff;
    font-size: 13px;
    font-weight: 700;
  }

  .ai-test-visualization-mode.style-test-mode .reveal-progress {
    min-width: 34px;
    padding: 2px 8px;
    border-radius: 999px;
    background: rgba(232, 253, 255, 0.1);
    color: rgba(232, 253, 255, 0.68);
    font-size: 11px;
    font-weight: 600;
    text-align: center;
  }

  .ai-test-visualization-mode.style-test-mode .reveal-icon,
  .ai-test-visualization-mode.style-test-mode .custom-btn .icon,
  .ai-test-visualization-mode.style-test-mode #detailed-button .icon,
  .ai-test-visualization-mode.style-test-mode .select-wrapper .icon.is-left {
    color: rgba(232, 253, 255, 0.72);
    opacity: 0.9;
  }

  .ai-test-visualization-mode .reveal-icon,
  .ai-test-visualization-mode .reveal-progress,
  .ai-test-visualization-mode .select-wrapper .icon.is-left {
    color: var(--ai-vis-text);
  }

  .ai-test-visualization-mode #detailed-button.is-activated,
  .ai-test-visualization-mode #detailed-button.is-activated:hover {
    color: #ffffff;
    border-color: #ffffff;
    background: linear-gradient(180deg, #10c7f4, #0875d1);
  }

  .ai-test-visualization-mode.style-test-mode #detailed-button.is-activated,
  .ai-test-visualization-mode.style-test-mode
    #detailed-button.is-activated:hover {
    color: #f8fdff;
    border-color: rgba(14, 252, 255, 0.58);
    background: rgba(8, 49, 113, 0.94);
  }

  .ai-test-visualization-mode .hover-label.badge {
    background: rgba(9, 43, 111, 0.92);
    border-color: rgba(14, 252, 255, 0.65);
    color: var(--ai-vis-text);
  }

  .ai-test-visualization-mode .custom-select {
    background: #0b347d;
    border-color: rgba(14, 252, 255, 0.65);
    color: #ffffff;
  }

  .ai-test-visualization-mode .custom-select:focus {
    border-color: #ffffff;
    box-shadow: 0 0 0 2px rgba(14, 252, 255, 0.22);
  }

  .ai-test-visualization-mode :global(.layer-label text),
  .ai-test-visualization-mode :global(.layer-detailed-label text),
  .ai-test-visualization-mode :global(.layer-intermediate-label text),
  .ai-test-visualization-mode :global(.output-text),
  .ai-test-visualization-mode :global(.output-winner-overlay text),
  .ai-test-visualization-mode :global(.classifier-hidden-layer text),
  .ai-test-visualization-mode :global(.classifier-dense-principle-panel text),
  .ai-test-visualization-mode :global(.classifier-dense-principle-panel tspan) {
    fill: #ffffff !important;
    opacity: 1 !important;
    text-shadow: 0 0 8px rgba(14, 252, 255, 0.55);
  }

  .ai-test-visualization-mode :global(.legend text),
  .ai-test-visualization-mode :global(.legend .tick text),
  .ai-test-visualization-mode :global(.input-annotation text),
  .ai-test-visualization-mode :global(.annotation-text) {
    fill: var(--ai-vis-text) !important;
    opacity: 0.96 !important;
    text-shadow: 0 0 8px rgba(14, 252, 255, 0.45);
  }

  .ai-test-visualization-mode :global(.layer-label-highlight) {
    fill: rgba(14, 252, 255, 0.16) !important;
    stroke: rgba(14, 252, 255, 0.75) !important;
  }

  .ai-test-visualization-mode :global(.layer-hover-outline) {
    fill: rgba(14, 252, 255, 0.08) !important;
    stroke: rgba(14, 252, 255, 0.62) !important;
  }

  .ai-test-visualization-mode :global(path.edge),
  .ai-test-visualization-mode :global(path.flow-edge),
  .ai-test-visualization-mode :global(path.flatten),
  .ai-test-visualization-mode :global(path.flatten-output),
  .ai-test-visualization-mode :global(path.dense-output),
  .ai-test-visualization-mode :global(path.classifier-head-link),
  .ai-test-visualization-mode :global(path.classifier-head-flatten-link),
  .ai-test-visualization-mode :global(path.classifier-head-dense-link),
  .ai-test-visualization-mode
    :global(path.classifier-dense-detail-selected-input-edge),
  .ai-test-visualization-mode :global(path.classifier-dense-detail-input-edge),
  .ai-test-visualization-mode
    :global(path.classifier-dense-detail-internal-edge),
  .ai-test-visualization-mode :global(path.classifier-dense-detail-output-edge),
  .ai-test-visualization-mode
    :global(path.classifier-dense-detail-context-edge),
  .ai-test-visualization-mode :global(path.softmax-edge),
  .ai-test-visualization-mode :global(.symbol-output),
  .ai-test-visualization-mode :global(.symbol-output-line),
  .ai-test-visualization-mode :global(.symbol-softmax) {
    stroke: rgba(96, 228, 255, 0.58) !important;
    filter: drop-shadow(0 0 3px rgba(14, 252, 255, 0.42));
  }

  .ai-test-visualization-mode
    :global(path.classifier-dense-detail-output-edge-active),
  .ai-test-visualization-mode
    :global(path.classifier-dense-detail-selected-input-edge) {
    stroke: #0efcff !important;
    opacity: 0.86 !important;
    filter: drop-shadow(0 0 5px rgba(14, 252, 255, 0.55));
  }

  .ai-test-visualization-mode :global(rect.bounding),
  .ai-test-visualization-mode :global(.bounding-flatten),
  .ai-test-visualization-mode :global(.kernel rect),
  .ai-test-visualization-mode :global(.underneath-gateway) {
    stroke: var(--ai-vis-cyan) !important;
  }

  .ai-test-visualization-mode :global(.output-rect) {
    fill: var(--ai-vis-cyan) !important;
    filter: drop-shadow(0 0 5px rgba(14, 252, 255, 0.56));
  }

  .ai-test-visualization-mode :global(.output-winner-overlay rect),
  .ai-test-visualization-mode :global(.output-winner-overlay path) {
    fill: rgba(14, 252, 255, 0.18) !important;
    stroke: var(--ai-vis-cyan) !important;
  }

  .ai-test-visualization-mode :global(.output-winner-overlay text) {
    fill: #ffffff !important;
    stroke: rgba(6, 28, 70, 0.62);
    stroke-width: 0.45px;
    paint-order: stroke fill;
  }

  .ai-test-visualization-mode.style-test-mode :global(.layer-label text),
  .ai-test-visualization-mode.style-test-mode
    :global(.layer-detailed-label text),
  .ai-test-visualization-mode.style-test-mode
    :global(.classifier-hidden-layer-label),
  .ai-test-visualization-mode.style-test-mode
    :global(.classifier-hidden-layer-count) {
    fill: rgba(232, 253, 255, 0.56) !important;
    opacity: 1 !important;
    text-shadow: none;
  }

  .ai-test-visualization-mode.style-test-mode
    :global(.layer-label.is-style-active text),
  .ai-test-visualization-mode.style-test-mode
    :global(.layer-detailed-label.is-style-active text),
  .ai-test-visualization-mode.style-test-mode
    :global(
      .classifier-hidden-layer.is-style-active .classifier-hidden-layer-label
    ),
  .ai-test-visualization-mode.style-test-mode
    :global(
      .classifier-hidden-layer.is-style-active .classifier-hidden-layer-count
    ) {
    fill: #f8fdff !important;
    opacity: 1 !important;
    text-shadow: 0 0 5px rgba(14, 252, 255, 0.28);
  }

  .ai-test-visualization-mode.style-test-mode
    :global(.layer-label.is-style-muted text),
  .ai-test-visualization-mode.style-test-mode
    :global(.layer-detailed-label.is-style-muted text),
  .ai-test-visualization-mode.style-test-mode
    :global(
      .classifier-hidden-layer.is-style-muted .classifier-hidden-layer-label
    ),
  .ai-test-visualization-mode.style-test-mode
    :global(
      .classifier-hidden-layer.is-style-muted .classifier-hidden-layer-count
    ) {
    fill: rgba(232, 253, 255, 0.34) !important;
    opacity: 1 !important;
    text-shadow: none;
  }

  .ai-test-visualization-mode.style-test-mode
    :global(.output-winner-overlay rect.output-winner-card) {
    fill: rgba(8, 31, 70, 0.94) !important;
    stroke: rgba(14, 252, 255, 0.58) !important;
    stroke-width: 1.1px !important;
  }

  .ai-test-visualization-mode.style-test-mode
    :global(.output-winner-overlay rect.output-winner-accent) {
    fill: #0efcff !important;
    stroke: none !important;
  }

  .ai-test-visualization-mode.style-test-mode
    :global(.output-winner-overlay text.output-winner-text) {
    fill: #f8fdff !important;
    stroke: none;
    font-weight: 700 !important;
    text-shadow: none;
  }

  .ai-test-visualization-mode.style-test-mode :global(.output-text) {
    fill: rgba(232, 253, 255, 0.72) !important;
    opacity: 1 !important;
    text-shadow: none;
  }

  .ai-test-visualization-mode.style-test-mode :global(.output-rect) {
    fill: rgba(164, 210, 232, 0.48) !important;
    opacity: 0.7 !important;
    filter: none;
  }

  .ai-test-visualization-mode :global(.legend line),
  .ai-test-visualization-mode :global(.legend path),
  .ai-test-visualization-mode :global(.color-legend line),
  .ai-test-visualization-mode :global(.color-legend path) {
    stroke: rgba(232, 253, 255, 0.65) !important;
  }

  .ai-test-visualization-mode :global(.legend > rect) {
    stroke: rgba(14, 252, 255, 0.45) !important;
  }

  .ai-test-visualization-mode :global(.softmax-symbol rect),
  .ai-test-visualization-mode :global(.symbol-softmax rect),
  .ai-test-visualization-mode :global(.classifier-dense-detail-halo),
  .ai-test-visualization-mode :global(.classifier-dense-detail-computation-bg),
  .ai-test-visualization-mode :global(.classifier-dense-detail-glow) {
    fill: rgba(8, 42, 108, 0.95) !important;
    stroke: rgba(14, 252, 255, 0.75) !important;
  }

  .ai-test-visualization-mode :global(.softmax-symbol text),
  .ai-test-visualization-mode :global(.symbol-softmax text),
  .ai-test-visualization-mode :global(.classifier-dense-detail text),
  .ai-test-visualization-mode :global(.classifier-dense-detail tspan) {
    fill: var(--ai-vis-text) !important;
  }

  :global(body.ai-test-ui-mode #detailview) {
    color: #ffffff;
    z-index: 30;
  }

  :global(body.ai-test-ui-mode #detailview .box) {
    background: linear-gradient(
      180deg,
      rgba(13, 67, 160, 0.98),
      rgba(8, 45, 113, 0.98)
    );
    border: 1px solid rgba(14, 252, 255, 0.7);
    color: #ffffff;
    box-shadow:
      inset 0 0 18px rgba(0, 142, 255, 0.22),
      0 0 20px rgba(0, 142, 255, 0.38);
  }

  :global(body.ai-test-ui-mode #detailview .title-text) {
    color: #0efcff;
    text-shadow: 0 0 8px rgba(14, 252, 255, 0.5);
  }

  :global(body.ai-test-ui-mode #detailview .annotation),
  :global(body.ai-test-ui-mode #detailview .annotation-text) {
    color: #e8fdff;
  }

  :global(body.ai-test-ui-mode #detailview .control-button) {
    color: #ffffff;
    opacity: 0.78;
  }

  #detailed-button {
    margin-right: 0;
  }

  #detailed-button.is-activated,
  #detailed-button.is-activated:hover {
    color: #1d4ed8;
    border-color: #93c5fd;
    background-color: #eff6ff;
  }

  .hover-label.badge {
    transition: opacity 300ms ease-in-out;
    text-overflow: ellipsis;
    pointer-events: none;
    margin-left: 12px;
    padding: 6px 12px;
    background-color: #f3f4f6;
    color: #4b5563;
    font-size: 13px;
    font-weight: 500;
    border-radius: 6px;
    border: 1px solid #e5e7eb;
    display: flex;
    align-items: center;
    gap: 6px;
    flex: 0 1 200px;
    min-width: 120px;
  }

  @media (max-width: 1500px) {
    .control-row {
      flex-wrap: wrap;
      justify-content: flex-start;
    }

    .left-control,
    .right-control {
      flex-wrap: wrap;
      justify-content: flex-start;
      flex: 1 1 100%;
    }
  }

  .image-container {
    width: 48px;
    height: 48px;
    border-radius: 6px;
    display: flex;
    position: relative;
    border: 2px solid #1f2937;
    cursor: pointer;
    overflow: hidden;
    align-items: center;
    justify-content: center;
    background-color: #ffffff;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  }

  .image-container img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
    transition: opacity 0.2s ease;
  }

  .image-container.inactive {
    border: 2px solid #e5e7eb;
    box-shadow: none;
  }

  .image-container.inactive > img {
    opacity: 0.45;
  }

  .image-container.inactive:hover > img {
    opacity: 0.7;
  }

  .image-container.inactive.disabled {
    border: 2px solid #e5e7eb;
    cursor: not-allowed;
  }

  .image-container.inactive.disabled:hover {
    border: 2.5px solid rgb(220, 220, 220);
    cursor: not-allowed;
  }

  .image-container.inactive.disabled > img {
    opacity: 0.45;
    cursor: not-allowed;
  }

  .image-container.inactive.disabled:hover > img {
    opacity: 0.45;
    cursor: not-allowed;
  }

  .image-container.inactive > .edit-icon {
    color: #bababa;
  }

  .image-container.inactive:hover > .edit-icon {
    color: #777777;
  }

  .image-container.inactive:hover {
    border-color: #d1d5db;
  }

  .image-container:hover:not(.disabled) {
    transform: translateY(-2px);
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  }

  .edit-icon {
    position: absolute;
    bottom: -4px;
    right: -4px;
    font-size: 10px;
    color: #3b82f6;
    transition: color 0.2s ease;
  }

  .image-container .custom-image {
    width: 20px;
    height: 20px;
    opacity: 0.4;
  }

  .image-container.inactive:hover .custom-image {
    opacity: 0.6;
  }

  .custom-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 14px;
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    color: #4b5563;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .custom-btn:hover:not(:disabled) {
    background-color: #f9fafb;
    border-color: #d1d5db;
  }

  .custom-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .layer-focus-tools {
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 0;
  }

  .layer-search-wrapper {
    flex: 0 1 180px;
    min-width: 140px;
  }

  .layer-search-input {
    width: 100%;
    min-width: 0;
    padding-right: 10px;
    cursor: text;
  }

  .layer-focus-icon-btn {
    width: 34px;
    height: 34px;
    padding: 0;
    justify-content: center;
    flex: 0 0 34px;
  }

  .layer-focus-submit-btn {
    height: 35px;
    padding: 4px 14px;
  }

  .layer-focus-status {
    color: #6b7280;
    font-size: 12px;
    font-weight: 500;
    white-space: nowrap;
  }

  .select-wrapper {
    position: relative;
    display: flex;
    align-items: center;
  }

  .custom-select {
    appearance: none;
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    color: #4b5563;
    font-size: 13px;
    font-weight: 500;
    padding: 8px 14px 8px 32px;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .custom-select:focus {
    outline: none;
    border-color: #3b82f6;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1);
  }

  .select-wrapper .icon.is-left {
    position: absolute;
    left: 12px;
    color: #9ca3af;
    font-size: 12px;
    pointer-events: none;
  }

  .icon i {
    font-size: 14px;
  }

  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }

  :global(canvas) {
    image-rendering: auto;
  }

  :global(.layer-focus-range-indicator rect) {
    fill: #ffffff;
    stroke: #d1d5db;
    stroke-width: 1;
  }

  :global(.layer-focus-range-indicator text) {
    fill: #6b7280;
    font-family:
      "Inter",
      -apple-system,
      sans-serif;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0;
  }

  :global(.layer-label),
  :global(.layer-detailed-label),
  :global(.layer-intermediate-label) {
    font-family:
      "Inter",
      -apple-system,
      sans-serif;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    text-anchor: middle;
    transition: opacity 300ms ease;
  }

  :global(.hidden) {
    opacity: 0;
    pointer-events: none;
  }

  :global(.very-strong) {
    stroke-width: 3px;
  }

  :global(.colorLegend) {
    font-size: 10px;
  }

  :global(.legend) {
    transition: opacity 400ms ease-in-out;
  }

  :global(.legend > rect) {
    opacity: 1;
  }

  :global(.legend text),
  :global(.legend line),
  :global(.legend path) {
    opacity: 0.7;
  }

  :global(.legend#output-legend > rect) {
    opacity: 1;
  }

  :global(.bounding),
  :global(.edge),
  :global(.edge-group),
  :global(foreignObject),
  :global(.bounding-flatten),
  :global(.underneath-gateway),
  :global(.input-annotation) {
    transition: opacity 300ms ease-in-out;
  }

  :global(rect.bounding) {
    transition:
      stroke-width 800ms ease-in-out,
      opacity 300ms ease-in-out;
  }

  :global(.annotation-text) {
    pointer-events: none;
    font-size: 10px;
    font-style: italic;
    fill: gray;
  }

  /* Change the cursor style on the detailed view input and output matrices */
  :global(rect.square) {
    cursor: crosshair;
  }

  :global(.animation-control-button) {
    font-family: FontAwesome;
    opacity: 0.8;
    cursor: pointer;
  }
</style>
