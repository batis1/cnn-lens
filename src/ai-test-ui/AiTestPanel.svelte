<script>
  import { createEventDispatcher, onDestroy, onMount } from "svelte";
  import {
    generateDataWithBackend,
    testWithBackend,
  } from "../utils/cnn-backend.js";
  import AiTestResultChart from "./AiTestResultChart.svelte";

  export let apiBase;
  export let defaultModelId = "cnn-net-28-ori";
  export let models = [];
  export let imageOptions = [];
  export let selectedImage = "";
  export let selectedModelId = "";

  const dispatch = createEventDispatcher();

  const aiImage = (fileName) =>
    `${import.meta.env.BASE_URL}assets/ai-test-ui/images/${fileName}`;
  const aiImageStyle = [
    `--ai-dashboard-bg: url("${aiImage("bg2.jpg")}")`,
    `--ai-title-main-bg: url("${aiImage("title_bg.png")}")`,
    `--ai-title-strip-bg: url("${aiImage("titbg.png")}")`,
    `--ai-title-left-bg: url("${aiImage("titleft.png")}")`,
    `--ai-title-right-bg: url("${aiImage("titright.png")}")`,
    `--ai-box-corner-bg: url("${aiImage("border_bg.jpg")}")`,
  ].join("; ");

  const defaultModelFileName = "CNN_NET_28_ori.pt";
  const testItems = [
    {
      value: "accuracy",
      label: "准确率",
      icon: "fas fa-bullseye",
      dataPath: "accuracy",
    },
    {
      value: "robustness",
      label: "鲁棒性",
      icon: "fas fa-shield-alt",
      dataPath: "robustness",
    },
    {
      value: "adversary",
      label: "对抗性",
      icon: "fas fa-bolt",
      dataPath: "adversary",
    },
    {
      value: "coverage",
      label: "覆盖率",
      icon: "fas fa-chart-bar",
      dataPath: "coverage",
    },
    {
      value: "visualization",
      label: "模型可视化模块",
      icon: "fas fa-eye",
      dataPath: "",
      isVisualization: true,
    },
  ];

  const generationMethods = [
    {
      value: "noise",
      label: "噪声生成方法",
      options: [
        { value: "salt_pepper_noise", label: "椒盐噪声" },
        { value: "gauss_noise", label: "高斯噪声" },
        { value: "poisson_noise", label: "泊松噪声" },
      ],
    },
    {
      value: "transform",
      label: "变换生成方法",
      options: [
        { value: "rotation", label: "旋转变换" },
        { value: "scale", label: "放缩变换" },
        { value: "translation", label: "平移变换" },
      ],
    },
    {
      value: "adversary",
      label: "对抗生成方法",
      options: [{ value: "fgsm", label: "快速梯度符号法(FGSM)" }],
    },
  ];
  const robustnessSeries = [
    { key: "gauss_noise", label: "高斯噪声" },
    { key: "poisson_noise", label: "泊松噪声" },
    { key: "salt_pepper_noise", label: "椒盐噪声" },
    { key: "rotation", label: "旋转变换" },
    { key: "scale", label: "放缩变换" },
    { key: "translation", label: "平移变换" },
  ];
  const adversarySeries = [
    { key: "epsilon_0.1", label: "epsilon_0.1" },
    { key: "epsilon_0.2", label: "epsilon_0.2" },
    { key: "epsilon_0.3", label: "epsilon_0.3" },
  ];
  const coverageSeries = [
    { key: "hidden1", label: "hidden1" },
    { key: "hidden2", label: "hidden2" },
    { key: "full", label: "total" },
  ];

  let selectedTestType = "accuracy";
  let modelPath = "";
  let selectedModelFileName = "";
  let testDataPath = "accuracy";
  let selectedDataFolderName = "";
  let refDataPath = "";
  let outputPath = "";
  let selectedGenerationMethod = "noise";
  let selectedGenerationOption = generationMethods[0].options[0].value;
  let result = null;
  let resultText = "";
  let errorMessage = "";
  let generationMessage = "";
  let isTesting = false;
  let isGenerating = false;
  let generationError = false;
  let hoveredAccuracySlice = "";

  $: selectedTestItem =
    testItems.find((item) => item.value === selectedTestType) || testItems[0];
  $: isVisualizationTestType = Boolean(selectedTestItem?.isVisualization);
  $: selectedGeneration =
    generationMethods.find((item) => item.value === selectedGenerationMethod) ||
    generationMethods[0];
  $: if (!selectedGeneration.options.some((option) => option.value === selectedGenerationOption)) {
    selectedGenerationOption = selectedGeneration.options[0].value;
  }
  $: correctRate = Number.isFinite(result?.accuracy) ? result.accuracy : 0;
  $: errorRate = Math.max(0, 1 - correctRate);
  $: activeChartType = result?.type || selectedTestType;
  $: robustnessChartData = buildKeyedChartData(
    result?.robustness,
    robustnessSeries,
  );
  $: adversaryChartData = buildKeyedChartData(
    result?.adversary,
    adversarySeries,
  );
  $: adversaryLinePoints = buildLinePoints(adversaryChartData);
  $: coverageChartData = buildCoverageChartData(result?.coverage);
  $: visualizationModelOptions = models.filter(
    (model) => model.supportedForExplanation !== false,
  );
  $: currentVisualizationModel =
    models.find((model) => model.id === selectedModelId) ||
    resolveRegisteredModel(modelPath) ||
    models.find((model) => model.id === defaultModelId);
  $: selectedVisualizationModel =
    visualizationModelOptions.find((model) => model.id === selectedModelId) ||
    visualizationModelOptions[0] ||
    currentVisualizationModel;
  $: selectedVisualizationModelId = selectedVisualizationModel?.id || "";
  $: selectedVisualizationImage =
    imageOptions.find((image) => image.file === selectedImage) ||
    imageOptions[0];
  $: selectedVisualizationImagePath =
    selectedVisualizationImage?.backendPath ||
    selectedVisualizationImage?.file ||
    "";
  $: resolvedModelLabel = selectedVisualizationModel?.label ||
    currentVisualizationModel?.label ||
    "CNN NET 28 original";

  onMount(() => {
    document.body.classList.add("ai-test-ui-mode");
    dispatch("test-type-change", { testType: selectedTestType });
  });

  onDestroy(() => {
    document.body.classList.remove("ai-test-ui-mode");
  });

  const handleModelFileChanged = (event) => {
    let file = event.currentTarget.files?.[0];
    selectedModelFileName = file?.name || "";
    modelPath = selectedModelFileName ? `./${selectedModelFileName}` : "";
    let registeredModel = resolveRegisteredModel(modelPath);
    if (registeredModel) {
      dispatch("visualization-model-change", { modelId: registeredModel.id });
    }
    resetResult();
  };

  const handleRefDataFolderChanged = (event) => {
    refDataPath = inferFolderPath(event.currentTarget.files?.[0]) || "";
    generationMessage = "";
    generationError = false;
  };

  const handleOutputFolderChanged = (event) => {
    let folderPath = inferFolderPath(event.currentTarget.files?.[0]) || "";
    outputPath = folderPath ? `./generate_data/${folderPath}` : "";
    generationMessage = "";
    generationError = false;
  };

  const handleDataFolderChanged = (event) => {
    let folderPath = inferFolderPath(event.currentTarget.files?.[0]) || "";
    let folderName = normalizeDataFolderName(folderPath);

    selectedDataFolderName = folderName || "";
    testDataPath = selectedDataFolderName || selectedTestItem.dataPath;
    resetResult();
  };

  const handleTestTypeChanged = (event) => {
    let nextTestType = event.currentTarget.value;
    let nextTestItem =
      testItems.find((item) => item.value === nextTestType) || testItems[0];

    selectedTestType = nextTestType;
    dispatch("test-type-change", { testType: nextTestType });
    selectedDataFolderName = "";
    resetResult();

    if (nextTestItem?.isVisualization) {
      if (
        selectedVisualizationModelId &&
        selectedVisualizationModelId !== selectedModelId
      ) {
        dispatch("visualization-model-change", {
          modelId: selectedVisualizationModelId,
        });
      }

      testDataPath = "";
      resultText = selectedVisualizationImagePath
        ? `当前可视化图片：${selectedVisualizationImagePath}`
        : "请选择待可视化图片。";
      return;
    }

    testDataPath = nextTestItem.dataPath;
  };

  const handleGenerateData = async () => {
    generationMessage = "";
    generationError = false;

    if (!refDataPath) {
      generationError = true;
      generationMessage = "请选择参照输入数据所在文件夹。";
      return;
    }
    if (!outputPath) {
      generationError = true;
      generationMessage = "请选择保存生成数据的文件夹。";
      return;
    }
    if (selectedGenerationMethod === "adversary") {
      generationError = true;
      generationMessage = "FGSM 需要模型和标签，将在对抗性生成步骤中接入。";
      return;
    }

    isGenerating = true;
    generationMessage = "正在生成测试数据...";
    try {
      let response = await generateDataWithBackend(
        {
          dataPath: refDataPath,
          outputPath,
          generationMethod: selectedGenerationMethod,
          option: selectedGenerationOption,
        },
        apiBase,
      );
      generationMessage = `数据生成完成：${response.generated} 个文件 -> ${response.outputPath}`;
    } catch (error) {
      generationError = true;
      generationMessage = `数据生成失败：${error.message || error}`;
    } finally {
      isGenerating = false;
    }
  };

  const runTest = async () => {
    errorMessage = "";
    result = null;
    resultText = "";

    if (isVisualizationTestType) {
      requestVisualization();
      return;
    }

    if (!modelPath) {
      modelPath = `./${defaultModelFileName}`;
    }

    if (!testDataPath) {
      errorMessage = "请选择测试数据所在文件夹。";
      resultText = errorMessage;
      return;
    }

    let folderName = normalizeDataFolderName(testDataPath);
    if (folderName && folderName !== selectedTestType) {
      errorMessage = "选择的测试条目与测试数据文件夹名字不匹配，请重新选择上传文件或选择待测试条目。";
      resultText = errorMessage;
      return;
    }

    isTesting = true;

    try {
      let payload = buildTestPayload();
      result = await testWithBackend(payload, apiBase);
      resultText = result.output || formatResultOutput(result);
    } catch (error) {
      errorMessage = error.message || "模型测试失败。";
      resultText = `错误：${errorMessage}`;
    } finally {
      isTesting = false;
    }
  };

  const buildTestPayload = () => {
    let payload = {
      testDataPath,
      testType: selectedTestType,
    };
    let registeredModel = resolveRegisteredModel(modelPath);

    if (registeredModel) {
      payload.modelId = registeredModel.id;
    } else if (!modelPath || extractFileName(modelPath) === defaultModelFileName) {
      payload.modelId = defaultModelId;
    } else {
      payload.modelPath = modelPath;
    }

    return payload;
  };

  const resolveRegisteredModel = (pathText) => {
    let fileName = extractFileName(pathText);
    return models.find((model) => extractFileName(model.sourcePath) === fileName);
  };

  const extractFileName = (pathText) =>
    (pathText || "").split(/[\\/]/).filter(Boolean).pop() || "";

  const inferFolderPath = (file) => {
    let parts = (file?.webkitRelativePath || "")
      .split(/[\\/]/)
      .filter(Boolean);
    return parts[0] || "";
  };

  const normalizeDataFolderName = (pathText) => {
    let normalized = (pathText || "")
      .replace(/^\.\//, "")
      .replace(/^generate_data[\\/]/, "");
    let parts = normalized.split(/[\\/]/).filter(Boolean);
    return parts[0] || "";
  };

  const resetResult = () => {
    result = null;
    resultText = "";
    errorMessage = "";
  };

  const percentNumber = (value) =>
    Number.isFinite(Number(value))
      ? Math.max(0, Math.min(100, Number(value) * 100))
      : 0;

  const percentText = (value, digits = 1) =>
    `${percentNumber(value).toFixed(digits)}%`;

  const formatPercent = (value) => {
    if (!Number.isFinite(value)) {
      return "0%";
    }

    return `${Math.round(value * 100)}%`;
  };

  const buildKeyedChartData = (values, series) =>
    series.map((item) => ({
      ...item,
      value: Number.isFinite(Number(values?.[item.key]))
        ? Number(values[item.key])
        : 0,
    }));

  const buildLinePoints = (items) => {
    let width = 360;
    let height = 160;
    let left = 28;
    let right = 18;
    let top = 18;
    let bottom = 34;
    let usableWidth = width - left - right;
    let usableHeight = height - top - bottom;

    return items
      .map((item, index) => {
        let x = left + (usableWidth * index) / Math.max(items.length - 1, 1);
        let y = top + usableHeight - (percentNumber(item.value) / 100) * usableHeight;
        return `${x},${y}`;
      })
      .join(" ");
  };

  const buildCoverageChartData = (coverage) => {
    let source = Array.isArray(coverage)
      ? coverage
      : coverage?.datasets || coverage?.data || [];
    let defaults = [
      { display: "原始数据集", hidden1: 0, hidden2: 0, full: 0 },
      { display: "新数据集", hidden1: 0, hidden2: 0, full: 0 },
    ];

    return defaults.map((fallback, index) => {
      let item = source[index] || {};
      return {
        display: item.display || item.name || fallback.display,
        hidden1: Number.isFinite(Number(item.hidden1))
          ? Number(item.hidden1)
          : fallback.hidden1,
        hidden2: Number.isFinite(Number(item.hidden2))
          ? Number(item.hidden2)
          : fallback.hidden2,
        full: Number.isFinite(Number(item.full))
          ? Number(item.full)
          : fallback.full,
      };
    });
  };

  const formatResultOutput = (payload) => {
    if (!payload) {
      return "";
    }

    return `${payload.type || selectedTestType}: ${formatPercent(payload.accuracy)}`;
  };

  const handleVisualizationModelChanged = (event) => {
    let modelId = event.currentTarget.value;
    if (!modelId) {
      return;
    }

    dispatch("visualization-model-change", { modelId });
    let nextModel = models.find((model) => model.id === modelId);
    resultText = `Visualization model: ${nextModel?.label || modelId}`;
  };

  const handleVisualizationImageChanged = (event) => {
    let imageFile = event.currentTarget.value;
    selectedImage = imageFile;
    requestVisualization(imageFile);
  };

  const requestVisualization = (imageFile = selectedVisualizationImagePath) => {
    if (!imageFile) {
      resultText = "请选择待可视化图片。";
      return;
    }

    dispatch("visualization-image-change", {
      imageFile,
    });
    resultText = `当前可视化图片：${imageFile}`;
  };

  const requestAutoVisualization = () => {
    if (!selectedVisualizationImagePath) {
      resultText = "请选择待可视化图片。";
      return;
    }

    dispatch("visualization-auto-reveal", {
      imageFile: selectedVisualizationImagePath,
    });
    resultText = `正在自动展示：${selectedVisualizationImagePath}`;
  };
</script>

<!--
  AI_TEST_UI_INTEGRATION:
  This recreates AI_Test_UI/UI.html's four dashboard modules. It intentionally
  does not render AI_Test_UI's old echart/FCNN visualization; the current
  CNN Explainer visualization is mounted below this panel in Overview.svelte.
-->
<section
  class="ai-test-dashboard"
  aria-label="AI Test UI dashboard"
  style={aiImageStyle}
>
  <div class="dashboard-title">
    <img
      class="dashboard-title-bg"
      src={aiImage("title_bg.png")}
      alt=""
      aria-hidden="true"
    />
    <span class="title-text">
      人工智能模型测试与评估平台
    </span>
  </div>

  <div class="dashboard-grid">
    <div class="ai-box model-box">
      <div class="box-title">
        <span><i class="fas fa-upload"></i> 模型导入和集成模块</span>
      </div>

      <div class="box-body">
        <div class="inline-field">
          <label for="ai-test-model-input">
            <i class="fas fa-file-upload"></i>
            请导入待测试的人工智能模型：
          </label>
          <input
            id="ai-test-model-input"
            class="file-picker-input"
            type="file"
            accept=".pt,.pth"
            on:change={handleModelFileChanged}
          />
          <label class="file-picker-button" for="ai-test-model-input">
            Choose File
          </label>
          {#if selectedModelFileName}
            <span class="file-picker-name">{selectedModelFileName}</span>
          {/if}
        </div>

        <label for="ai-test-model-path">
          <i class="fas fa-link"></i>
          所导入的人工智能模型相对路径：
        </label>
        <input
          id="ai-test-model-path"
          class="blue-textbox"
          type="text"
          bind:value={modelPath}
          readonly
        />

        <div class="model-note">
          <i class="fas fa-cube"></i>
          默认测试模型：{resolvedModelLabel}
        </div>
      </div>
    </div>

    <div class="ai-box generate-box">
      <div class="box-title">
        <span><i class="fas fa-database"></i> 测试数据生成模块</span>
      </div>

      <div class="box-body">
        <div class="inline-field">
          <label for="ai-test-ref-data-input">
            <i class="fas fa-folder-open"></i>
            请选择参照输入数据所在文件夹：
          </label>
          <input
            id="ai-test-ref-data-input"
            class="file-picker-input"
            type="file"
            webkitdirectory=""
            directory=""
            multiple
            on:change={handleRefDataFolderChanged}
          />
          <label class="file-picker-button" for="ai-test-ref-data-input">
            Choose Folder
          </label>
          {#if refDataPath}
            <span class="file-picker-name">{refDataPath}</span>
          {/if}
        </div>

        <label for="ai-test-ref-data-path">
          <i class="fas fa-link"></i>
          所导入的参照输入数据相对路径：
        </label>
        <input
          id="ai-test-ref-data-path"
          class="blue-textbox"
          type="text"
          bind:value={refDataPath}
          readonly
        />

        <div class="inline-field">
          <label for="ai-test-output-folder-input">
            <i class="fas fa-save"></i>
            请选择保存生成的测试数据的文件夹：
          </label>
          <input
            id="ai-test-output-folder-input"
            class="file-picker-input"
            type="file"
            webkitdirectory=""
            directory=""
            on:change={handleOutputFolderChanged}
          />
          <label class="file-picker-button" for="ai-test-output-folder-input">
            Choose Folder
          </label>
          {#if outputPath}
            <span class="file-picker-name">{outputPath}</span>
          {/if}
        </div>

        <label for="ai-test-output-folder-path">
          <i class="fas fa-link"></i>
          所选择的保存测试数据文件夹相对路径：
        </label>
        <input
          id="ai-test-output-folder-path"
          class="blue-textbox"
          type="text"
          bind:value={outputPath}
          readonly
        />

        <div class="generate-controls">
          <label for="ai-test-generation-method">
            <i class="fas fa-sliders-h"></i>
            请选择数据生成方法：
          </label>
          <select
            id="ai-test-generation-method"
            bind:value={selectedGenerationMethod}
          >
            {#each generationMethods as method}
              <option value={method.value}>{method.label}</option>
            {/each}
          </select>

          <select
            id="ai-test-generation-option"
            bind:value={selectedGenerationOption}
          >
            {#each selectedGeneration.options as option}
              <option value={option.value}>{option.label}</option>
            {/each}
          </select>

          <button
            type="button"
            class="active"
            disabled={isGenerating}
            on:click={handleGenerateData}
          >
            {#if isGenerating}
              <span class="button-symbol button-spinner" aria-hidden="true"></span>
              执行中
            {:else}
              <span class="button-symbol button-play" aria-hidden="true"></span>
              执行生成测试数据
            {/if}
          </button>
        </div>

        {#if generationMessage}
          <div class:error-status={generationError} class="status-line" role="status">
            {generationMessage}
          </div>
        {/if}
      </div>
    </div>

    <div class="ai-box test-box">
      <div class="box-title">
        <span><i class="fas fa-vial"></i> 测试功能模块</span>
      </div>

      <div class="box-body">
        <div class="inline-field compact">
          <label for="ai-test-item">
            <i class={selectedTestItem.icon}></i>
            请选择测试条目：
          </label>
          <select
            id="ai-test-item"
            bind:value={selectedTestType}
            on:change={handleTestTypeChanged}
          >
            {#each testItems as testItem}
              <option value={testItem.value}>{testItem.label}</option>
            {/each}
          </select>
        </div>

        {#if isVisualizationTestType}
          <label for="ai-test-visual-model">
            <i class="fas fa-project-diagram"></i>
            当前可视化模型：
          </label>
          <select
            id="ai-test-visual-model"
            class="blue-textbox select-textbox"
            value={selectedVisualizationModelId}
            disabled={!visualizationModelOptions.length}
            on:change={handleVisualizationModelChanged}
          >
            {#if visualizationModelOptions.length}
              {#each visualizationModelOptions as modelOption}
                <option value={modelOption.id}>{modelOption.label}</option>
              {/each}
            {:else}
              <option value={selectedModelId}>{resolvedModelLabel}</option>
            {/if}
          </select>

          <div class="inline-field">
            <label for="ai-test-visual-image">
              <i class="fas fa-image"></i>
              请选择待可视化图片：
            </label>
            <select
              id="ai-test-visual-image"
              bind:value={selectedImage}
              on:change={handleVisualizationImageChanged}
              disabled={!imageOptions.length}
            >
              {#each imageOptions as image}
                <option value={image.file}>{image.class || image.file}</option>
              {/each}
            </select>
          </div>

          <label for="ai-test-visual-image-path">
            <i class="fas fa-link"></i>
            所选择的可视化图片相对路径：
          </label>
          <input
            id="ai-test-visual-image-path"
            class="blue-textbox"
            type="text"
            value={selectedVisualizationImagePath}
            readonly
          />

          {#if selectedVisualizationImage}
            <div class="visualization-preview">
              <img
                src={selectedVisualizationImage.src}
                alt={selectedVisualizationImage.class || "selected image"}
              />
              <span>{selectedVisualizationImage.class}</span>
            </div>
          {/if}

          <label for="ai-test-output">
            <i class="fas fa-clipboard-check"></i>
            可视化状态：
          </label>
          <input
            id="ai-test-output"
            class:error={errorMessage}
            class="blue-textbox"
            type="text"
            value={resultText}
            readonly
          />

          <div class="visualization-actions">
            <button
              type="button"
              class="active"
              disabled={!selectedVisualizationImagePath}
              on:click={requestAutoVisualization}
            >
              <i class="fas fa-play-circle"></i>
              自动展示输入到输出
            </button>
          </div>
        {:else}
          <div class="inline-field">
            <label for="ai-test-data-input">
              <i class="fas fa-folder-open"></i>
              请选择测试数据所在文件夹：
            </label>
            <input
              id="ai-test-data-input"
              class="file-picker-input"
              type="file"
              webkitdirectory=""
              directory=""
              multiple
              on:change={handleDataFolderChanged}
            />
            <label class="file-picker-button" for="ai-test-data-input">
              Choose Folder
            </label>
            {#if selectedDataFolderName}
              <span class="file-picker-name">{selectedDataFolderName}</span>
            {/if}
          </div>

          <label for="ai-test-data-path">
            <i class="fas fa-link"></i>
            所选择的测试数据所在文件夹相对路径：
          </label>
          <input
            id="ai-test-data-path"
            class="blue-textbox"
            type="text"
            bind:value={testDataPath}
            readonly
          />

          <label for="ai-test-output">
            <i class="fas fa-clipboard-check"></i>
            测试结果：
          </label>
          <input
            id="ai-test-output"
            class:error={errorMessage}
            class="blue-textbox"
            type="text"
            value={resultText}
            readonly
          />

          <button
            type="button"
            class="active test-action"
            disabled={isTesting}
            on:click={runTest}
          >
            {#if isTesting}
              <span class="button-symbol button-spinner" aria-hidden="true"></span>
              执行中
            {:else}
              <span class="button-symbol button-play" aria-hidden="true"></span>
              执行模型测试
            {/if}
          </button>
        {/if}
      </div>
    </div>

    <div class="ai-box result-box">
      <div class="box-title">
        <span><i class="fas fa-chart-pie"></i> 测试结果可视化模块</span>
      </div>

      <div class="box-body result-body">
        <AiTestResultChart type={activeChartType} {result} />
        {#if false}
          <div class="axis-chart" role="img" aria-label="Robustness accuracy bar chart">
            <div class="y-axis">
              {#each [100, 80, 60, 40, 20, 0] as tick}
                <span>{tick}%</span>
              {/each}
            </div>
            <div class="plot-area">
              {#each [100, 80, 60, 40, 20, 0] as tick}
                <span class="grid-line" style={`bottom: ${tick}%;`}></span>
              {/each}
              <div class="bar-row">
                {#each robustnessChartData as item}
                  <div class="bar-slot">
                    <span
                      class="rounded-bar"
                      style={`height: ${percentNumber(item.value)}%;`}
                    ></span>
                    <span class="chart-tooltip">
                      <strong>{item.label}</strong>
                      <span><i></i>{percentNumber(item.value).toFixed(1)}</span>
                    </span>
                    <span class="x-label">{item.label}</span>
                  </div>
                {/each}
              </div>
            </div>
          </div>
        {:else if false}
          <div class="axis-chart" role="img" aria-label="Adversary accuracy line chart">
            <div class="y-axis">
              {#each [100, 80, 60, 40, 20, 0] as tick}
                <span>{tick}%</span>
              {/each}
            </div>
            <div class="plot-area line-plot">
              {#each [100, 80, 60, 40, 20, 0] as tick}
                <span class="grid-line" style={`bottom: ${tick}%;`}></span>
              {/each}
              <svg viewBox="0 0 360 160" preserveAspectRatio="none" aria-hidden="true">
                <polyline points={adversaryLinePoints}></polyline>
                {#each adversaryChartData as item, index}
                  <circle
                    cx={28 + (314 * index) / Math.max(adversaryChartData.length - 1, 1)}
                    cy={126 - (percentNumber(item.value) / 100) * 108}
                    r="4"
                  ></circle>
                {/each}
              </svg>
              <div class="line-labels">
                {#each adversaryChartData as item}
                  <span>{item.label}</span>
                {/each}
              </div>
            </div>
          </div>
        {:else if false}
          <div class="axis-chart" role="img" aria-label="Coverage grouped bar chart">
            <div class="y-axis">
              {#each [100, 80, 60, 40, 20, 0] as tick}
                <span>{tick}%</span>
              {/each}
            </div>
            <div class="plot-area">
              {#each [100, 80, 60, 40, 20, 0] as tick}
                <span class="grid-line" style={`bottom: ${tick}%;`}></span>
              {/each}
              <div class="coverage-row">
                {#each coverageChartData as group}
                  <div class="coverage-group">
                    <div class="coverage-bars">
                      {#each coverageSeries as metric}
                        <span
                          class="rounded-bar coverage-bar"
                          style={`height: ${percentNumber(group[metric.key])}%;`}
                        ></span>
                      {/each}
                    </div>
                    <span class="chart-tooltip coverage-tooltip">
                      <strong>{group.display}</strong>
                      {#each coverageSeries as metric}
                        <span>
                          <i></i>{metric.label}: {percentNumber(group[metric.key]).toFixed(1)}
                        </span>
                      {/each}
                    </span>
                    <span class="x-label">{group.display}</span>
                  </div>
                {/each}
              </div>
            </div>
          </div>
        {:else if false}
          <div
            class="echart-donut"
            class:accuracy-emphasis={hoveredAccuracySlice === "accuracy"}
            class:error-emphasis={hoveredAccuracySlice === "error"}
            role="img"
            aria-label={`Accuracy ${percentText(correctRate)}, error ${percentText(errorRate)}`}
            style={`--accuracy-pct: ${percentNumber(correctRate)}; --error-pct: ${percentNumber(errorRate)};`}
          >
            <svg class="donut-svg" viewBox="0 0 300 220" aria-hidden="true">
              <defs>
                <filter id="accuracyPieShadow" x="-45%" y="-45%" width="190%" height="190%">
                  <feDropShadow
                    dx="0"
                    dy="0"
                    stdDeviation="8"
                    flood-color="rgba(0, 0, 0, 0.22)"
                  />
                </filter>
              </defs>
              <circle
                class="donut-segment donut-error"
                cx="150"
                cy="132"
                r="62"
                pathLength="100"
              ></circle>
              <circle
                class="donut-segment donut-accuracy"
                cx="150"
                cy="132"
                r="62"
                pathLength="100"
              ></circle>
              <polyline class="donut-guide accuracy-guide" points="105,184 91,198 72,198"></polyline>
              <polyline class="donut-guide error-guide" points="166,71 176,58 198,58"></polyline>
            </svg>

            <button
              type="button"
              class="donut-hit accuracy-hit"
              on:mouseenter={() => (hoveredAccuracySlice = "accuracy")}
              on:mouseleave={() => (hoveredAccuracySlice = "")}
              on:focus={() => (hoveredAccuracySlice = "accuracy")}
              on:blur={() => (hoveredAccuracySlice = "")}
            >
              <span class="donut-tooltip">
                <strong>&#20934;&#30830;&#29575;</strong>
                <span><i></i>{percentNumber(correctRate).toFixed(1)}</span>
              </span>
            </button>
            <button
              type="button"
              class="donut-hit error-hit"
              on:mouseenter={() => (hoveredAccuracySlice = "error")}
              on:mouseleave={() => (hoveredAccuracySlice = "")}
              on:focus={() => (hoveredAccuracySlice = "error")}
              on:blur={() => (hoveredAccuracySlice = "")}
            >
              <span class="donut-tooltip">
                <strong>&#38169;&#35823;&#29575;</strong>
                <span><i></i>{percentNumber(errorRate).toFixed(1)}</span>
              </span>
            </button>
          </div>

          <div class="legend">
            <span><i class="legend-dot correct-dot"></i> 准确率</span>
            <span><i class="legend-dot error-dot"></i> 错误率</span>
          </div>

          <div class="rate-label rate-left">
            <strong>{percentNumber(correctRate).toFixed(1)}%</strong>
            <span>准确率</span>
          </div>
          <div class="rate-label rate-right">
            <strong>{percentNumber(errorRate).toFixed(1)}%</strong>
            <span>错误率</span>
          </div>
        {/if}

        {#if result}
          <div class="result-count">
            <i class="fas fa-check-circle"></i>
            {result.correct || 0} / {result.total || 0}
          </div>
        {/if}
      </div>
    </div>
  </div>
</section>

<style>
  .ai-test-dashboard {
    width: 100%;
    min-height: 690px;
    box-sizing: border-box;
    padding: 8px 36px 30px;
    color: #ffffff;
    background: #0b347d;
    font-family:
      "Microsoft YaHei",
      -apple-system,
      BlinkMacSystemFont,
      "Segoe UI",
      sans-serif;
  }

  :global(body.ai-test-ui-mode) {
    background: #0b347d;
  }

  :global(body.ai-test-ui-mode .app-header),
  :global(body.ai-test-ui-mode recommender-overlay) {
    display: none !important;
  }

  .dashboard-title {
    position: relative;
    width: 100%;
    height: 52px;
    margin-bottom: 8px;
    text-align: center;
    background: #0b347d;
    overflow: visible;
  }

  .dashboard-title-bg {
    position: absolute;
    left: 0;
    top: -20%;
    width: 100%;
    height: 140%;
    object-fit: fill;
    pointer-events: none;
  }

  .title-text {
    position: relative;
    display: block;
    min-width: 0;
    min-height: 0;
    padding: 0;
    color: #0efcff;
    font-size: 30px;
    font-weight: 800;
    line-height: 52px;
    letter-spacing: 0;
    text-shadow: 0 0 10px rgba(14, 252, 255, 0.65);
  }

  .dashboard-grid {
    display: grid;
    grid-template-columns: minmax(360px, 1fr) minmax(360px, 1fr);
    gap: 14px;
  }

  .ai-box {
    min-height: 284px;
    position: relative;
    border: 2px solid #005ce8;
    box-shadow:
      inset 0 0 10px rgba(7, 118, 181, 0.7),
      0 0 7px rgba(0, 92, 232, 0.7);
    border-radius: 7px;
    background:
      linear-gradient(rgba(13, 59, 151, 0.82), rgba(13, 59, 151, 0.82)),
      var(--ai-box-corner-bg)
      right top / 56px 52px
      no-repeat;
  }

  .ai-box:before {
    position: absolute;
    width: calc(100% - 6px);
    height: calc(100% - 6px);
    border: 2px solid #005ce8;
    left: 1px;
    top: 1px;
    content: "";
    border-radius: 5px;
    opacity: 0.6;
    z-index: 1;
    pointer-events: none;
  }

  .box-title {
    height: 30px;
    text-align: center;
    overflow: hidden;
  }

  .box-title span {
    position: relative;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    min-width: 220px;
    height: 28px;
    padding: 0 16px;
    color: #ffffff;
    font-size: 18px;
    font-weight: 500;
    line-height: 28px;
    background: var(--ai-title-strip-bg) repeat-x;
  }

  .box-title span:before,
  .box-title span:after {
    content: "";
    position: absolute;
    top: 0;
    width: 51px;
    height: 27px;
    pointer-events: none;
  }

  .box-title span:before {
    left: -51px;
    background: var(--ai-title-left-bg) no-repeat;
  }

  .box-title span:after {
    right: -51px;
    background: var(--ai-title-right-bg) no-repeat;
  }

  .box-title i {
    color: #0efcff;
  }

  .box-body {
    position: relative;
    z-index: 2;
    padding: 22px 12px 42px;
  }

  .inline-field {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 18px;
  }

  .inline-field.compact {
    margin-bottom: 24px;
  }

  label {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    margin: 0 0 4px;
    color: #ffffff;
    font-size: 18px;
    font-weight: 400;
    line-height: 1.35;
  }

  label i {
    width: 18px;
    color: #0efcff;
    text-align: center;
  }

  input,
  select {
    box-sizing: border-box;
    min-height: 24px;
    border: 1px solid #0374d4;
    border-radius: 0;
    padding: 2px 8px;
    color: #000000;
    font-size: 14px;
    background-color: #ffffff;
  }

  input[type="file"] {
    color: #ffffff;
    max-width: 260px;
    padding: 0;
    border: 0;
    background: transparent;
  }

  .file-picker-input {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0 0 0 0);
    white-space: nowrap;
    border: 0;
  }

  .file-picker-button {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-height: 26px;
    margin: 0;
    padding: 2px 10px;
    border: 1px solid #0efcff;
    color: #ffffff;
    background: #0c64bd;
    font-size: 14px;
    line-height: 1.2;
    cursor: pointer;
  }

  .file-picker-button:hover,
  .file-picker-input:focus + .file-picker-button {
    background: #1080de;
  }

  .file-picker-name {
    max-width: 240px;
    color: #0efcff;
    font-size: 13px;
    line-height: 1.3;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .blue-textbox {
    width: 100%;
    height: 24px;
    display: block;
    margin-bottom: 22px;
    background-color: #0efcff;
  }

  .select-textbox {
    color: #000000;
    appearance: auto;
  }

  .blue-textbox.error {
    color: #7f1d1d;
    background-color: #fecaca;
    border-color: #ef4444;
  }

  .model-note,
  .status-line {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    color: #0efcff;
    font-size: 13px;
  }

  .status-line.error-status {
    color: #ffcbcb;
  }

  .visualization-preview {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    margin: 0 0 16px;
    color: #0efcff;
    font-size: 15px;
  }

  .visualization-preview img {
    width: 56px;
    height: 56px;
    border: 2px solid #0efcff;
    border-radius: 3px;
    background: #000000;
    image-rendering: pixelated;
    object-fit: cover;
  }

  .visualization-actions {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: -6px;
  }

  .generate-controls {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
  }

  button.active {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    min-height: 26px;
    border: 1px solid #0efcff;
    border-radius: 2px;
    padding: 2px 8px;
    color: #ffffff;
    background: rgba(0, 143, 220, 0.8);
    box-shadow: inset 0 0 6px rgba(14, 252, 255, 0.42);
    font-size: 14px;
    cursor: pointer;
  }

  button.active:hover:not(:disabled) {
    background: rgba(0, 168, 236, 0.95);
  }

  button.active:disabled {
    cursor: not-allowed;
    opacity: 0.7;
  }

  .button-symbol {
    width: 13px;
    height: 13px;
    display: inline-block;
    flex: 0 0 auto;
  }

  .button-play {
    width: 0;
    height: 0;
    border-top: 7px solid transparent;
    border-bottom: 7px solid transparent;
    border-left: 11px solid #ffffff;
  }

  .button-spinner {
    border: 2px dotted rgba(255, 255, 255, 0.95);
    border-radius: 50%;
    animation: aiButtonSpin 900ms linear infinite;
  }

  .test-action {
    position: absolute;
    right: 28px;
    bottom: 14px;
  }

  .result-body {
    min-height: 228px;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    padding-bottom: 22px;
  }

  .axis-chart {
    width: 100%;
    height: 220px;
    display: grid;
    grid-template-columns: 52px minmax(0, 1fr);
    gap: 8px;
    padding: 12px 18px 4px 4px;
    box-sizing: border-box;
  }

  .y-axis {
    height: 178px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    align-items: flex-end;
    color: #ffffff;
    font-size: 12px;
    line-height: 1;
  }

  .plot-area {
    position: relative;
    height: 178px;
    min-width: 0;
  }

  .grid-line {
    position: absolute;
    left: 0;
    right: 0;
    height: 0;
    border-top: 1px dotted rgba(255, 255, 255, 0.13);
    transform: translateY(1px);
  }

  .bar-row {
    position: absolute;
    inset: 0;
    display: grid;
    grid-template-columns: repeat(6, minmax(58px, 1fr));
    align-items: end;
    gap: 18px;
  }

  .bar-slot,
  .coverage-group {
    height: 100%;
    min-width: 0;
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-end;
  }

  .bar-slot:before,
  .coverage-group:before {
    content: "";
    position: absolute;
    inset: 0;
    bottom: 24px;
    background: rgba(126, 177, 255, 0.24);
    opacity: 0;
    pointer-events: none;
    transition: opacity 120ms ease-out;
  }

  .bar-slot:hover:before,
  .coverage-group:hover:before,
  .bar-slot:focus-within:before,
  .coverage-group:focus-within:before {
    opacity: 1;
  }

  .rounded-bar {
    width: 30%;
    min-width: 32px;
    max-width: 42px;
    min-height: 3px;
    display: block;
    border-radius: 999px;
    background: linear-gradient(180deg, #01fdcc 0%, #11a1d8 80%);
    position: relative;
    z-index: 1;
    transition: filter 120ms ease-out, transform 120ms ease-out;
  }

  .bar-slot:hover .rounded-bar,
  .coverage-group:hover .rounded-bar,
  .bar-slot:focus-within .rounded-bar,
  .coverage-group:focus-within .rounded-bar {
    filter: brightness(1.08);
    transform: translateY(-2px);
  }

  .chart-tooltip {
    position: absolute;
    left: calc(50% + 8px);
    bottom: 44px;
    z-index: 3;
    min-width: 72px;
    padding: 8px 10px;
    border-radius: 3px;
    color: #ffffff;
    background: rgba(37, 42, 50, 0.9);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.24);
    font-size: 12px;
    line-height: 1.3;
    opacity: 0;
    pointer-events: none;
    transform: translate(4px, 6px);
    transition: opacity 120ms ease-out, transform 120ms ease-out;
  }

  .chart-tooltip strong,
  .chart-tooltip span {
    display: block;
    white-space: nowrap;
  }

  .chart-tooltip span {
    margin-top: 5px;
  }

  .chart-tooltip i {
    width: 11px;
    height: 11px;
    display: inline-block;
    margin-right: 6px;
    border-radius: 50%;
    vertical-align: -1px;
    background: #01fdcc;
  }

  .bar-slot:hover .chart-tooltip,
  .coverage-group:hover .chart-tooltip,
  .bar-slot:focus-within .chart-tooltip,
  .coverage-group:focus-within .chart-tooltip {
    opacity: 1;
    transform: translate(0, 0);
  }

  .coverage-tooltip {
    left: calc(50% + 20px);
    bottom: 52px;
  }

  .x-label {
    width: 100%;
    min-height: 24px;
    margin-top: 8px;
    color: #ffffff;
    font-size: 12px;
    line-height: 1.2;
    text-align: center;
    white-space: nowrap;
  }

  .line-plot svg {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    overflow: visible;
  }

  .line-plot polyline {
    fill: none;
    stroke: #01fdcc;
    stroke-width: 3;
    vector-effect: non-scaling-stroke;
  }

  .line-plot circle {
    fill: #01fdcc;
    stroke: #01fdcc;
    vector-effect: non-scaling-stroke;
  }

  .line-labels {
    position: absolute;
    left: 28px;
    right: 18px;
    bottom: -2px;
    display: flex;
    justify-content: space-between;
    color: #ffffff;
    font-size: 12px;
    font-weight: 600;
  }

  .coverage-row {
    position: absolute;
    inset: 0;
    display: grid;
    grid-template-columns: repeat(2, minmax(140px, 1fr));
    align-items: end;
    column-gap: 76px;
  }

  .coverage-bars {
    width: 126px;
    height: calc(100% - 32px);
    display: flex;
    align-items: flex-end;
    justify-content: center;
    gap: 4px;
  }

  .coverage-bar {
    width: 34px;
    min-width: 34px;
  }

  .echart-donut {
    width: 300px;
    height: 220px;
    position: relative;
    flex: 0 0 300px;
    margin-top: 4px;
  }

  .donut-svg {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    overflow: visible;
  }

  .donut-segment {
    fill: none;
    stroke-width: 62;
    transform: rotate(-90deg);
    transform-origin: 150px 132px;
    filter: url("#accuracyPieShadow");
    transition: transform 180ms ease-out, filter 180ms ease-out;
  }

  .donut-error {
    stroke: #0086e5;
  }

  .donut-accuracy {
    stroke: #30c5ed;
    stroke-dasharray: var(--accuracy-pct) 100;
    animation: donutReveal 850ms cubic-bezier(0.22, 0.72, 0.18, 1) both;
  }

  .echart-donut.accuracy-emphasis .donut-accuracy,
  .echart-donut.error-emphasis .donut-error {
    transform: rotate(-90deg) scale(1.07);
    filter: url("#accuracyPieShadow") brightness(1.08);
  }

  .donut-guide {
    fill: none;
    stroke: rgb(98, 137, 169);
    stroke-width: 1;
    opacity: 0;
    animation: labelLineReveal 320ms ease-out 720ms both;
  }

  .donut-hit {
    position: absolute;
    z-index: 4;
    padding: 0;
    border: 0;
    background: transparent;
    cursor: default;
    outline: none;
  }

  .accuracy-hit {
    left: 68px;
    top: 62px;
    width: 146px;
    height: 146px;
    border-radius: 50%;
  }

  .error-hit {
    left: 156px;
    top: 44px;
    width: 64px;
    height: 54px;
  }

  .donut-tooltip {
    position: absolute;
    left: 70%;
    top: 54%;
    z-index: 5;
    min-width: 72px;
    padding: 7px 9px;
    border-radius: 3px;
    color: #ffffff;
    background: rgba(37, 42, 50, 0.9);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.24);
    font-size: 12px;
    line-height: 1.3;
    opacity: 0;
    pointer-events: none;
    transform: translate(4px, 6px);
    transition: opacity 120ms ease-out, transform 120ms ease-out;
  }

  .error-hit .donut-tooltip {
    left: 34px;
    top: 8px;
  }

  .donut-tooltip strong,
  .donut-tooltip span {
    display: block;
    white-space: nowrap;
  }

  .donut-tooltip span {
    margin-top: 5px;
  }

  .donut-tooltip i {
    width: 11px;
    height: 11px;
    display: inline-block;
    margin-right: 6px;
    border-radius: 50%;
    vertical-align: -1px;
    background: #01fdcc;
  }

  .donut-hit:hover .donut-tooltip,
  .donut-hit:focus .donut-tooltip {
    opacity: 1;
    transform: translate(0, 0);
  }

  .legend {
    position: absolute;
    top: 20px;
    left: 50%;
    display: flex;
    gap: 14px;
    transform: translateX(-50%);
    color: rgba(255, 255, 255, 0.42);
    font-size: 12px;
  }

  .legend span {
    display: inline-flex;
    align-items: center;
    gap: 5px;
  }

  .legend-dot {
    width: 11px;
    height: 11px;
    display: inline-block;
    border-radius: 3px;
  }

  .correct-dot {
    background: #0a90d8;
  }

  .error-dot {
    background: #138bd6;
  }

  .rate-label {
    position: absolute;
    z-index: 2;
    display: flex;
    flex-direction: column;
    align-items: center;
    color: rgb(98, 137, 169);
    font-size: 12px;
    line-height: 1.1;
    opacity: 0;
    animation: labelLineReveal 320ms ease-out 720ms both;
  }

  .rate-label strong {
    color: rgb(241, 246, 104);
    font-size: 16px;
    font-weight: 800;
    line-height: 1;
  }

  .rate-left {
    left: calc(50% - 110px);
    top: 176px;
  }

  .rate-right {
    left: calc(50% + 48px);
    top: 58px;
  }

  @keyframes donutReveal {
    from {
      stroke-dasharray: 0 100;
    }
    to {
      stroke-dasharray: var(--accuracy-pct) 100;
    }
  }

  @keyframes labelLineReveal {
    from {
      opacity: 0;
      transform: translateY(4px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  @keyframes aiButtonSpin {
    to {
      transform: rotate(360deg);
    }
  }

  .result-count {
    position: absolute;
    right: 16px;
    bottom: 12px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: #0efcff;
    font-size: 13px;
  }

  @media (max-width: 1180px) {
    .ai-test-dashboard {
      padding: 8px 14px 24px;
    }

    .dashboard-title {
      height: 48px;
    }

    .title-text {
      min-width: 0;
      font-size: 24px;
      line-height: 48px;
      padding: 0;
    }

    .dashboard-grid {
      grid-template-columns: 1fr;
    }

    .ai-box {
      min-height: 270px;
    }
  }

</style>
