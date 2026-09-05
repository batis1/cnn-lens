<script>
  import { onDestroy, onMount } from "svelte";
  import * as echarts from "echarts";

  export let type = "accuracy";
  export let result = null;

  const robustnessLabels = {
    gauss_noise: "高斯噪声",
    poisson_noise: "泊松噪声",
    salt_pepper_noise: "椒盐噪声",
    rotation: "旋转变换",
    scale: "放缩变换",
    translation: "平移变换",
  };

  let chartElement;
  let chart;
  let resizeObserver;

  $: option = buildOption(type, result);
  $: if (chart && option) {
    chart.setOption(option, true);
  }

  onMount(() => {
    chart = echarts.init(chartElement);
    chart.setOption(option, true);
    resizeObserver = new ResizeObserver(() => chart?.resize());
    resizeObserver.observe(chartElement);
  });

  onDestroy(() => {
    resizeObserver?.disconnect();
    chart?.dispose();
  });

  const percent = (value) =>
    Number.isFinite(Number(value)) ? (Number(value) * 100).toFixed(1) : "0.0";

  const gradient = () =>
    new echarts.graphic.LinearGradient(0, 0, 0, 1, [
      { offset: 0, color: "#01fdcc" },
      { offset: 0.8, color: "#11a1d8" },
    ]);

  const axisOption = (name) => ({
    name,
    nameTextStyle: { color: "#fff", fontSize: 14 },
    axisLine: { show: false },
    axisTick: { show: false },
    axisLabel: { color: "#fff", fontSize: 12, formatter: "{value}%" },
    splitLine: {
      lineStyle: { color: "rgba(255,255,255,.1)", type: "dotted" },
    },
    interval: 20,
    max: 100,
  });

  const cartesianBase = (labels, yAxisName, pointerType) => ({
    animationDuration: 1000,
    animationDurationUpdate: 650,
    animationEasing: "cubicOut",
    tooltip: {
      trigger: "axis",
      axisPointer: { type: pointerType },
      backgroundColor: "rgba(37,42,50,.92)",
      borderWidth: 0,
      textStyle: { color: "#fff" },
    },
    grid: { left: "0%", top: "15px", right: "0%", bottom: "0%", containLabel: true },
    xAxis: {
      data: labels,
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: "#fff", fontSize: 12, fontWeight: 600 },
    },
    yAxis: axisOption(yAxisName),
  });

  function buildOption(chartType, payload) {
    if (chartType === "robustness") {
      let entries = Object.entries(robustnessLabels);
      return {
        ...cartesianBase(entries.map(([, label]) => label), "准确率(%)", "shadow"),
        series: [{
          type: "bar",
          barWidth: "30%",
          showBackground: false,
          emphasis: { focus: "series", itemStyle: { shadowBlur: 14, shadowColor: "rgba(1,253,204,.5)" } },
          itemStyle: { borderRadius: 50, color: gradient() },
          data: entries.map(([key]) => percent(payload?.robustness?.[key])),
        }],
      };
    }

    if (chartType === "adversary") {
      let values = payload?.adversary || {};
      let labels = Object.keys(values).length
        ? Object.keys(values)
        : ["epsilon_0.1", "epsilon_0.2", "epsilon_0.3"];
      return {
        ...cartesianBase(labels, "准确度(%)", "line"),
        series: [{
          type: "line",
          symbol: "circle",
          symbolSize: 8,
          smooth: false,
          lineStyle: { width: 3, color: gradient() },
          itemStyle: { color: "#01fdcc" },
          emphasis: { scale: 1.8, itemStyle: { shadowBlur: 12, shadowColor: "rgba(1,253,204,.65)" } },
          data: labels.map((key) => percent(values[key])),
        }],
      };
    }

    if (chartType === "coverage") {
      let coverage = Array.isArray(payload?.coverage)
        ? payload.coverage
        : payload?.coverage?.datasets || payload?.coverage?.data || [];
      let rows = coverage.length
        ? coverage
        : [
            { display: "原始数据集", hidden1: 0, hidden2: 0, full: 0 },
            { display: "新数据集", hidden1: 0, hidden2: 0, full: 0 },
          ];
      let base = cartesianBase(rows.map((row) => row.display), "覆盖率(%)", "shadow");
      return {
        ...base,
        tooltip: { ...base.tooltip, formatter: undefined },
        series: [
          ["hidden1", "hidden1"],
          ["hidden2", "hidden2"],
          ["full", "total"],
        ].map(([key, name]) => ({
          name,
          type: "bar",
          barWidth: "10%",
          barGap: "1%",
          emphasis: { focus: "series", itemStyle: { shadowBlur: 14, shadowColor: "rgba(1,253,204,.5)" } },
          itemStyle: { borderRadius: 50, color: gradient() },
          data: rows.map((row) => percent(row[key])),
        })),
      };
    }

    let hasAccuracy = Number.isFinite(Number(payload?.accuracy));
    let accuracy = hasAccuracy ? Number(percent(payload.accuracy)) : 0;
    let errors = hasAccuracy ? Number((100 - accuracy).toFixed(1)) : 0;
    return {
      animationDuration: 1000,
      animationEasing: "cubicOut",
      legend: {
        top: 20,
        left: "center",
        itemWidth: 10,
        itemHeight: 10,
        data: ["准确率", "错误率"],
        textStyle: { color: "rgba(255,255,255,.5)", fontSize: 12 },
      },
      tooltip: {
        trigger: "item",
        formatter: "{b} : {c} ({d}%)",
        backgroundColor: "rgba(37,42,50,.92)",
        borderWidth: 0,
        textStyle: { color: "#fff" },
      },
      series: [{
        name: "准确率",
        type: "pie",
        radius: ["30%", "60%"],
        center: ["50%", "60%"],
        color: ["#0086e5", "#30c5ed", "#9fe7b8", "#fedb5b", "#ff9f7d", "#fb7293", "#e7bcf2"],
        data: [
          { value: errors, name: "错误率" },
          { value: accuracy, name: "准确率" },
        ].sort((a, b) => a.value - b.value),
        roseType: "radius",
        emphasis: { scale: true, scaleSize: 10 },
        label: {
          formatter: ["{d|{d}%}", "{b|{b}}"].join("\n"),
          rich: {
            d: { color: "rgb(241,246,104)", fontSize: 14, fontWeight: "bold" },
            b: { color: "rgb(98,137,169)", fontSize: 12 },
          },
        },
        labelLine: {
          lineStyle: { color: "rgb(98,137,169)" },
          smooth: 0.2,
          length: 5,
          length2: 9,
        },
        itemStyle: { shadowColor: "rgba(0,0,0,.1)", shadowBlur: 50 },
      }],
    };
  }
</script>

<div
  class="result-chart"
  bind:this={chartElement}
  role="img"
  aria-label={`${type} test result chart`}
></div>

<style>
  .result-chart {
    width: 100%;
    height: 250px;
    min-width: 0;
  }

  @media (max-width: 1180px) {
    .result-chart {
      height: 270px;
    }
  }
</style>
