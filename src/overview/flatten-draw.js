/* global d3, SmoothScroll */

import {
  svgStore, vSpaceAroundGapStore, hSpaceAroundGapStore, cnnStore,
  nodeCoordinateStore, selectedScaleLevelStore, cnnLayerRangesStore,
  cnnLayerMinMaxStore, isInSoftmaxStore, softmaxDetailViewStore,
  denseDetailViewStore, hoverInfoStore, allowsSoftmaxAnimationStore, detailedModeStore
} from '../stores.js';
import {
  getOutputKnot, getInputKnot, gappedColorScale, getMidCoords
} from './draw-utils.js';
import {
  drawIntermediateLayerLegend, moveLayerX, addOverlayGradient,
  drawArrow
} from './intermediate-utils.js';
import { overviewConfig } from '../config.js';

// Configs
const layerColorScales = overviewConfig.layerColorScales;
const edgeInitColor = overviewConfig.edgeInitColor;
const edgeHoverColor = overviewConfig.edgeHoverColor;
const edgeStrokeWidth = overviewConfig.edgeStrokeWidth;
const nodeLength = overviewConfig.nodeLength;
const plusSymbolRadius = overviewConfig.plusSymbolRadius;
const intermediateColor = overviewConfig.intermediateColor;
const kernelRectLength = overviewConfig.kernelRectLength;
const svgPaddings = overviewConfig.svgPaddings;
const gapRatio = overviewConfig.gapRatio;
const classList = overviewConfig.classLists;
const formater = d3.format('.4f');

const isStyleTestMode = () =>
  Boolean(document.querySelector('.overview.style-test-mode'));

const formatLayerLabel = (name) => {
  if (!name) {
    return '';
  }

  let normalized = String(name).toLowerCase();
  if (normalized.includes('flatten')) {
    return 'Flatten';
  }
  if (normalized.includes('avg_pool')) {
    return 'Avg Pool';
  }
  if (normalized.includes('max_pool') || normalized.includes('pool')) {
    return 'Pool';
  }
  if (normalized.includes('sigmoid')) {
    return 'Sigmoid';
  }
  if (normalized.includes('relu')) {
    return 'ReLU';
  }
  if (normalized.includes('dense')) {
    let suffix = normalized.match(/dense[_-]?(\d+)/)?.[1];
    return suffix ? `Dense ${suffix}` : 'Dense';
  }
  if (normalized.includes('conv')) {
    return 'Conv';
  }

  return String(name)
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
};

// Shared variables
let svg = undefined;
svgStore.subscribe(value => { svg = value; })

let vSpaceAroundGap = undefined;
vSpaceAroundGapStore.subscribe(value => { vSpaceAroundGap = value; })

let hSpaceAroundGap = undefined;
hSpaceAroundGapStore.subscribe(value => { hSpaceAroundGap = value; })

let cnn = undefined;
cnnStore.subscribe(value => { cnn = value; })

let nodeCoordinate = undefined;
nodeCoordinateStore.subscribe(value => { nodeCoordinate = value; })

let selectedScaleLevel = undefined;
selectedScaleLevelStore.subscribe(value => { selectedScaleLevel = value; })

let cnnLayerRanges = undefined;
cnnLayerRangesStore.subscribe(value => { cnnLayerRanges = value; })

let cnnLayerMinMax = undefined;
cnnLayerMinMaxStore.subscribe(value => { cnnLayerMinMax = value; })

let isInSoftmax = undefined;
isInSoftmaxStore.subscribe(value => { isInSoftmax = value; })

let allowsSoftmaxAnimation = undefined;
allowsSoftmaxAnimationStore.subscribe(value => { allowsSoftmaxAnimation = value; })

let softmaxDetailViewInfo = undefined;
softmaxDetailViewStore.subscribe(value => { softmaxDetailViewInfo = value; })

let hoverInfo = undefined;
hoverInfoStore.subscribe(value => { hoverInfo = value; })

let detailedMode = undefined;
detailedModeStore.subscribe(value => { detailedMode = value; })

let hasInitialized = false;
let logits = [];
let flattenFactoredFDict = {};
let scalarNodeImageCache = new Map();
let isSoftmaxTransitioning = false;
let denseDetailState = undefined;
let denseDetailSelections = [];

const updateStyleTestDenseLayerHierarchy = (activeLayerIndex = null) => {
  if (!svg || !isStyleTestMode()) {
    return;
  }

  let hasActiveLayer = Number.isFinite(activeLayerIndex);
  svg.selectAll('.classifier-hidden-layer')
    .classed('is-style-active', (_, layerIndex) =>
      hasActiveLayer && layerIndex === activeLayerIndex)
    .classed('is-style-muted', (_, layerIndex) =>
      !hasActiveLayer || layerIndex !== activeLayerIndex);
};

// AI_TEST_UI_INTEGRATION:
// Detail panels should follow the actual CNN SVG position. The AI_Test_UI
// dashboard can push the SVG lower than the original fixed 100px offset.
const getSvgPageOffset = () => {
  let svgElement = document.getElementById('cnn-svg');
  if (!svgElement) {
    return { top: 100, left: 0 };
  }

  let rect = svgElement.getBoundingClientRect();
  return {
    top: rect.top + window.scrollY,
    left: rect.left + window.scrollX,
  };
};

const positionFloatingDetailView = (detailview, topWithinSvg, leftWithinSvg) => {
  if (!detailview) {
    return;
  }

  let svgOffset = getSvgPageOffset();
  detailview.style.top = `${svgOffset.top + topWithinSvg}px`;
  detailview.style.left = `${svgOffset.left + leftWithinSvg}px`;
  detailview.style.position = 'absolute';
  detailview.style.zIndex = '30';
};

const denseActivationColors = {
  activeFill: '#BFE7F7',
  activeStroke: '#67BCE7',
  activeText: '#3D8CB8',
  inactiveFill: '#F1F5F8',
  inactiveStroke: '#B8C5D0',
  inactiveText: '#8B97A3',
};

const getDenseActivationStyle = (node) => {
  let isActive = (node?.output || 0) > 0;
  return {
    isActive,
    fill: isActive ? denseActivationColors.activeFill : denseActivationColors.inactiveFill,
    stroke: isActive ? denseActivationColors.activeStroke : denseActivationColors.inactiveStroke,
    text: isActive ? denseActivationColors.activeText : denseActivationColors.inactiveText,
    glow: isActive
      ? 'drop-shadow(0 0 5px rgba(103, 188, 231, 0.34))'
      : null,
  };
};

const getSafeLegendMinMax = (values, fallbackMinMax = { min: 0, max: 1 }) => {
  let numericValues = values.filter((value) => Number.isFinite(value));
  let extent = d3.extent(numericValues);
  let min = Number.isFinite(extent[0]) ? extent[0] : fallbackMinMax.min;
  let max = Number.isFinite(extent[1]) ? extent[1] : fallbackMinMax.max;
  min = Math.min(0, min);
  max = Math.max(0, max);

  if (min === max) {
    max = min + 1;
  }

  return { min, max };
}

const drawDenseActivationLegend = ({
  group, x, y, width, legendLayerIndex, values = []
}) => {
  let legendHeight = 5;
  let minMax = getSafeLegendMinMax(values, { min: 0, max: 1 });
  let totalRange = minMax.max - minMax.min || 1;
  let zeroLocation = Math.max(0, Math.min(1, (0 - minMax.min) / totalRange));
  let transitionWidth = 0.035;
  let gradientName = `dense-activation-gradient-${legendLayerIndex}`;

  addOverlayGradient(gradientName, [
    {
      offset: 0,
      color: denseActivationColors.inactiveFill,
      opacity: 1,
    },
    {
      offset: Math.max(0, zeroLocation - transitionWidth),
      color: denseActivationColors.inactiveFill,
      opacity: 1,
    },
    {
      offset: Math.min(1, zeroLocation + transitionWidth),
      color: denseActivationColors.activeFill,
      opacity: 1,
    },
    {
      offset: 1,
      color: denseActivationColors.activeFill,
      opacity: 1,
    },
  ], group);

  let legend = group.append('g')
    .attr('class', `intermediate-legend-${legendLayerIndex} classifier-dense-activation-legend`)
    .attr('transform', `translate(${x}, ${y})`)
    .attr('data-min', minMax.min)
    .attr('data-max', minMax.max)
    .attr('data-base-width', width);

  let legendScale = d3.scaleLinear()
    .range([0, width - 1.2])
    .domain([minMax.min, minMax.max]);

  let legendAxis = d3.axisBottom()
    .scale(legendScale)
    .tickFormat(d3.format('.2f'))
    .tickValues([minMax.min, minMax.max]);

  let legendGroup = legend.append('g')
    .attr('class', 'classifier-dense-activation-legend-axis')
    .attr('transform', `translate(0, ${legendHeight - 3})`)
    .call(legendAxis);

  legendGroup.selectAll('text')
    .style('font-size', '9px')
    .style('fill', intermediateColor);

  legendGroup.selectAll('path, line')
    .style('stroke', intermediateColor);

  legend.append('rect')
    .attr('class', 'classifier-dense-activation-legend-bar')
    .attr('width', width)
    .attr('height', legendHeight)
    .style('fill', `url(#${gradientName})`);
}

const drawDenseActivationLegendAxis = (legend, width) => {
  let min = +legend.attr('data-min');
  let max = +legend.attr('data-max');

  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return;
  }

  let legendScale = d3.scaleLinear()
    .range([0, width - 1.2])
    .domain([min, max]);

  let legendAxis = d3.axisBottom()
    .scale(legendScale)
    .tickFormat(d3.format('.2f'))
    .tickValues([min, max]);

  let legendGroup = legend.select('.classifier-dense-activation-legend-axis')
    .call(legendAxis);

  legendGroup.selectAll('text')
    .style('font-size', '9px')
    .style('fill', intermediateColor);

  legendGroup.selectAll('path, line')
    .style('stroke', intermediateColor);
}

const setDenseActivationLegendWidth = (legend, width, duration = 0) => {
  drawDenseActivationLegendAxis(legend, width);

  let bar = legend.select('.classifier-dense-activation-legend-bar')
    .interrupt('softmax');

  if (duration > 0) {
    bar.transition('softmax')
      .duration(duration)
      .ease(d3.easeCubicInOut)
      .attr('width', width);
    return;
  }

  bar.attr('width', width);
}

const restoreDenseDetailTransform = function () {
  let element = d3.select(this);
  let baseTransform = element.attr('data-dense-detail-base-transform') || '';
  element.attr('transform', baseTransform);
  element.attr('data-dense-detail-base-transform', null);
}

const restoreDenseDetailOpacity = function () {
  let element = d3.select(this);
  let baseOpacity = element.attr('data-dense-detail-base-opacity');
  element.style('opacity', baseOpacity === null ? null : baseOpacity);
  element.attr('data-dense-detail-base-opacity', null);
}

const getDenseDetailBaseTransform = (selection) => {
  let baseTransform = selection.attr('data-dense-detail-base-transform') ||
    selection.attr('transform') || '';
  selection.attr('data-dense-detail-base-transform', baseTransform);
  return baseTransform;
}

const transitionLegendTransform = (legend, transform, duration) => {
  legend.interrupt('softmax');
  if (duration > 0) {
    legend.transition('softmax')
      .duration(duration)
      .ease(d3.easeCubicInOut)
      .attr('transform', transform);
    return;
  }

  legend.attr('transform', transform);
}

const shiftDenseDetailLegends = ({
  prevFeatureLayerIndex,
  layerLayouts,
  layerIndex,
  nodeSize,
  upstreamShift,
  duration,
}) => {
  let flattenLegendSelector =
    `.intermediate-legend-${prevFeatureLayerIndex}.classifier-flatten-activation-legend`;
  let denseLegendSelector =
    `.intermediate-legend-${prevFeatureLayerIndex}.classifier-dense-activation-legend`;

  svg.selectAll(flattenLegendSelector)
    .each(function () {
      let legend = d3.select(this);
      let baseTransform = getDenseDetailBaseTransform(legend);
      transitionLegendTransform(
        legend,
        `${baseTransform} translate(${upstreamShift}, 0)`,
        duration,
      );
    });

  svg.selectAll(denseLegendSelector)
    .each(function () {
      let legend = d3.select(this);
      let baseTransform = getDenseDetailBaseTransform(legend);
      let baseWidth = +legend.attr('data-dense-detail-base-width') ||
        +legend.attr('data-base-width') ||
        +legend.select('.classifier-dense-activation-legend-bar').attr('width');
      let lastLayerIndex = layerLayouts.length - 1;
      let startShift = upstreamShift;
      let endShift = lastLayerIndex <= layerIndex ? upstreamShift : 0;
      let shiftedWidth = Math.max(nodeSize, baseWidth + endShift - startShift);

      legend.attr('data-dense-detail-base-width', baseWidth);
      setDenseActivationLegendWidth(legend, shiftedWidth, duration);
      transitionLegendTransform(
        legend,
        `${baseTransform} translate(${startShift}, 0)`,
        duration,
      );
    });

  return svg.selectAll(`${flattenLegendSelector}, ${denseLegendSelector}`);
}

const restoreDenseDetailLegends = (legends, duration = 0) => {
  legends.each(function () {
    let legend = d3.select(this);
    let baseTransform = legend.attr('data-dense-detail-base-transform');
    let baseWidth = +legend.attr('data-dense-detail-base-width');

    if (baseTransform !== null) {
      transitionLegendTransform(legend, baseTransform, duration);
      legend.attr('data-dense-detail-base-transform', null);
    }

    if (legend.classed('classifier-dense-activation-legend') &&
      Number.isFinite(baseWidth)) {
      setDenseActivationLegendWidth(legend, baseWidth, duration);
      legend.attr('data-dense-detail-base-width', null);
    }
  });
}

export const resetDenseNeuronDetailLayout = () => {
  if (!svg) {
    return;
  }

  let linkGen = d3.linkHorizontal()
    .x(d => d.x)
    .y(d => d.y);

  svg.selectAll('.classifier-dense-detail, .classifier-dense-detail-bridges')
    .interrupt('softmax')
    .remove();

  svg.selectAll('.classifier-dense-overview-layer-clone')
    .interrupt('softmax')
    .remove();

  svg.selectAll('g.cnn-layer-group')
    .interrupt('softmax')
    .attr('transform', null)
    .attr('data-dense-detail-shifted', null);

  svg.selectAll('[data-dense-detail-base-transform]')
    .interrupt('softmax')
    .each(restoreDenseDetailTransform);

  restoreDenseDetailLegends(
    svg.selectAll('.classifier-dense-activation-legend[data-dense-detail-base-width]'),
    0,
  );

  svg.selectAll('[data-dense-detail-base-opacity]')
    .interrupt('softmax')
    .each(restoreDenseDetailOpacity);

  svg.select('g.edge-group')
    .selectAll('path.edge')
    .interrupt('softmax')
    .attr('d', d => d ? linkGen({ source: d.source, target: d.target }) : null);

  updateStyleTestDenseLayerHierarchy(null);
  denseDetailState = undefined;
  denseDetailSelections = [];
  denseDetailViewStore.set({ show: false });
}

const getScalarNodeImageHref = (color) => {
  if (scalarNodeImageCache.has(color)) {
    return scalarNodeImageCache.get(color);
  }

  let canvas = document.createElement('canvas');
  canvas.width = nodeLength * 2;
  canvas.height = nodeLength * 2;
  let context = canvas.getContext('2d');
  context.fillStyle = color;
  context.fillRect(0, 0, canvas.width, canvas.height);
  let href = canvas.toDataURL();
  scalarNodeImageCache.set(color, href);
  return href;
}

const getClassifierHead = () => {
  return {
    flatten: cnn.classifierHead?.flattenView || cnn.flatten,
    flattenRaw: cnn.classifierHead?.flatten || cnn.flatten,
    denseLayers: cnn.classifierHead?.denseLayers || [],
    output: cnn.classifierHead?.output || cnn[cnn.length - 1],
  };
}

const getLogitsForOutputLayer = (flattenLayerData, outputLayerData) => {
  return outputLayerData.map((outputNode, outputI) => {
    if (Number.isFinite(outputNode.logit)) {
      return outputNode.logit;
    }

    return flattenLayerData.reduce((sum, flattenNode) => {
      let weight = flattenNode.outputLinks?.[outputI]?.weight ?? 0;
      return sum + flattenNode.output * weight;
    }, outputNode.bias ?? 0);
  });
}

const getSafeLogitExtent = (logitValues) => {
  let extent = d3.extent(logitValues);
  if (!Number.isFinite(extent[0]) || !Number.isFinite(extent[1])) {
    return [-1, 1];
  }
  if (extent[0] === extent[1]) {
    return [extent[0] - 1, extent[1] + 1];
  }
  return extent;
}

const removeDenseNeuronDetail = (animate = false, options = {}) => {
  let keepSelections = options.keepSelections || false;
  let detail = svg.selectAll('.classifier-dense-detail');
  let bridgeDetails = svg.selectAll('.classifier-dense-detail-bridges');
  let duration = animate ? 320 : 0;
  const restoreTransition = (selection) => {
    selection.interrupt('softmax');
    if (!animate) {
      return selection;
    }
    return selection.transition('softmax')
      .duration(duration)
      .ease(d3.easeCubicInOut);
  };

  if (denseDetailState) {
    svg.selectAll('.classifier-hidden-node-border')
      .style('opacity', 0)
      .style('stroke-width', 1)
      .style('stroke', '#8F8F8F')
      .style('filter', null);

    restoreTransition(denseDetailState.shiftedPrevFeatureLayer)
      .attr('transform', null)
      .attr('data-dense-detail-shifted', null);

    restoreTransition(denseDetailState.shiftedPrevFeatureLabels)
      .attr('transform', function () {
        return d3.select(this).attr('data-dense-detail-base-transform') || '';
      })
      .attr('data-dense-detail-base-transform', null);

    restoreTransition(denseDetailState.shiftedFlattenLayer)
      .attr('transform', 'translate(0, 0)');

    restoreTransition(denseDetailState.fixedOperationEdges)
      .attr('transform', function () {
        return d3.select(this).attr('data-dense-detail-base-transform') || '';
      })
      .attr('data-dense-detail-base-transform', null);

    restoreTransition(denseDetailState.shiftedFlattenLabels)
      .attr('transform', function () {
        return d3.select(this).attr('data-dense-detail-base-transform') || '';
      })
      .attr('data-dense-detail-base-transform', null);

    restoreTransition(denseDetailState.shiftedFlattenAnnotations)
      .attr('transform', function () {
        return d3.select(this).attr('data-dense-detail-base-transform') || '';
      })
      .attr('data-dense-detail-base-transform', null);

    if (denseDetailState.shiftedLegends) {
      restoreDenseDetailLegends(denseDetailState.shiftedLegends, duration);
    }

    restoreTransition(denseDetailState.shiftedLayers)
      .attr('transform', 'translate(0, 0)');

    restoreTransition(denseDetailState.shiftedEdges)
      .attr('d', denseDetailState.getEdgePath(0))
      .style('opacity', function () {
        return d3.select(this).attr('data-original-opacity') || 0.28;
      });

    restoreTransition(denseDetailState.shiftedOutputEdges)
      .attr('d', denseDetailState.getOutputEdgePath(0))
      .style('opacity', function () {
        return d3.select(this).attr('data-original-opacity') || 0.28;
      });

    restoreTransition(denseDetailState.subduedPreviousEdges)
      .style('opacity', function () {
        return d3.select(this).attr('data-original-opacity') || 0.28;
      })
      .style('stroke-width', edgeStrokeWidth)
      .style('stroke', edgeInitColor)
      .style('stroke-dasharray', null)
      .style('filter', null);

    restoreTransition(denseDetailState.subduedNextEdges)
      .style('opacity', function () {
        return d3.select(this).attr('data-original-opacity') || 0.28;
      })
      .style('stroke-width', edgeStrokeWidth)
      .style('stroke', edgeInitColor)
      .style('stroke-dasharray', null)
      .style('filter', null);

    restoreTransition(denseDetailState.subduedDenseOutputEdges)
      .style('opacity', function () {
        return d3.select(this).attr('data-original-opacity') || 0.28;
      })
      .style('stroke-width', edgeStrokeWidth)
      .style('stroke', edgeInitColor);

    restoreTransition(denseDetailState.shiftedLogitDenseEdges)
      .attr('d', denseDetailState.getLogitDenseEdgePath(0));

    restoreTransition(denseDetailState.shiftedOverviewEdges)
      .attr('d', denseDetailState.getOverviewEdgePath(0));

    restoreTransition(denseDetailState.shiftedOverviewLayerClones)
      .style('opacity', 0)
      .remove();

    restoreTransition(denseDetailState.subduedPrevFeatureElements)
      .style('opacity', function () {
        return d3.select(this).attr('data-dense-detail-base-opacity');
      })
      .attr('data-dense-detail-base-opacity', null);

    restoreTransition(bridgeDetails)
      .style('opacity', 0)
      .remove();

    denseDetailState = undefined;
    if (!keepSelections) {
      denseDetailSelections = [];
      denseDetailViewStore.set({ show: false });
    }
  }

  if (detail.empty()) {
    if (!bridgeDetails.empty()) {
      bridgeDetails.remove();
    }
    if (!keepSelections) {
      denseDetailViewStore.set({ show: false });
    }
    return;
  }

  if (animate) {
    detail
      .transition('softmax')
      .duration(320)
      .ease(d3.easeCubicInOut)
      .style('opacity', 0)
      .remove();
    bridgeDetails
      .transition('softmax')
      .duration(320)
      .ease(d3.easeCubicInOut)
      .style('opacity', 0)
      .remove();
  } else {
    detail.remove();
    bridgeDetails.remove();
  }

  if (!keepSelections) {
    denseDetailViewStore.set({ show: false });
  }
}

export const closeDenseNeuronDetail = (animate = true) => {
  let hasDomDetail = svg
    ? !svg.selectAll('.classifier-dense-detail, .classifier-dense-detail-bridges').empty()
    : false;
  let hasDenseDetail = denseDetailState ||
    denseDetailSelections.length ||
    hasDomDetail;

  if (!hasDenseDetail) {
    return false;
  }

  denseDetailSelections = [];
  removeDenseNeuronDetail(animate);
  return true;
}

const getDensePreActivation = (node) => {
  if (Number.isFinite(node.logit)) {
    return node.logit;
  }

  return node.inputLinks.reduce((sum, link) =>
    sum + (link.source.output || 0) * (link.weight || 0),
    node.bias || 0
  );
}

const getDenseDetailViewData = (node, layerIndex, visibleTermCount = 8) => {
  let allTerms = node.inputLinks
    .map((link) => ({
      input: link.source.output || 0,
      weight: link.weight || 0,
      sourceIndex: link.source.index,
      contribution: (link.source.output || 0) * (link.weight || 0),
    }))
    .sort((a, b) =>
      Math.abs(b.contribution) - Math.abs(a.contribution) ||
      a.sourceIndex - b.sourceIndex);
  let visibleTerms = allTerms.slice(0, visibleTermCount);
  let totalContribution = allTerms.reduce((sum, term) =>
    sum + term.contribution, 0);
  let visibleContribution = visibleTerms.reduce((sum, term) =>
    sum + term.contribution, 0);
  let restContribution = totalContribution - visibleContribution;
  let restCount = allTerms.length - visibleTerms.length;
  let restTerm = restCount > 0
    ? {
      isRest: true,
      count: restCount,
      input: 1,
      weight: restContribution,
      sourceIndex: 'rest',
      contribution: restContribution,
    }
    : null;
  let preActivation = getDensePreActivation(node);

  return {
    show: true,
    layerName: node.layerName,
    layerIndex,
    nodeIndex: node.index,
    terms: visibleTerms,
    restTerm,
    bias: node.bias || 0,
    preActivation,
    output: node.output || 0,
    isActive: (node.output || 0) > 0,
  };
}

const getEdgeNumericAttr = (edge, attrName) => {
  let rawValue = edge.attr(attrName);
  return rawValue === null || rawValue === '' ? NaN : +rawValue;
}

const getTranslateCoords = (selection) => {
  let transform = selection.attr('transform') || '';
  let match = transform.match(/translate\(\s*([-\d.]+)(?:[,\s]+([-\d.]+))?/);
  return {
    x: match ? +match[1] : 0,
    y: match && match[2] !== undefined ? +match[2] : 0,
  };
}

const getBiasPlusInputPoint = (svg, targetGroup, fallbackX, fallbackY) => {
  let plusSymbol = svg.select('.flatten-layer .plus-symbol');
  if (plusSymbol.empty()) {
    return { x: fallbackX, y: fallbackY };
  }

  let plusNode = plusSymbol.node();
  let targetNode = targetGroup?.node ? targetGroup.node() : null;
  let ownerSvg = plusNode?.ownerSVGElement;
  if (ownerSvg && plusNode?.getScreenCTM && targetNode?.getScreenCTM) {
    let plusMatrix = plusNode.getScreenCTM();
    let targetMatrix = targetNode.getScreenCTM();
    if (plusMatrix && targetMatrix) {
      let point = ownerSvg.createSVGPoint();
      point.x = -plusSymbolRadius;
      point.y = 0;
      let localPoint = point
        .matrixTransform(plusMatrix)
        .matrixTransform(targetMatrix.inverse());
      return {
        x: Number.isFinite(localPoint.x) ? localPoint.x : fallbackX,
        y: Number.isFinite(localPoint.y) ? localPoint.y : fallbackY,
      };
    }
  }

  let { x, y } = getTranslateCoords(plusSymbol);
  return {
    x: Number.isFinite(x) ? x - plusSymbolRadius : fallbackX,
    y: Number.isFinite(y) ? y : fallbackY,
  };
}

const getPathEndPoint = (pathSelection, targetGroup, fallbackX, fallbackY) => {
  let pathNode = pathSelection.node();
  let targetNode = targetGroup?.node ? targetGroup.node() : null;
  let ownerSvg = pathNode?.ownerSVGElement;

  if (!pathNode || !targetNode || !ownerSvg ||
    typeof pathNode.getTotalLength !== 'function' ||
    typeof pathNode.getPointAtLength !== 'function' ||
    !pathNode.getScreenCTM || !targetNode.getScreenCTM) {
    return { x: fallbackX, y: fallbackY };
  }

  let pathMatrix = pathNode.getScreenCTM();
  let targetMatrix = targetNode.getScreenCTM();
  if (!pathMatrix || !targetMatrix) {
    return { x: fallbackX, y: fallbackY };
  }

  let endPoint = pathNode.getPointAtLength(pathNode.getTotalLength());
  let point = ownerSvg.createSVGPoint();
  point.x = endPoint.x;
  point.y = endPoint.y;
  let localPoint = point
    .matrixTransform(pathMatrix)
    .matrixTransform(targetMatrix.inverse());

  return {
    x: Number.isFinite(localPoint.x) ? localPoint.x : fallbackX,
    y: Number.isFinite(localPoint.y) ? localPoint.y : fallbackY,
  };
}

const animateDenseSignalPath = (path, delay = 0, duration = 620, options = {}) => {
  let node = path.node();
  if (!node || typeof node.getTotalLength !== 'function') {
    return path;
  }

  let length = node.getTotalLength();
  let repeat = options.repeat || false;
  let repeatDelay = options.repeatDelay ?? 720;
  let transitionName = options.name || 'dense-signal';

  const run = (currentDelay) => {
    if (!node.isConnected) {
      return;
    }

    path
      .interrupt(transitionName)
      .attr('stroke-dasharray', `${length} ${length}`)
      .attr('stroke-dashoffset', length)
      .transition(transitionName)
      .delay(currentDelay)
      .duration(duration)
      .ease(d3.easeCubicInOut)
      .attr('stroke-dashoffset', 0)
      .on('end', function () {
        if (!repeat || !node.isConnected) {
          return;
        }
        d3.select(this)
          .transition(transitionName)
          .duration(repeatDelay)
          .attr('stroke-dashoffset', -length * 0.16)
          .on('end', () => run(0));
      });
  };

  run(delay);
  return path;
}

const animateDenseSignalParticle = (group, path, options = {}) => {
  let pathNode = path.node();
  if (!pathNode || typeof pathNode.getTotalLength !== 'function' ||
    typeof pathNode.getPointAtLength !== 'function') {
    return undefined;
  }

  let length = pathNode.getTotalLength();
  let radius = options.radius ?? 2;
  let opacity = options.opacity ?? 0.75;
  let color = options.color ?? '#8ACDF3';
  let duration = options.duration ?? 850;
  let delay = options.delay ?? 0;
  let repeatDelay = options.repeatDelay ?? 520;
  let transitionName = options.name || 'dense-value-particle';
  let startPoint = pathNode.getPointAtLength(0);
  let particle = group.append('circle')
    .attr('class', 'classifier-dense-value-particle')
    .attr('cx', startPoint.x)
    .attr('cy', startPoint.y)
    .attr('r', radius * 0.72)
    .style('fill', color)
    .style('stroke', '#FFFFFF')
    .style('stroke-width', 0.45)
    .style('opacity', 0)
    .style('pointer-events', 'none');

  if (options.title) {
    particle.append('title').text(options.title);
  }

  const run = (currentDelay) => {
    if (!pathNode.isConnected || !particle.node()?.isConnected) {
      return;
    }

    particle
      .interrupt(transitionName)
      .attr('r', radius * 0.72)
      .style('opacity', 0)
      .transition(transitionName)
      .delay(currentDelay)
      .duration(duration)
      .ease(d3.easeCubicInOut)
      .style('opacity', opacity)
      .attr('r', radius)
      .tween('dense-value-position', () => (t) => {
        if (!pathNode.isConnected) {
          return;
        }
        let point = pathNode.getPointAtLength(length * t);
        particle
          .attr('cx', point.x)
          .attr('cy', point.y);
      })
      .transition(`${transitionName}-fade`)
      .duration(260)
      .style('opacity', 0)
      .attr('r', radius * 0.58)
      .transition(`${transitionName}-rest`)
      .duration(repeatDelay)
      .on('end', () => run(0));
  };

  run(delay);
  return particle;
}

const animateDenseReluOutputDot = (dot, startY, endY) => {
  let node = dot.node();
  const run = (delay = 0) => {
    if (!node?.isConnected) {
      return;
    }

    dot
      .interrupt('dense-relu-signal')
      .attr('cy', startY)
      .attr('r', 2.4)
      .transition('dense-relu-signal')
      .delay(delay)
      .duration(850)
      .ease(d3.easeCubicInOut)
      .attr('cy', endY)
      .attr('r', 4.1)
      .transition('dense-relu-signal-rest')
      .duration(720)
      .attr('r', 2.8)
      .on('end', () => run(0));
  };

  run(620);
}

const animateDenseReluPulse = (pulse, targetRadius) => {
  let node = pulse.node();
  const run = (delay = 0) => {
    if (!node?.isConnected) {
      return;
    }

    pulse
      .interrupt('dense-relu-pulse')
      .attr('r', 1.5)
      .style('opacity', 0)
      .transition('dense-relu-pulse')
      .delay(delay)
      .duration(750)
      .ease(d3.easeCubicOut)
      .attr('r', targetRadius)
      .style('opacity', 0.62)
      .transition('dense-relu-pulse-fade')
      .duration(400)
      .style('opacity', 0)
      .transition('dense-relu-pulse-rest')
      .duration(520)
      .on('end', () => run(0));
  };

  run(1350);
}

const formatDenseValue = (value) => (
  Number.isFinite(value) ? formater(value) : 'n/a'
);

const drawDenseArrowHead = (group, x, y, color, opacity = 0.75, scale = 1) => {
  group.append('path')
    .attr('d', `M${x},${y} l${-5 * scale},${-3.2 * scale} v${6.4 * scale} Z`)
    .style('fill', color)
    .style('opacity', opacity)
    .style('pointer-events', 'none');
}

const drawDensePrincipleAnimation = (
  group,
  terms,
  bias,
  preActivation,
  output,
  activationStyle,
  width,
  height,
) => {
  let isActive = output > 0;
  let cy = height / 2;
  let leftCx = 43;
  let rightCx = width - 42;
  let radius = Math.min(29, height / 2 - 8);
  let gateX = width / 2;
  let linkGen = d3.linkHorizontal()
    .x(d => d.x)
    .y(d => d.y);
  let allTerms = terms
    .map((term) => ({
      ...term,
      contribution: term.input * term.weight,
    }))
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  let visibleTerms = allTerms.slice(0, 5);
  let visibleContribution = visibleTerms.reduce((sum, term) =>
    sum + term.contribution, 0);
  let totalContribution = allTerms.reduce((sum, term) =>
    sum + term.contribution, 0);
  let remainderContribution = totalContribution - visibleContribution;
  let showRemainder = allTerms.length > visibleTerms.length &&
    Math.abs(remainderContribution) > 1e-9;
  let topTerms = showRemainder
    ? [
      ...visibleTerms,
      {
        input: 1,
        weight: remainderContribution,
        contribution: remainderContribution,
        sourceIndex: 'rest',
        isRemainder: true,
        count: allTerms.length - visibleTerms.length,
      },
    ]
    : visibleTerms;
  let contributionRange = Math.max(
    0.001,
    ...topTerms.map((term) => Math.abs(term.contribution)),
    Math.abs(bias),
    Math.abs(preActivation),
  );
  let termRange = Math.max(0.001,
    ...topTerms.map((term) => Math.abs(term.contribution)));
  let sumColor = preActivation >= 0 ? '#82BFD9' : '#D9AF8D';
  let blockedColor = '#AEB8C2';
  let activationColor = isActive ? activationStyle.stroke : blockedColor;

  group.append('title')
    .text(`weighted sum + bias = ${formatDenseValue(preActivation)}
ReLU output = ${formatDenseValue(output)}
state = ${isActive ? 'activated' : 'not activated'}`);

  group.append('rect')
    .attr('x', -3)
    .attr('y', -3)
    .attr('width', width + 6)
    .attr('height', height + 6)
    .attr('rx', 15)
    .attr('ry', 15)
    .style('fill', '#FAFDFF')
    .style('stroke', '#DCEEF8')
    .style('stroke-width', 0.72)
    .style('opacity', 0.82);

  let leftFill = group.append('path')
    .attr('class', 'classifier-dense-principle-sum-semicircle')
    .attr('d', `M${leftCx},${cy - radius}
      A${radius},${radius} 0 0 0 ${leftCx},${cy + radius}
      L${leftCx},${cy - radius} Z`)
    .style('fill', preActivation >= 0 ? '#EEF9FD' : '#FCF3ED')
    .style('stroke', sumColor)
    .style('stroke-width', 1.35)
    .style('opacity', 0.92)
    .style('filter', 'drop-shadow(0 0 5px rgba(126, 190, 232, 0.18))');
  leftFill.append('title')
    .text(`sum(input * weight) + bias = ${formatDenseValue(preActivation)}`);

  let rightFill = group.append('path')
    .attr('class', 'classifier-dense-principle-output-semicircle')
    .attr('d', `M${rightCx},${cy - radius}
      A${radius},${radius} 0 0 1 ${rightCx},${cy + radius}
      L${rightCx},${cy - radius} Z`)
    .style('fill', activationStyle.fill)
    .style('stroke', activationStyle.stroke)
    .style('stroke-width', isActive ? 1.55 : 1.15)
    .style('opacity', isActive ? 0.95 : 0.66)
    .style('filter', activationStyle.glow);
  rightFill.append('title')
    .text(isActive
      ? `activated: ReLU output ${formatDenseValue(output)}`
      : `not activated: ReLU output ${formatDenseValue(output)}`);

  let termGap = topTerms.length > 1 ? 6 : 0;
  let startY = cy - ((topTerms.length - 1) * termGap) / 2;
  topTerms.forEach((term, index) => {
    let termY = startY + index * termGap;
    let contributionRatio = Math.abs(term.contribution) / termRange;
    let color = gappedColorScale(
      layerColorScales.weight,
      termRange * 2,
      term.contribution,
      0.12,
    );
    let termPath = group.append('path')
      .attr('class', 'classifier-dense-principle-term')
      .attr('d', linkGen({
        source: { x: 8, y: termY },
        target: { x: leftCx - 1, y: cy },
      }))
      .style('fill', 'none')
      .style('stroke', color)
      .style('stroke-width', 0.58 + Math.min(1.45, contributionRatio * 2.2))
      .style('stroke-linecap', 'round')
      .style('stroke-dasharray', term.isRemainder ? '1.4 2.3' : null)
      .style('opacity', (term.isRemainder ? 0.28 : 0.36) +
        Math.min(0.38, contributionRatio * 0.38))
      .style('pointer-events', 'stroke');
    termPath.append('title')
      .text(term.isRemainder
        ? `${term.count} smaller terms: ${formatDenseValue(term.contribution)}`
        : `input * weight: ${formatDenseValue(term.contribution)}`);
    animateDenseSignalPath(termPath, 90 + index * 60, 620, {
      repeat: true,
      repeatDelay: 850,
      name: `dense-principle-term-${index}`,
    });
    animateDenseSignalParticle(group, termPath, {
      delay: 130 + index * 75,
      duration: 880 - Math.min(260, contributionRatio * 260),
      repeatDelay: 680,
      radius: 1.15 + Math.min(2.2, contributionRatio * 2.2),
      opacity: 0.38 + Math.min(0.42, contributionRatio * 0.42),
      color,
      name: `dense-principle-term-particle-${index}`,
      title: term.isRemainder
        ? `${term.count} smaller terms`
        : `input * weight: ${formatDenseValue(term.contribution)}`,
    });

    group.append('circle')
      .attr('cx', 6)
      .attr('cy', termY)
      .attr('r', term.isRemainder ? 1.35 : 1.8)
      .style('fill', term.isRemainder ? '#F5F8FB' : '#EAF6FD')
      .style('stroke', term.isRemainder ? '#CBD7E0' : '#9DCCE9')
      .style('stroke-width', 0.6)
      .style('opacity', term.isRemainder ? 0.65 : 0.92);
  });

  let biasPath = group.append('path')
    .attr('class', 'classifier-dense-principle-bias')
    .attr('d', linkGen({
      source: { x: leftCx - 16, y: cy - radius - 5 },
      target: { x: leftCx - 2, y: cy - 8 },
    }))
    .style('fill', 'none')
    .style('stroke', '#BFD7EC')
    .style('stroke-width', 0.75)
    .style('opacity', 0.72);
  biasPath.append('title').text(`bias: ${formatDenseValue(bias)}`);
  animateDenseSignalPath(biasPath, 260, 440, {
    repeat: true,
    repeatDelay: 940,
    name: 'dense-principle-bias',
  });

  group.append('circle')
    .attr('cx', leftCx - 17)
    .attr('cy', cy - radius - 6)
    .attr('r', 2.5)
    .style('fill', gappedColorScale(layerColorScales.weight,
      contributionRange * 2, bias, 0.12))
    .style('stroke', '#B8DDF4')
    .style('stroke-width', 0.65)
    .style('opacity', 0.82)
    .append('title')
    .text(`bias: ${formatDenseValue(bias)}`);

  group.append('text')
    .attr('x', leftCx - radius / 2)
    .attr('y', cy + 1)
    .style('text-anchor', 'middle')
    .style('dominant-baseline', 'middle')
    .style('font-size', '17px')
    .style('font-weight', 700)
    .style('fill', '#738A9C')
    .style('opacity', 0.9)
    .text('\u03A3');

  let sumOutPath = group.append('path')
    .attr('class', 'classifier-dense-principle-sum-out')
    .attr('d', `M${leftCx + 2},${cy} C${leftCx + 22},${cy} ${gateX - 28},${cy} ${gateX - 14},${cy}`)
    .style('fill', 'none')
    .style('stroke', sumColor)
    .style('stroke-width', 1.25)
    .style('stroke-linecap', 'round')
    .style('opacity', 0.82);
  sumOutPath.append('title')
    .text(`pre-activation: ${formatDenseValue(preActivation)}`);
  animateDenseSignalPath(sumOutPath, 560, 560, {
    repeat: true,
    repeatDelay: 760,
    name: 'dense-principle-sum-out',
  });
  animateDenseSignalParticle(group, sumOutPath, {
    delay: 610,
    duration: 720,
    repeatDelay: 720,
    radius: 2.1,
    opacity: 0.72,
    color: sumColor,
    name: 'dense-principle-sum-particle',
    title: `pre-activation: ${formatDenseValue(preActivation)}`,
  });
  drawDenseArrowHead(group, gateX - 11, cy, sumColor, 0.78, 0.82);

  group.append('circle')
    .attr('cx', gateX)
    .attr('cy', cy)
    .attr('r', 9.4)
    .style('fill', '#FFFFFF')
    .style('stroke', isActive ? '#8B75D7' : '#B7C7D9')
    .style('stroke-width', 0.9)
    .style('opacity', 0.94)
    .append('title')
    .text(`ReLU(${formatDenseValue(preActivation)}) = ${formatDenseValue(output)}`);

  group.append('path')
    .attr('class', 'classifier-dense-principle-relu-symbol')
    .attr('d', isActive
      ? `M${gateX - 4.8},${cy + 4} L${gateX},${cy + 4} L${gateX + 5.6},${cy - 5.2}`
      : `M${gateX - 5.5},${cy + 4} L${gateX + 5.5},${cy + 4}`)
    .style('fill', 'none')
    .style('stroke', isActive ? '#7A5AC8' : '#AEB8C2')
    .style('stroke-width', 1.4)
    .style('stroke-linecap', 'round')
    .style('stroke-linejoin', 'round')
    .style('opacity', 0.9);

  if (!isActive) {
    group.append('line')
      .attr('x1', gateX + 8)
      .attr('x2', gateX + 8)
      .attr('y1', cy - 7)
      .attr('y2', cy + 7)
      .style('stroke', '#AEB8C2')
      .style('stroke-width', 1)
      .style('opacity', 0.7);
  }

  let activationPath = group.append('path')
    .attr('class', 'classifier-dense-principle-activation-out')
    .attr('d', `M${gateX + 12},${cy} C${gateX + 31},${cy} ${rightCx - 24},${cy} ${rightCx - 2},${cy}`)
    .style('fill', 'none')
    .style('stroke', activationColor)
    .style('stroke-width', isActive ? 1.45 : 0.92)
    .style('stroke-dasharray', isActive ? null : '1.5 3')
    .style('stroke-linecap', 'round')
    .style('opacity', isActive ? 0.88 : 0.34);
  activationPath.append('title')
    .text(isActive
      ? `activation passed: ${formatDenseValue(output)}`
      : `not activated: output ${formatDenseValue(output)}`);
  animateDenseSignalPath(activationPath, 980, 680, {
    repeat: true,
    repeatDelay: 830,
    name: 'dense-principle-activation-out',
  });
  if (isActive) {
    animateDenseSignalParticle(group, activationPath, {
      delay: 1030,
      duration: 760,
      repeatDelay: 720,
      radius: 2.25,
      opacity: 0.76,
      color: activationStyle.stroke,
      name: 'dense-principle-activation-particle',
      title: `activation: ${formatDenseValue(output)}`,
    });
  }
  drawDenseArrowHead(group, rightCx - 2, cy, activationColor,
    isActive ? 0.82 : 0.32, 0.82);

  group.append('circle')
    .attr('class', 'classifier-dense-principle-output-dot')
    .attr('cx', rightCx + radius / 2)
    .attr('cy', cy)
    .attr('r', isActive ? 4.2 : 3.4)
    .style('fill', activationStyle.fill)
    .style('stroke', activationStyle.stroke)
    .style('stroke-width', 0.9)
    .style('opacity', isActive ? 0.95 : 0.64)
    .append('title')
    .text(isActive
      ? `activated neuron output: ${formatDenseValue(output)}`
      : `not activated neuron output: ${formatDenseValue(output)}`);
}

const drawDenseWeightedSum = (group, terms, bias, preActivation, width, height) => {
  let allTerms = terms
    .map((term) => ({
      ...term,
      contribution: term.input * term.weight,
    }))
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  let visibleTerms = allTerms.slice(0, 6);
  let visibleContribution = visibleTerms.reduce((sum, term) =>
    sum + term.contribution, 0);
  let totalContribution = allTerms.reduce((sum, term) =>
    sum + term.contribution, 0);
  let remainderContribution = totalContribution - visibleContribution;
  let shouldShowRemainder = allTerms.length > visibleTerms.length &&
    Math.abs(remainderContribution) > 1e-9;
  let topTerms = shouldShowRemainder
    ? [
      ...visibleTerms,
      {
        input: 1,
        weight: remainderContribution,
        contribution: remainderContribution,
        sourceIndex: 'rest',
        isRemainder: true,
        count: allTerms.length - visibleTerms.length,
      },
    ]
    : visibleTerms;
  let contributionRange = Math.max(
    0.001,
    ...topTerms.map((term) => Math.abs(term.contribution)),
    Math.abs(bias),
    Math.abs(preActivation),
  );
  let termContributionRange = Math.max(
    0.001,
    ...topTerms.map((term) => Math.abs(term.contribution)),
  );
  let inputX = 8;
  let sumX = width - 18;
  let centerY = height / 2;
  let termGap = topTerms.length > 1 ? 6 : 0;
  let startY = centerY - ((topTerms.length - 1) * termGap) / 2;
  let linkGen = d3.linkHorizontal()
    .x(d => d.x)
    .y(d => d.y);

  group.append('title')
    .text(`weighted sum: sum(input * weight) + bias = ${formatDenseValue(preActivation)}
sum(input * weight): ${formatDenseValue(totalContribution)}
bias: ${formatDenseValue(bias)}`);

  group.append('rect')
    .attr('x', -2)
    .attr('y', 1)
    .attr('width', width + 4)
    .attr('height', height - 2)
    .attr('rx', 12)
    .attr('ry', 12)
    .style('fill', '#F5FBFF')
    .style('stroke', '#DAEEF9')
    .style('stroke-width', 0.7)
    .style('opacity', 0.76);

  topTerms.forEach((term, index) => {
    let termY = startY + index * termGap;
    let contributionRatio = Math.abs(term.contribution) / termContributionRange;
    let signedColorRange = Math.max(0.001, termContributionRange * 2);
    let contributionOpacity = term.isRemainder
      ? 0.22 + Math.min(0.42, contributionRatio * 0.42)
      : 0.28 + Math.min(0.56, contributionRatio * 0.56);
    let color = gappedColorScale(
      layerColorScales.weight,
      signedColorRange,
      term.contribution,
      0.12,
    );

    let path = group.append('path')
      .attr('d', linkGen({
        source: { x: inputX + 8, y: termY },
        target: { x: sumX - 13, y: centerY },
      }))
      .style('fill', 'none')
      .style('stroke', color)
      .style('stroke-width', 0.62 + Math.min(1.75, contributionRatio * 3))
      .style('stroke-linecap', 'round')
      .style('stroke-dasharray', term.isRemainder ? '1.4 2.2' : null)
      .style('opacity', contributionOpacity)
      .style('pointer-events', 'stroke');
    path.append('title')
      .text(term.isRemainder
        ? `${term.count} smaller contributions: ${formatDenseValue(term.contribution)}`
        : `input: ${formatDenseValue(term.input)}
weight: ${formatDenseValue(term.weight)}
input * weight: ${formatDenseValue(term.contribution)}`);
    animateDenseSignalPath(path, 90 + index * 55, 560, {
      repeat: true,
      repeatDelay: 820,
      name: `dense-sum-term-${index}`,
    });
    animateDenseSignalParticle(group, path, {
      delay: 130 + index * 70,
      duration: 940 - Math.min(320, contributionRatio * 320),
      repeatDelay: 620 + index * 45,
      radius: (term.isRemainder ? 1.2 : 1.45) + Math.min(2.35, contributionRatio * 2.35),
      opacity: (term.isRemainder ? 0.34 : 0.46) + Math.min(0.46, contributionRatio * 0.46),
      color,
      name: `dense-sum-term-particle-${index}`,
      title: term.isRemainder
        ? `${term.count} smaller contributions: ${formatDenseValue(term.contribution)}`
        : `input * weight: ${formatDenseValue(term.contribution)}`,
    });

    group.append('circle')
      .attr('cx', inputX)
      .attr('cy', termY)
      .attr('r', term.isRemainder ? 1.45 : 1.9)
      .style('fill', term.isRemainder ? '#F4F8FB' : '#EAF6FD')
      .style('stroke', term.isRemainder ? '#C9D9E5' : '#9DCCE9')
      .style('stroke-width', 0.65)
      .style('opacity', term.isRemainder ? 0.72 : 0.94)
      .append('title')
      .text(term.isRemainder
        ? `${term.count} smaller contributions`
        : `source ${term.sourceIndex}`);

    group.append('rect')
      .attr('x', inputX + 4.7)
      .attr('y', termY - 1.15)
      .attr('width', 5 + Math.min(8, contributionRatio * 8))
      .attr('height', 2.3)
      .attr('rx', 1)
      .style('fill', color)
      .style('opacity', contributionOpacity + 0.1)
      .append('title')
      .text(term.isRemainder
        ? `remaining contribution: ${formatDenseValue(term.contribution)}`
        : `contribution: ${formatDenseValue(term.contribution)}`);
  });

  let biasDot = group.append('circle')
    .attr('cx', sumX)
    .attr('cy', centerY - 15)
    .attr('r', 2.8)
    .style('fill', gappedColorScale(layerColorScales.weight,
      contributionRange * 2, bias, 0.12))
    .style('stroke', '#B8DDF4')
    .style('stroke-width', 0.7)
    .style('opacity', 0.5 + Math.min(0.36, Math.abs(bias) / contributionRange * 0.36));
  biasDot.append('title')
    .text(`bias: ${formatDenseValue(bias)}`);
  let biasRatio = Math.abs(bias) / contributionRange;

  let biasPath = group.append('path')
    .attr('d', linkGen({
      source: { x: sumX, y: centerY - 12 },
      target: { x: sumX, y: centerY - 9 },
    }))
    .style('fill', 'none')
    .style('stroke', '#BFD7EC')
    .style('stroke-width', 0.75)
    .style('opacity', 0.74)
    .style('pointer-events', 'none');
  animateDenseSignalPath(biasPath, 230, 420, {
    repeat: true,
    repeatDelay: 900,
    name: 'dense-sum-bias',
  });
  animateDenseSignalParticle(group, biasPath, {
    delay: 260,
    duration: 560 - Math.min(180, biasRatio * 180),
    repeatDelay: 860,
    radius: 1.05 + Math.min(2.1, biasRatio * 2.1),
    opacity: 0.32 + Math.min(0.52, biasRatio * 0.52),
    color: gappedColorScale(layerColorScales.weight,
      contributionRange * 2, bias, 0.12),
    name: 'dense-sum-bias-particle',
    title: `bias: ${formatDenseValue(bias)}`,
  });

  let sumCircle = group.append('circle')
    .attr('cx', sumX)
    .attr('cy', centerY)
    .attr('r', 10.5)
    .style('fill', '#FAFDFF')
    .style('stroke', '#BFDDF2')
    .style('stroke-width', 0.95)
    .style('filter', 'drop-shadow(0 0 5px rgba(126, 190, 232, 0.24))');
  sumCircle.append('title')
    .text(`full pre-activation: ${formatDenseValue(preActivation)}`);
  let preActivationRatio = Math.abs(preActivation) / contributionRange;

  group.append('text')
    .attr('x', sumX)
    .attr('y', centerY + 1)
    .style('dominant-baseline', 'middle')
    .style('text-anchor', 'middle')
    .style('font-size', '15px')
    .style('font-weight', 700)
    .style('fill', '#7D8FA1')
    .text('\u03A3')
    .append('title')
    .text(`sum(input * weight) + bias = ${formatDenseValue(preActivation)}`);

  let sumSignalColor = preActivation >= 0 ? '#82BFD9' : '#D6AA8A';
  let sumSignal = group.append('path')
    .attr('d', `M${sumX + 10.5},${centerY} L${width + 2},${centerY}`)
    .style('stroke', '#BFD7EC')
    .style('fill', 'none')
    .style('stroke', sumSignalColor)
    .style('stroke-width', 1.2)
    .style('stroke-linecap', 'round')
    .style('opacity', 0.86);
  sumSignal.append('title')
    .text(`pre-activation: ${formatDenseValue(preActivation)}`);
  animateDenseSignalPath(sumSignal, 560, 520, {
    repeat: true,
    repeatDelay: 700,
    name: 'dense-sum-output',
  });
  animateDenseSignalParticle(group, sumSignal, {
    delay: 620,
    duration: 680 - Math.min(220, preActivationRatio * 220),
    repeatDelay: 650,
    radius: 1.35 + Math.min(2.5, preActivationRatio * 2.5),
    opacity: 0.44 + Math.min(0.46, preActivationRatio * 0.46),
    color: sumSignalColor,
    name: 'dense-sum-output-particle',
    title: `pre-activation: ${formatDenseValue(preActivation)}`,
  });

  group.append('circle')
    .attr('cx', width + 2)
    .attr('cy', centerY)
    .attr('r', 3.2)
    .style('fill', preActivation >= 0 ? '#D8F0F8' : '#F4E9E3')
    .style('stroke', preActivation >= 0 ? '#92C7DE' : '#D6AA8A')
    .style('stroke-width', 0.75)
    .style('opacity', 0.95);
}

const drawDenseReluGraph = (group, preActivation, output, width, height) => {
  let margin = { left: 10, right: 11, top: 9, bottom: 10 };
  let graphWidth = width - margin.left - margin.right;
  let graphHeight = height - margin.top - margin.bottom;
  let extent = Math.max(1, Math.abs(preActivation), Math.abs(output || 0));
  let xScale = d3.scaleLinear()
    .domain([-extent, extent])
    .range([margin.left, margin.left + graphWidth]);
  let yScale = d3.scaleLinear()
    .domain([0, extent])
    .range([margin.top + graphHeight, margin.top]);
  let zeroX = xScale(0);
  let baselineY = yScale(0);
  let reluOutput = Number.isFinite(output) ? output : Math.max(0, preActivation);
  let reluY = yScale(Math.max(0, reluOutput));
  let markerX = Math.max(margin.left, Math.min(margin.left + graphWidth,
    xScale(preActivation)));
  let isPassed = reluOutput > 0;

  group.append('title')
    .text(`ReLU(${formatDenseValue(preActivation)}) = ${formatDenseValue(reluOutput)}`);

  group.append('rect')
    .attr('x', -2)
    .attr('y', 1)
    .attr('width', width + 4)
    .attr('height', height - 2)
    .attr('rx', 12)
    .attr('ry', 12)
    .style('fill', '#F8FBFF')
    .style('stroke', '#E2E9F3')
    .style('stroke-width', 0.7)
    .style('opacity', 0.76);

  group.append('line')
    .attr('x1', margin.left)
    .attr('x2', margin.left + graphWidth)
    .attr('y1', baselineY)
    .attr('y2', baselineY)
    .style('stroke', '#D6DEE8')
    .style('stroke-width', 0.8)
    .style('opacity', 0.75);

  group.append('line')
    .attr('x1', zeroX)
    .attr('x2', zeroX)
    .attr('y1', margin.top)
    .attr('y2', baselineY + 4)
    .style('stroke', '#D6DEE8')
    .style('stroke-width', 0.8)
    .style('opacity', 0.75);

  let reluFlat = group.append('line')
    .attr('x1', margin.left)
    .attr('x2', zeroX)
    .attr('y1', baselineY)
    .attr('y2', baselineY)
    .style('stroke', '#AEB8C2')
    .style('stroke-width', 1.7)
    .style('stroke-linecap', 'round');
  animateDenseSignalPath(reluFlat, 650, 420, {
    repeat: true,
    repeatDelay: 920,
    name: 'dense-relu-flat',
  });

  let reluSlope = group.append('line')
    .attr('x1', zeroX)
    .attr('x2', margin.left + graphWidth)
    .attr('y1', baselineY)
    .attr('y2', margin.top)
    .style('stroke', '#7A5AC8')
    .style('stroke-width', 1.9)
    .style('stroke-linecap', 'round');
  animateDenseSignalPath(reluSlope, 760, 520, {
    repeat: true,
    repeatDelay: 780,
    name: 'dense-relu-slope',
  });

  group.append('path')
    .attr('class', 'classifier-dense-relu-value-line')
    .attr('d', `M${markerX},${baselineY} L${markerX},${reluY}`)
    .style('fill', 'none')
    .style('stroke', isPassed ? '#7A5AC8' : '#AEB8C2')
    .style('stroke-width', 0.95)
    .style('stroke-dasharray', isPassed ? '2 2' : '1 3')
    .style('opacity', isPassed ? 0.75 : 0.42);

  group.append('circle')
    .attr('class', 'classifier-dense-relu-input-dot')
    .attr('cx', markerX)
    .attr('cy', baselineY)
    .attr('r', 3.3)
    .style('fill', '#FFFFFF')
    .style('stroke', isPassed ? '#7A5AC8' : '#AEB8C2')
    .style('stroke-width', 0.9)
    .style('opacity', 0.86);

  let blockedPath = group.append('path')
    .attr('class', 'classifier-dense-relu-blocked-branch')
    .attr('d', `M${margin.left - 3},${baselineY}
       C${margin.left + 10},${baselineY} ${Math.min(markerX, zeroX) - 7},${baselineY} ${Math.min(markerX, zeroX)},${baselineY}
       C${Math.min(markerX, zeroX) + 10},${baselineY} ${width - 22},${baselineY} ${width - 8},${baselineY}`)
    .style('fill', 'none')
    .style('stroke', '#AEB8C2')
    .style('stroke-width', isPassed ? 0.7 : 1.25)
    .style('stroke-linecap', 'round')
    .style('stroke-linejoin', 'round')
    .style('stroke-dasharray', isPassed ? '1.5 2.4' : null)
    .style('opacity', isPassed ? 0.2 : 0.64)
    .style('pointer-events', 'stroke');
  blockedPath.append('title')
    .text(isPassed
      ? 'zero branch shown for comparison'
      : `blocked by ReLU: ${formatDenseValue(preActivation)} becomes 0`);
  animateDenseSignalPath(blockedPath, isPassed ? 1180 : 980, isPassed ? 520 : 680, {
    repeat: true,
    repeatDelay: isPassed ? 940 : 760,
    name: 'dense-relu-blocked',
  });

  let passPath = group.append('path')
    .attr('class', 'classifier-dense-relu-pass-branch')
    .attr('d', isPassed
      ? `M${Math.max(margin.left - 3, markerX - 24)},${baselineY}
         C${markerX - 12},${baselineY} ${markerX - 5},${baselineY} ${markerX},${baselineY}
         L${markerX},${reluY}
         C${markerX + 8},${reluY} ${width - 20},${reluY} ${width - 7},${reluY}`
      : `M${zeroX},${baselineY}
         C${zeroX + 8},${baselineY} ${width - 25},${margin.top + 6} ${width - 7},${margin.top}`)
    .style('fill', 'none')
    .style('stroke', isPassed ? '#7A5AC8' : '#AEB8C2')
    .style('stroke-width', isPassed ? 1.35 : 0.75)
    .style('stroke-linecap', 'round')
    .style('stroke-linejoin', 'round')
    .style('stroke-dasharray', isPassed ? null : '1.5 2.6')
    .style('opacity', isPassed ? 0.7 : 0.22)
    .style('pointer-events', 'stroke');
  passPath.append('title')
    .text(isPassed
      ? `positive value passes through: ${formatDenseValue(reluOutput)}`
      : 'pass-through branch shown for comparison');
  animateDenseSignalPath(passPath, isPassed ? 980 : 1220, isPassed ? 680 : 520, {
    repeat: true,
    repeatDelay: isPassed ? 760 : 940,
    name: 'dense-relu-pass',
  });

  let reluOutputDot = group.append('circle')
    .attr('class', 'classifier-dense-relu-output-dot')
    .attr('cx', markerX)
    .attr('cy', baselineY)
    .attr('r', 2.4)
    .style('fill', isPassed ? '#EEE9FF' : '#F3F6F8')
    .style('stroke', isPassed ? '#7A5AC8' : '#AEB8C2')
    .style('stroke-width', 1)
    .style('opacity', 0.9);
  animateDenseReluOutputDot(reluOutputDot, baselineY, reluY);

  let reluPulse = group.append('circle')
    .attr('class', 'classifier-dense-relu-pulse')
    .attr('cx', markerX)
    .attr('cy', reluY)
    .attr('r', 1.5)
    .style('fill', 'none')
    .style('stroke', isPassed ? '#7A5AC8' : '#AEB8C2')
    .style('stroke-width', 0.8)
    .style('opacity', 0);
  animateDenseReluPulse(reluPulse, isPassed ? 8 : 6);

  group.append('circle')
    .attr('cx', width - 6)
    .attr('cy', isPassed ? reluY : baselineY)
    .attr('r', 3.1)
    .style('fill', isPassed ? '#EEE9FF' : '#F4F6F8')
    .style('stroke', isPassed ? '#7A5AC8' : '#AEB8C2')
    .style('stroke-width', 0.9)
    .style('opacity', isPassed ? 0.92 : 0.54);
}

const getDenseConnectionEdgeStyle = (edgeCount, options = {}) => {
  let emphasized = options.emphasized || false;
  let isLargeFanIn = options.largeFanIn || false;
  let baseOpacity = edgeCount > 512 ? 0.12 :
    edgeCount > 96 ? 0.18 :
      edgeCount > 32 ? 0.2 : 0.42;
  let baseWidth = edgeCount > 512 ? 0.25 :
    edgeCount > 96 ? 0.32 : 0.46;

  if (isLargeFanIn) {
    baseOpacity = edgeCount > 512 ? 0.045 : 0.085;
    baseWidth = edgeCount > 512 ? 0.11 : 0.18;
  }

  return {
    color: '#111111',
    width: emphasized ? Math.max(0.52, baseWidth) : baseWidth,
    opacity: emphasized ? Math.min(0.48, baseOpacity + 0.2) : baseOpacity,
    dasharray: 'none',
    markerOpacity: emphasized ? 0.58 : 0.2,
  };
}

const drawDenseOutgoingTargetMarker = (group, x, y, edgeStyle, title) => {
  let marker = group.append('circle')
    .attr('class', 'classifier-dense-detail-output-marker')
    .attr('cx', x)
    .attr('cy', y)
    .attr('r', 2.2)
    .style('fill', '#FFFFFF')
    .style('stroke', edgeStyle.color)
    .style('stroke-width', 0.8)
    .style('opacity', edgeStyle.markerOpacity)
    .style('pointer-events', 'all');
  marker.append('title').text(title);
  return marker;
}

const drawDenseNeuronDetail = (arg) => {
  let {
    headGroup,
    layerLayouts,
    layerIndex,
    node,
    nodeX,
    nodeY,
    nodeSize,
    prevFeatureLayerIndex,
    flattenAnchorForFlatIndex,
  } = arg;
  let skipExistingRemoval = arg.skipExistingRemoval || false;
  let skipLayoutShift = arg.skipLayoutShift || false;
  const panelWidth = 232;
  const panelHeight = 82;
  const activationGateX = panelWidth / 2;
  const panelGap = 16;
  const layerShift = panelWidth + panelGap * 2;
  const upstreamShift = -layerShift;
  const shiftedNodeX = nodeX + upstreamShift;
  const panelX = shiftedNodeX + nodeSize + panelGap;
  const panelY = nodeY + nodeSize / 2 - panelHeight / 2;
  const duration = 800;
  let nextLayout = layerLayouts[layerIndex + 1];
  let preActivation = getDensePreActivation(node);
  let output = node.output;
  let activationStyle = getDenseActivationStyle(node);
  let linkGen = d3.linkHorizontal()
    .x(d => d.x)
    .y(d => d.y);

  let denseNodeElement = svg.select(`.classifier-hidden-node-${layerIndex}-${node.index}`).node();
  let detailview = document.getElementById('detailview');
  if (denseNodeElement && detailview) {
    let pos = getMidCoords(svg, denseNodeElement);
    let wholeSvg = d3.select('#cnn-svg');
    let svgYMid = +wholeSvg.style('height').replace('px', '') / 2;
    let svgWidth = +wholeSvg.style('width').replace('px', '');
    let detailViewTop = svgYMid - 195 / 2;
    let detailViewLeft = Math.max(24, Math.min(svgWidth - 540, pos.left - 250));
    positionFloatingDetailView(detailview, detailViewTop, detailViewLeft);
  }

  softmaxDetailViewStore.set({
    show: false,
    logits: [],
  });
  denseDetailViewStore.set(getDenseDetailViewData(node, layerIndex));

  if (!skipExistingRemoval) {
    removeDenseNeuronDetail(false, { keepSelections: true });
  }

  if (!skipExistingRemoval) {
    svg.selectAll('.classifier-hidden-node-border')
      .style('opacity', 0)
      .style('stroke-width', 1)
      .style('stroke', '#8F8F8F')
      .style('filter', null);
  }

  svg.selectAll(`.classifier-hidden-node-border-${layerIndex}-${node.index}`)
    .style('opacity', 1)
    .style('stroke-width', 1.5)
    .style('stroke', '#8ACDF3')
    .style('filter', 'drop-shadow(0 0 6px rgba(126, 190, 232, 0.5))');

  let shiftedOverviewLayerIndexes = [prevFeatureLayerIndex - 1, prevFeatureLayerIndex]
    .filter((index) => index >= 0);
  let shiftedOverviewLayerSet = new Set(shiftedOverviewLayerIndexes);
  let shiftedPrevFeatureLayer = svg.selectAll(
    shiftedOverviewLayerIndexes
      .map((index) => `#cnn-layer-group-${index}`)
      .join(', ')
  );
  if (!skipLayoutShift) {
    svg.selectAll('.classifier-dense-overview-layer-clone').remove();
  }
  let shiftedOverviewLayerClones = svg.append('g')
    .attr('class', 'classifier-dense-overview-layer-clone')
    .style('opacity', 1)
    .style('pointer-events', 'none');

  let sourceLayer = svg.select(`#cnn-layer-group-${prevFeatureLayerIndex}`).node();
  if (sourceLayer && !skipLayoutShift) {
    shiftedOverviewLayerClones.node().appendChild(sourceLayer.cloneNode(true));
  }
  shiftedOverviewLayerClones.selectAll('[id]').attr('id', null);
  let shiftedPrevFeatureLabels = svg.selectAll(
    shiftedOverviewLayerIndexes
      .flatMap((index) => [`#layer-label-${index}`, `#layer-detailed-label-${index}`])
      .join(', ')
  );
  let directPrevFeatureLabels = svg.selectAll(
    [`#layer-label-${prevFeatureLayerIndex}`, `#layer-detailed-label-${prevFeatureLayerIndex}`]
      .join(', ')
  );
  let subduedPrevFeatureIndex = prevFeatureLayerIndex - 1;
  let subduedPrevFeatureElements = svg.selectAll(
    subduedPrevFeatureIndex >= 0
      ? [
        `#cnn-layer-group-${subduedPrevFeatureIndex}`,
        `#layer-label-${subduedPrevFeatureIndex}`,
        `#layer-detailed-label-${subduedPrevFeatureIndex}`,
      ].join(', ')
      : '.classifier-dense-no-subdued-layer'
  );
  let shiftedFlattenLayer = svg.selectAll('.flatten-layer');
  let fixedOperationEdges = svg.selectAll(
    '.flatten-layer .symbol-softmax, .flatten-layer .symbol-output, .flatten-layer .symbol-output-line'
  );
  let shiftedFlattenLabels = svg.selectAll(
    '.intermediate-layer > .layer-label, .intermediate-layer > .layer-detailed-label'
  );
  let shiftedFlattenAnnotations = svg.selectAll('.flatten-annotation');
  let shiftedLegends = svg.selectAll(
    `.intermediate-legend-${prevFeatureLayerIndex}.classifier-flatten-activation-legend, ` +
    `.intermediate-legend-${prevFeatureLayerIndex}.classifier-dense-activation-legend`
  );
  let shiftedLayers = headGroup.selectAll('.classifier-hidden-layer')
    .filter((_, index) => index <= layerIndex);
  let shiftedOverviewEdges = svg.select('g.edge-group').selectAll('path.edge')
    .filter((d) => d &&
      (shiftedOverviewLayerSet.has(d.sourceLayerIndex) ||
        shiftedOverviewLayerSet.has(d.targetLayerIndex)));
  let shiftedEdges = headGroup.selectAll('.classifier-head-link')
    .filter(function () {
      let edge = d3.select(this);
      let sourceLayer = getEdgeNumericAttr(edge, 'data-source-layer');
      let targetLayer = getEdgeNumericAttr(edge, 'data-target-layer');
      return sourceLayer === -1 ||
        (Number.isFinite(sourceLayer) && sourceLayer <= layerIndex) ||
        (Number.isFinite(targetLayer) && targetLayer <= layerIndex);
    });
  let shiftedOutputEdges = svg.selectAll('.dense-output')
    .filter(function () {
      let lastDenseLayerIndex = layerLayouts.length - 1;
      return lastDenseLayerIndex <= layerIndex &&
        Number.isFinite(+d3.select(this).attr('data-source-x'));
    });
  let subduedDenseOutputEdges = svg.selectAll('.dense-output')
    .filter(function () {
      let sourceLayer = getEdgeNumericAttr(d3.select(this), 'data-source-layer');
      return Number.isFinite(sourceLayer) && sourceLayer > layerIndex;
    });
  let subduedPreviousEdges = headGroup.selectAll('.classifier-head-link')
    .filter(function () {
      let edge = d3.select(this);
      let targetLayer = getEdgeNumericAttr(edge, 'data-target-layer');
      return targetLayer === layerIndex && !edge.classed('dense-output');
    });
  let subduedNextEdges = nextLayout
    ? headGroup.selectAll('.classifier-head-dense-link')
      .filter(function () {
        return getEdgeNumericAttr(d3.select(this), 'data-source-layer') === layerIndex;
      })
    : svg.selectAll('.dense-output')
      .filter(function () {
        return getEdgeNumericAttr(d3.select(this), 'data-source-layer') === layerIndex;
      });
  let shiftedLogitDenseEdges = svg.select('.underneath').selectAll('path.softmax-edge')
    .filter(function () {
      let sourceLayer = getEdgeNumericAttr(d3.select(this), 'data-source-layer');
      return Number.isFinite(sourceLayer) && sourceLayer <= layerIndex;
    });

  const getEdgePath = (shift) => function () {
    let edge = d3.select(this);
    let sourceLayer = getEdgeNumericAttr(edge, 'data-source-layer');
    let targetLayer = getEdgeNumericAttr(edge, 'data-target-layer');
    let softmaxSourceShift = getEdgeNumericAttr(edge, 'data-softmax-source-shift');
    let sourceX = +edge.attr('data-source-x') +
      (sourceLayer === -1 && Number.isFinite(softmaxSourceShift) ? softmaxSourceShift : 0) +
      (sourceLayer === -1 ||
        (Number.isFinite(sourceLayer) && sourceLayer <= layerIndex) ? shift : 0);
    let sourceY = +edge.attr('data-source-y');
    let targetX = +edge.attr('data-target-x') +
      (Number.isFinite(targetLayer) && targetLayer <= layerIndex ? shift : 0);
    let targetY = +edge.attr('data-target-y');

    return linkGen({
      source: { x: sourceX, y: sourceY },
      target: { x: targetX, y: targetY },
    });
  };
  const getOutputEdgePath = (shift) => function () {
    let edge = d3.select(this);
    let sourceX = +edge.attr('data-source-x');
    let sourceY = +edge.attr('data-source-y');
    let targetX = +edge.attr('data-target-x') - shift;
    let targetY = +edge.attr('data-target-y');

    return linkGen({
      source: { x: sourceX, y: sourceY },
      target: { x: targetX, y: targetY },
    });
  };
  const getLogitDenseEdgePath = (shift) => function () {
    let edge = d3.select(this);
    let sourceLayer = getEdgeNumericAttr(edge, 'data-source-layer');
    let sourceX = +edge.attr('data-source-x') +
      (sourceLayer <= layerIndex ? shift : 0);
    let sourceY = +edge.attr('data-source-y');
    let targetX = +edge.attr('data-target-x');
    let targetY = +edge.attr('data-target-y');

    return linkGen({
      source: { x: sourceX, y: sourceY },
      target: { x: targetX, y: targetY },
    });
  };
  const getOverviewEdgePath = (shift) => (edge) => {
    let sourceX = edge.source.x +
      (shiftedOverviewLayerSet.has(edge.sourceLayerIndex) ? shift : 0);
    let targetX = edge.target.x +
      (shiftedOverviewLayerSet.has(edge.targetLayerIndex) ? shift : 0);

    return linkGen({
      source: { x: sourceX, y: edge.source.y },
      target: { x: targetX, y: edge.target.y },
    });
  };

  if (!skipLayoutShift) {
    shiftedOverviewLayerClones.selectAll('image.node-image')
      .style('opacity', 1);

    subduedPrevFeatureElements
      .interrupt('softmax')
      .attr('data-dense-detail-base-opacity', function () {
        return d3.select(this).style('opacity') || 1;
      })
      .transition('softmax')
      .duration(duration)
      .ease(d3.easeCubicInOut)
      .style('opacity', 0.18);

    directPrevFeatureLabels.raise();

    shiftedOverviewLayerClones
      .interrupt('softmax')
      .transition('softmax')
      .duration(duration)
      .ease(d3.easeCubicInOut)
      .attr('transform', `translate(${upstreamShift}, 0)`);

    shiftedPrevFeatureLayer
      .interrupt('softmax')
      .attr('data-dense-detail-shifted', 'true')
      .transition('softmax')
      .duration(duration)
      .ease(d3.easeCubicInOut)
      .attr('transform', `translate(${upstreamShift}, 0)`);

    shiftedPrevFeatureLabels
      .interrupt('softmax')
      .transition('softmax')
      .duration(duration)
      .ease(d3.easeCubicInOut)
      .attr('transform', function () {
        let label = d3.select(this);
        let baseTransform = label.attr('data-dense-detail-base-transform') ||
          label.attr('transform') || '';
        label.attr('data-dense-detail-base-transform', baseTransform);
        return `${baseTransform} translate(${upstreamShift}, 0)`;
      });

    shiftedFlattenLayer
      .interrupt('softmax')
      .transition('softmax')
      .duration(duration)
      .ease(d3.easeCubicInOut)
      .attr('transform', `translate(${upstreamShift}, 0)`);

    fixedOperationEdges
      .interrupt('softmax')
      .transition('softmax')
      .duration(duration)
      .ease(d3.easeCubicInOut)
      .attr('transform', function () {
        let edge = d3.select(this);
        let baseTransform = edge.attr('data-dense-detail-base-transform') ||
          edge.attr('transform') || '';
        edge.attr('data-dense-detail-base-transform', baseTransform);
        return `${baseTransform} translate(${-upstreamShift}, 0)`;
      });

    shiftedFlattenLabels
      .interrupt('softmax')
      .transition('softmax')
      .duration(duration)
      .ease(d3.easeCubicInOut)
      .attr('transform', function () {
        let label = d3.select(this);
        let baseTransform = label.attr('data-dense-detail-base-transform') ||
          label.attr('transform') || '';
        label.attr('data-dense-detail-base-transform', baseTransform);
        return `${baseTransform} translate(${upstreamShift}, 0)`;
      });

    shiftedFlattenAnnotations
      .transition('softmax')
      .duration(duration)
      .ease(d3.easeCubicInOut)
      .attr('transform', function () {
        let annotation = d3.select(this);
        let baseTransform = annotation.attr('data-dense-detail-base-transform') ||
          annotation.attr('transform') || '';
        annotation.attr('data-dense-detail-base-transform', baseTransform);
        return `${baseTransform} translate(${upstreamShift}, 0)`;
      });

    shiftedLegends = shiftDenseDetailLegends({
      prevFeatureLayerIndex,
      layerLayouts,
      layerIndex,
      nodeSize,
      upstreamShift,
      duration,
    });

    shiftedLayers
      .transition('softmax')
      .duration(duration)
      .ease(d3.easeCubicInOut)
      .attr('transform', `translate(${upstreamShift}, 0)`);

    shiftedEdges
      .transition('softmax')
      .duration(duration)
      .ease(d3.easeCubicInOut)
      .attr('d', getEdgePath(upstreamShift));

    subduedPreviousEdges
      .transition('softmax')
      .duration(duration)
      .ease(d3.easeCubicInOut)
      .style('opacity', 0)
      .style('stroke-width', edgeStrokeWidth)
      .style('stroke', edgeInitColor);

    subduedNextEdges
      .transition('softmax')
      .duration(duration)
      .ease(d3.easeCubicInOut)
      .style('opacity', 0)
      .style('stroke-width', edgeStrokeWidth)
      .style('stroke', edgeInitColor);

    shiftedOutputEdges
      .transition('softmax')
      .duration(duration)
      .ease(d3.easeCubicInOut)
      .attr('d', getOutputEdgePath(upstreamShift));

    subduedDenseOutputEdges
      .transition('softmax')
      .duration(duration)
      .ease(d3.easeCubicInOut)
      .style('opacity', 0.055)
      .style('stroke-width', edgeStrokeWidth)
      .style('stroke', edgeInitColor);

    shiftedLogitDenseEdges
      .transition('softmax')
      .duration(duration)
      .ease(d3.easeCubicInOut)
      .attr('d', getLogitDenseEdgePath(upstreamShift));

    shiftedOverviewEdges
      .transition('softmax')
      .duration(duration)
      .ease(d3.easeCubicInOut)
      .attr('d', getOverviewEdgePath(upstreamShift));
  }

  let detailGroup = headGroup.append('g')
    .attr('class',
      `classifier-dense-detail classifier-dense-detail-${layerIndex}-${node.index}`)
    .attr('transform', `translate(${panelX}, ${panelY}) scale(0.92)`)
    .style('opacity', 0)
    .style('pointer-events', 'all')
    .on('click', () => {
      d3.event.stopPropagation();
    });

  detailGroup.append('rect')
    .attr('class', 'classifier-dense-detail-halo')
    .attr('x', shiftedNodeX - panelX - 7)
    .attr('y', nodeY - panelY - 7)
    .attr('width', nodeSize + 14)
    .attr('height', nodeSize + 14)
    .attr('rx', 9)
    .attr('ry', 9)
    .style('fill', 'none')
    .style('stroke', '#9ACFF2')
    .style('stroke-width', 1.6)
    .style('filter', 'drop-shadow(0 0 9px rgba(126, 190, 232, 0.62))');

  detailGroup.append('rect')
    .attr('class', 'classifier-dense-detail-glow')
    .attr('x', shiftedNodeX - panelX - 14)
    .attr('y', nodeY - panelY - 14)
    .attr('width', nodeSize + 28)
    .attr('height', nodeSize + 28)
    .attr('rx', 18)
    .attr('ry', 18)
    .style('fill', '#DFF2FF')
    .style('stroke', 'none')
    .style('opacity', 0.22);

  detailGroup.select('.classifier-dense-detail-glow').lower();

  detailGroup.append('path')
    .attr('d', d3.linkHorizontal()
      .x(d => d.x)
      .y(d => d.y)({
        source: {
          x: shiftedNodeX - panelX + nodeSize + 6,
          y: nodeY - panelY + nodeSize / 2,
        },
        target: { x: 0, y: panelHeight / 2 },
      }))
    .style('fill', 'none')
    .style('stroke', '#BFD7EC')
    .style('stroke-width', 1.2)
    .style('pointer-events', 'none');

  detailGroup.append('rect')
    .attr('class', 'classifier-dense-detail-computation-bg')
    .attr('x', -5)
    .attr('y', -4)
    .attr('width', panelWidth + 10)
    .attr('height', panelHeight + 8)
    .attr('rx', 15)
    .attr('ry', 15)
    .style('fill', '#FBFEFF')
    .style('stroke', '#DCEEF8')
    .style('stroke-width', 0.6)
    .style('opacity', 0.74);

  let principleGroup = detailGroup.append('g')
    .attr('class', 'classifier-dense-principle-panel')
    .attr('transform', 'translate(0, 0)');
  drawDensePrincipleAnimation(
    principleGroup,
    node.inputLinks.map((link) => ({
      input: link.source.output || 0,
      weight: link.weight || 0,
      sourceIndex: link.source.index,
    })),
    node.bias || 0,
    preActivation,
    output,
    activationStyle,
    panelWidth,
    panelHeight,
  );

  let selectedBridgeEdges = headGroup.append('g')
    .attr('class', 'classifier-dense-detail-bridges')
    .style('opacity', 1);
  let bridgeStart = {
    x: shiftedNodeX + nodeSize,
    y: nodeY + nodeSize / 2,
  };
  let bridgeMidIn = {
    x: panelX,
    y: panelY + panelHeight / 2,
  };
  let bridgeMidOut = {
    x: panelX + activationGateX,
    y: panelY + panelHeight / 2,
  };
  let reluOut = {
    x: panelX + panelWidth,
    y: panelY + panelHeight / 2,
  };
  let incomingLinks = node.inputLinks || [];
  let prevLayout = layerIndex > 0 ? layerLayouts[layerIndex - 1] : undefined;
  let visibleIncomingLinks = incomingLinks.filter((link) => {
    if (layerIndex === 0) {
      return typeof flattenAnchorForFlatIndex !== 'function' ||
        flattenAnchorForFlatIndex(link.source.index) !== undefined;
    }

    return (link.source?.output || 0) > 0 &&
      prevLayout?.slotByNodeIndex?.has(link.source.index);
  });
  let inputEdgeStyle = getDenseConnectionEdgeStyle(visibleIncomingLinks.length, {
    largeFanIn: layerIndex === 0,
  });
  let selectedInputEdges = selectedBridgeEdges.append('g')
    .attr('class', 'classifier-dense-detail-selected-inputs');

  const drawSelectedInputEdge = (sourcePoint) => {
    if (!sourcePoint ||
      !Number.isFinite(sourcePoint.x) ||
      !Number.isFinite(sourcePoint.y)) {
      return;
    }

    selectedInputEdges.append('path')
      .attr('class', 'classifier-dense-detail-selected-input-edge')
      .attr('d', linkGen({
        source: sourcePoint,
        target: {
          x: shiftedNodeX,
          y: nodeY + nodeSize / 2,
        },
      }))
      .style('fill', 'none')
      .style('stroke', inputEdgeStyle.color)
      .style('stroke-width', inputEdgeStyle.width)
      .style('stroke-dasharray', inputEdgeStyle.dasharray)
      .style('stroke-linecap', 'round')
      .style('stroke-linejoin', 'round')
      .style('opacity', inputEdgeStyle.opacity)
      .style('pointer-events', 'none');
  };

  if (layerIndex === 0 && typeof flattenAnchorForFlatIndex === 'function') {
    visibleIncomingLinks.forEach((link) => {
      let anchor = flattenAnchorForFlatIndex(link.source.index);
      if (!anchor) {
        return;
      }

      drawSelectedInputEdge({
        x: anchor.x + upstreamShift,
        y: anchor.y,
      });
    });
  } else if (layerIndex > 0) {
    visibleIncomingLinks.forEach((link) => {
      let sourceY = prevLayout?.getNodeCenterY(link.source.index);
      if (sourceY === undefined) {
        return;
      }

      drawSelectedInputEdge({
        x: prevLayout.x + nodeSize + upstreamShift,
        y: sourceY,
      });
    });
  }

  selectedBridgeEdges.append('path')
    .attr('class', 'classifier-dense-detail-input-edge')
    .attr('d', linkGen({
      source: bridgeStart,
      target: bridgeMidIn,
    }))
    .style('fill', 'none')
    .style('stroke', '#8EA9BF')
    .style('stroke-width', 1.25)
    .style('opacity', 0.74)
    .style('pointer-events', 'none');

  selectedBridgeEdges.append('path')
    .attr('class', 'classifier-dense-detail-internal-edge')
    .attr('d', linkGen({
      source: bridgeMidOut,
      target: reluOut,
    }))
    .style('fill', 'none')
    .style('stroke', output > 0 ? '#8B75D7' : '#B7C7D9')
    .style('stroke-width', output > 0 ? 1.05 : 0.75)
    .style('opacity', output > 0 ? 0.72 : 0.32)
    .style('pointer-events', 'none');

  if (nextLayout) {
    headGroup.selectAll('.classifier-head-dense-link')
      .filter(function () {
        return +d3.select(this).attr('data-source-layer') === layerIndex;
      })
      .transition('softmax')
      .duration(duration)
      .ease(d3.easeCubicInOut)
      .style('opacity', 0);

    let outgoingTargets = (node.outputLinks || [])
      .map((link) => {
        let nextY = nextLayout.getNodeCenterY(link.dest.index);
        return {
          link,
          nextY,
          contribution: node.output * link.weight,
        };
      })
      .filter((entry) => entry.nextY !== undefined);
    let emphasizedTargetIndexes = new Set(outgoingTargets
      .map((entry) => ({
        targetIndex: entry.link.dest.index,
        contribution: entry.contribution,
      }))
      .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
      .slice(0, 8)
      .map((entry) => entry.targetIndex));

    outgoingTargets
      .forEach(({ link, nextY, contribution }) => {
        let isEmphasized = emphasizedTargetIndexes.has(link.dest.index);
        let edgeStyle = getDenseConnectionEdgeStyle(outgoingTargets.length, {
          emphasized: isEmphasized,
        });
        let edgeTitle = node.output > 0
          ? `activation * next weight = ${formatDenseValue(contribution)}`
          : `blocked after ReLU: activation is ${formatDenseValue(node.output)}`;

        let outgoingEdge = selectedBridgeEdges.append('path')
          .attr('class', 'classifier-dense-detail-output-edge')
          .attr('d', linkGen({
            source: reluOut,
            target: {
              x: nextLayout.x,
              y: nextY,
            },
          }))
          .style('fill', 'none')
          .style('stroke', edgeStyle.color)
          .style('stroke-width', edgeStyle.width)
          .style('stroke-dasharray', edgeStyle.dasharray)
          .style('stroke-linecap', 'round')
          .style('opacity', edgeStyle.opacity)
          .style('pointer-events', 'stroke');
        outgoingEdge.append('title').text(edgeTitle);
        if (isEmphasized) {
          animateDenseSignalPath(outgoingEdge, 980, 760);
          drawDenseOutgoingTargetMarker(
            selectedBridgeEdges,
            nextLayout.x,
            nextY,
            edgeStyle,
            edgeTitle,
          );
        }
      });

    let downstreamContextLinks = outgoingTargets
      .filter((entry) => emphasizedTargetIndexes.has(entry.link.dest.index))
      .map((entry) => {
        let sourceY = entry.nextY;
        let outputEdge = svg.selectAll('.dense-output')
          .filter(function () {
            let edge = d3.select(this);
            return getEdgeNumericAttr(edge, 'data-source-layer') === layerIndex + 1 &&
              getEdgeNumericAttr(edge, 'data-source-index') === entry.link.dest.index;
          })
          .filter(function (_, index) {
            return index === 0;
          });

        return {
          sourceY,
          sourceIndex: entry.link.dest.index,
          outputEdge,
        };
      })
      .filter((entry) => entry.sourceY !== undefined && !entry.outputEdge.empty());

    if (downstreamContextLinks.length) {
      let firstOutputEdge = downstreamContextLinks[0].outputEdge;
      let fallbackBiasX = +firstOutputEdge.attr('data-target-x');
      let fallbackBiasY = +firstOutputEdge.attr('data-target-y');
      let biasInputPoint = getBiasPlusInputPoint(
        svg,
        selectedBridgeEdges,
        fallbackBiasX,
        fallbackBiasY,
      );

      downstreamContextLinks.forEach((entry) => {
        let contextEdgeStyle = getDenseConnectionEdgeStyle(downstreamContextLinks.length);
        let sourcePoint = {
          x: nextLayout.x + nodeSize,
          y: entry.sourceY,
        };
        let targetPoint = isInSoftmax
          ? getPathEndPoint(
            entry.outputEdge,
            selectedBridgeEdges,
            biasInputPoint.x,
            biasInputPoint.y,
          )
          : biasInputPoint;

        let contextEdge = selectedBridgeEdges.append('path')
          .attr('class', 'classifier-dense-detail-output-edge classifier-dense-detail-context-edge')
          .attr('data-source-layer', layerIndex + 1)
          .attr('data-source-index', entry.sourceIndex)
          .attr('d', linkGen({
            source: sourcePoint,
            target: targetPoint,
          }))
          .style('fill', 'none')
          .style('stroke', contextEdgeStyle.color)
          .style('stroke-width', contextEdgeStyle.width)
          .style('stroke-dasharray', contextEdgeStyle.dasharray)
          .style('stroke-linecap', 'round')
          .style('opacity', contextEdgeStyle.opacity)
          .style('pointer-events', 'none');

        contextEdge.append('title')
          .text('faint continuation from Dense 2 into the logit sum');
      });
    }

    headGroup.selectAll(`.classifier-head-node-link-${layerIndex}-${node.index}`)
      .filter('.classifier-head-dense-link')
      .transition('softmax')
      .duration(duration)
      .ease(d3.easeCubicInOut)
      .style('opacity', 0)
      .style('stroke-width', edgeStrokeWidth)
      .style('stroke', edgeInitColor);
  } else {
    svg.selectAll('.dense-output')
      .filter(function () {
        return +d3.select(this).attr('data-source-layer') === layerIndex;
      })
      .transition('softmax')
      .duration(duration)
      .ease(d3.easeCubicInOut)
      .style('opacity', 0);

    svg.selectAll('.dense-output')
      .filter(function () {
        return +d3.select(this).attr('data-source-layer') === layerIndex &&
          +d3.select(this).attr('data-source-index') === node.index;
      })
      .each(function () {
        let outputEdge = d3.select(this);
        let fallbackTargetX = +outputEdge.attr('data-target-x');
        let fallbackTargetY = +outputEdge.attr('data-target-y');
        let sumInputPoint = getBiasPlusInputPoint(
          svg,
          selectedBridgeEdges,
          fallbackTargetX,
          fallbackTargetY,
        );
        let currentOutputTarget = isInSoftmax
          ? getPathEndPoint(
            outputEdge,
            selectedBridgeEdges,
            sumInputPoint.x,
            sumInputPoint.y,
          )
          : sumInputPoint;
        let targetX = currentOutputTarget.x;
        let targetY = currentOutputTarget.y;
        let link = node.outputLinks.find((entry) =>
          entry.dest.index === +outputEdge.attr('data-target-index'));
        let contribution = node.output * (link?.weight || 0);
        let edgeStyle = getDenseConnectionEdgeStyle(node.outputLinks.length, {
          emphasized: true,
        });
        let edgeTitle = node.output > 0
          ? `activation * logit weight enters sum = ${formatDenseValue(contribution)}`
          : `blocked before logit: activation is ${formatDenseValue(node.output)}`;

        let outgoingEdge = selectedBridgeEdges.append('path')
          .attr('class', 'classifier-dense-detail-output-edge classifier-dense-detail-output-edge-active')
          .attr('data-target-index', +outputEdge.attr('data-target-index'))
          .attr('d', linkGen({
            source: reluOut,
            target: {
              x: targetX,
              y: targetY,
            },
          }))
          .style('fill', 'none')
          .style('stroke', edgeStyle.color)
          .style('stroke-width', edgeStyle.width)
          .style('stroke-dasharray', edgeStyle.dasharray)
          .style('stroke-linecap', 'round')
          .style('opacity', edgeStyle.opacity)
          .style('filter', node.output > 0
            ? 'drop-shadow(0 0 3px rgba(90, 90, 90, 0.22))'
            : null)
          .style('pointer-events', 'stroke');
        outgoingEdge.append('title').text(edgeTitle);
        drawDenseOutgoingTargetMarker(
          selectedBridgeEdges,
          targetX,
          targetY,
          edgeStyle,
          edgeTitle,
        );
      });

    svg.selectAll(`.dense-output.classifier-head-node-link-${layerIndex}-${node.index}`)
      .transition('softmax')
      .duration(duration)
      .ease(d3.easeCubicInOut)
      .attr('d', getOutputEdgePath(upstreamShift))
      .style('opacity', 0)
      .style('stroke-width', edgeStrokeWidth)
      .style('stroke', edgeInitColor);
  }

  selectedBridgeEdges.transition('softmax')
    .duration(duration)
    .ease(d3.easeCubicInOut)
    .style('opacity', 1);

  detailGroup.transition('softmax')
    .duration(duration)
    .ease(d3.easeCubicInOut)
    .style('opacity', 1)
    .attr('transform', `translate(${panelX}, ${panelY}) scale(1)`);

  if (!skipLayoutShift) {
    denseDetailState = {
      selectedLayerIndex: layerIndex,
      selectedNodeIndex: node.index,
      shiftedPrevFeatureLayer,
      shiftedPrevFeatureLabels,
      shiftedFlattenLayer,
      fixedOperationEdges,
      shiftedFlattenLabels,
      shiftedFlattenAnnotations,
      shiftedLegends,
      shiftedLayers,
      shiftedEdges,
      shiftedOutputEdges,
      shiftedLogitDenseEdges,
      shiftedOverviewEdges,
      shiftedOverviewLayerClones,
      subduedPrevFeatureElements,
      subduedDenseOutputEdges,
      subduedPreviousEdges,
      subduedNextEdges,
      selectedBridgeEdges,
      getEdgePath,
      getOutputEdgePath,
      getLogitDenseEdgePath,
      getOverviewEdgePath,
    };
    updateStyleTestDenseLayerHierarchy(layerIndex);
  }

  selectedBridgeEdges.raise();
  detailGroup.raise();
}

const redrawDenseDetailSelections = () => {
  let selections = [...denseDetailSelections];
  removeDenseNeuronDetail(false, { keepSelections: true });
  denseDetailSelections = selections;

  svg.selectAll('.classifier-hidden-node-border')
    .style('opacity', 0)
    .style('stroke-width', 1)
    .style('stroke', '#8F8F8F')
    .style('filter', null);

  if (!denseDetailSelections.length) {
    return;
  }

  drawDenseNeuronDetail({
    ...denseDetailSelections[0],
    skipExistingRemoval: true,
  });
}

const toggleDenseNeuronDetail = (arg) => {
  let currentSelection = denseDetailSelections[0];
  let isSameNode = currentSelection &&
    currentSelection.layerIndex === arg.layerIndex &&
    currentSelection.node.index === arg.node.index;

  if (isSameNode) {
    denseDetailSelections = [];
    removeDenseNeuronDetail(true);
    return;
  }

  denseDetailSelections = [arg];
  redrawDenseDetailSelections();
}

const drawHiddenDenseLayers = (arg) => {
  let group = arg.group;
  let denseLayers = arg.denseLayers;
  let startX = arg.startX;
  let endX = arg.endX;
  let anchorY = arg.anchorY;
  let outputTopY = arg.outputTopY;
  let outputBottomY = arg.outputBottomY;
  let selectedOutputNode = arg.selectedOutputNode;
  let flattenAnchorForFlatIndex = arg.flattenAnchorForFlatIndex;
  let prevFeatureLayerIndex = arg.prevFeatureLayerIndex;

  const isHiddenActivationLayer = (layer) => {
    let layerName = layer?.[0]?.layerName || '';
    return /^(relu|sigmoid|tanh)[_-]/i.test(layerName);
  };

  // AI_TEST_UI_VISUAL_CLEANUP:
  // Keep activation layers in the backend graph for computations, but do not
  // draw them as separate classifier-head columns. The visual stack should show
  // Dense_1 -> Dense_2, not Dense_1 -> ReLU_Dense_1 -> Dense_2 -> ReLU_Dense_2.
  let displayedDenseLayers = denseLayers.filter(
    (layer) => !isHiddenActivationLayer(layer),
  );

  if (!displayedDenseLayers.length) {
    return undefined;
  }

  let headGroup = group.append('g')
    .attr('class', 'classifier-head');

  let linkGen = d3.linkHorizontal()
    .x(d => d.x)
    .y(d => d.y);

  const getOrderedDenseLayers = () => {
    let orderedLayers = Array.from({ length: displayedDenseLayers.length }, () => []);
    let orderMaps = Array.from({ length: displayedDenseLayers.length }, () => new Map());

    let lastLayer = displayedDenseLayers[displayedDenseLayers.length - 1]
      .map((node) => ({
        node,
        score: Math.abs(selectedOutputNode.inputLinks[node.index]?.weight || 0),
      }))
      .sort((a, b) => b.score - a.score || a.node.index - b.node.index)
      .map((entry) => entry.node);

    orderedLayers[displayedDenseLayers.length - 1] = lastLayer;
    lastLayer.forEach((node, position) => {
      orderMaps[displayedDenseLayers.length - 1].set(node.index, position);
    });

    const getLinksToNextDisplayedLayer = (node, nextOrder) => {
      let directLinks = node.outputLinks
        .filter((link) => nextOrder.has(link.dest.index));
      if (directLinks.length) {
        return directLinks.map((link) => ({
          destIndex: link.dest.index,
          weight: link.weight,
        }));
      }

      return node.outputLinks.flatMap((link) => {
        if (!isHiddenActivationLayer([{ layerName: link.dest?.layerName }])) {
          return [];
        }

        return link.dest.outputLinks
          .filter((nextLink) => nextOrder.has(nextLink.dest.index))
          .map((nextLink) => ({
            destIndex: nextLink.dest.index,
            weight: nextLink.weight,
          }));
      });
    };

    for (let layerIndex = displayedDenseLayers.length - 2; layerIndex >= 0; layerIndex--) {
      let nextOrder = orderMaps[layerIndex + 1];
      let orderedLayer = displayedDenseLayers[layerIndex]
        .map((node) => {
          let weightedPositions = getLinksToNextDisplayedLayer(node, nextOrder).map((link) => ({
            position: nextOrder.get(link.destIndex) ?? 0,
            weight: Math.abs(link.weight),
          }));
          let totalWeight = weightedPositions.reduce((sum, entry) => sum + entry.weight, 0) || 1;
          let barycenter = weightedPositions.reduce(
            (sum, entry) => sum + entry.position * entry.weight,
            0,
          ) / totalWeight;
          return { node, barycenter };
        })
        .sort((a, b) => a.barycenter - b.barycenter || a.node.index - b.node.index)
        .map((entry) => entry.node);

      orderedLayers[layerIndex] = orderedLayer;
      orderedLayer.forEach((node, position) => {
        orderMaps[layerIndex].set(node.index, position);
      });
    }

    return { orderedLayers, orderMaps };
  };

  let { orderedLayers, orderMaps } = getOrderedDenseLayers();
  let availableWidth = Math.max(endX - startX, 180);
  let innerPad = Math.max(12, Math.min(28, availableWidth / 8));
  let headStartX = startX + innerPad;
  let minLayerSpan = Math.max(orderedLayers.length - 1, 0) * 74;
  let headEndX = Math.max(endX - innerPad, headStartX + minLayerSpan);
  let layerXs = orderedLayers.map((_, index) =>
    headStartX + ((headEndX - headStartX) * index) /
    Math.max(orderedLayers.length - 1, 1),
  );
  let nodeSize = Math.min(nodeLength, 30);
  let visibleEndCount = 10;
  let visibleActiveTargetCount = 4;
  let denseTopY = outputTopY;
  let denseBottomY = outputBottomY - nodeSize;
  let denseEdgeOpacity = 0.28;

  const isDenseNodeActive = (node) => (node?.output || 0) > 0;

  const injectActiveDenseSamples = (firstNodes, lastNodes, layerNodes) => {
    let visibleNodes = [...firstNodes, ...lastNodes];
    let visibleIndexes = new Set(visibleNodes.map((node) => node.index));
    let activeNodes = layerNodes.filter(isDenseNodeActive);
    let desiredActiveCount = Math.min(visibleActiveTargetCount, activeNodes.length);
    let visibleActiveCount = visibleNodes.filter(isDenseNodeActive).length;
    if (visibleActiveCount >= desiredActiveCount) {
      return { firstNodes, lastNodes };
    }

    let activeCandidates = activeNodes
      .filter((node) => !visibleIndexes.has(node.index))
      .sort((a, b) => (b.output || 0) - (a.output || 0) || a.index - b.index);
    let replacementGroups = [
      {
        nodes: firstNodes,
        indices: d3.range(firstNodes.length - 1, -1, -1),
      },
      {
        nodes: lastNodes,
        indices: d3.range(0, lastNodes.length),
      },
    ];

    activeCandidates.some((activeNode, activeCandidateIndex) => {
      if (visibleActiveCount >= desiredActiveCount) {
        return true;
      }

      for (let offset = 0; offset < replacementGroups.length; offset++) {
        let replacementGroup =
          replacementGroups[(activeCandidateIndex + offset) % replacementGroups.length];
        let replaceIndex = replacementGroup.indices.find((index) =>
          !isDenseNodeActive(replacementGroup.nodes[index]));

        if (replaceIndex === undefined) {
          continue;
        }

        let replacedNode = replacementGroup.nodes[replaceIndex];
        visibleIndexes.delete(replacedNode.index);
        replacementGroup.nodes[replaceIndex] = activeNode;
        visibleIndexes.add(activeNode.index);
        visibleActiveCount++;
        return false;
      }

      return false;
    });

    return { firstNodes, lastNodes };
  };

  const getDisplaySlots = (layerNodes) => {
    if (layerNodes.length <= visibleEndCount * 2) {
      return layerNodes.map((node) => ({ node }));
    }

    let firstNodes = layerNodes.slice(0, visibleEndCount);
    let lastNodes = layerNodes.slice(-visibleEndCount);
    ({ firstNodes, lastNodes } = injectActiveDenseSamples(
      firstNodes,
      lastNodes,
      layerNodes,
    ));

    return [
      ...firstNodes.map((node) => ({ node })),
      { isEllipsis: true },
      ...lastNodes.map((node) => ({ node })),
    ];
  };

  let layerLayouts = orderedLayers.map((layerNodes, layerIndex) => {
    let slots = getDisplaySlots(layerNodes);
    let slotByNodeIndex = new Map();

    slots.forEach((slot, slotIndex) => {
      if (slot.node) {
        slotByNodeIndex.set(slot.node.index, slotIndex);
      }
    });

    return {
      layerIndex,
      layerNodes,
      slots,
      slotByNodeIndex,
      x: layerXs[layerIndex],
    };
  });

  const getSlotY = (layout, slotIndex) => {
    if (layout.slots.length <= 1) {
      return anchorY + nodeLength / 2 - nodeSize / 2;
    }

    return denseTopY +
      ((denseBottomY - denseTopY) * slotIndex) /
      Math.max(layout.slots.length - 1, 1);
  };

  const getNodeY = (layerIndex, nodeIndex) => {
    let layout = layerLayouts[layerIndex];
    let slotIndex = layout?.slotByNodeIndex.get(nodeIndex);
    if (slotIndex === undefined) {
      return undefined;
    }
    return getSlotY(layout, slotIndex);
  };

  const getNodeCenterY = (layerIndex, nodeIndex) => {
    let y = getNodeY(layerIndex, nodeIndex);
    return y === undefined ? undefined : y + nodeSize / 2;
  };

  const getFullNodeY = (layerIndex, nodeIndex) => {
    let layerNodes = orderedLayers[layerIndex];
    let position = orderMaps[layerIndex]?.get(nodeIndex);
    if (!layerNodes || position === undefined) {
      return undefined;
    }

    if (layerNodes.length <= 1) {
      return anchorY + nodeLength / 2 - nodeSize / 2;
    }

    return denseTopY +
      ((denseBottomY - denseTopY) * position) /
      Math.max(layerNodes.length - 1, 1);
  };

  const getFullNodeCenterY = (layerIndex, nodeIndex) => {
    let y = getFullNodeY(layerIndex, nodeIndex);
    return y === undefined ? undefined : y + nodeSize / 2;
  };

  const getNodeColor = (node) => getDenseActivationStyle(node).fill;

  const isSelectedDenseNode = (layerIndex, nodeIndex) =>
    denseDetailSelections.some((selection) =>
      selection.layerIndex === layerIndex && selection.node.index === nodeIndex) ||
    (denseDetailState &&
      denseDetailState.selectedLayerIndex === layerIndex &&
      denseDetailState.selectedNodeIndex === nodeIndex);

  const restoreDenseNodeHoverState = (layerIndex, nodeIndex) => {
    svg.selectAll(`.classifier-head-node-link-${layerIndex}-${nodeIndex}`)
      .raise()
      .style('opacity', function () {
        if (!denseDetailState) {
          return denseEdgeOpacity;
        }

        let edge = d3.select(this);
        let sourceLayer = getEdgeNumericAttr(edge, 'data-source-layer');
        let targetLayer = getEdgeNumericAttr(edge, 'data-target-layer');

        if (sourceLayer === denseDetailState.selectedLayerIndex ||
          targetLayer === denseDetailState.selectedLayerIndex) {
          return 0;
        }

        if (edge.classed('dense-output')) {
          return 0.055;
        }

        return edge.attr('data-original-opacity') || denseEdgeOpacity;
      })
      .style('stroke-width', function () {
        return edgeStrokeWidth;
      })
      .style('stroke', function () {
        return edgeInitColor;
      });

    svg.selectAll(`.classifier-hidden-node-border-${layerIndex}-${nodeIndex}`)
      .style('opacity', isSelectedDenseNode(layerIndex, nodeIndex) ? 1 : 0)
      .style('stroke-width', isSelectedDenseNode(layerIndex, nodeIndex) ? 1.5 : 1)
      .style('stroke', isSelectedDenseNode(layerIndex, nodeIndex) ? '#8ACDF3' : '#8F8F8F')
      .style('filter', isSelectedDenseNode(layerIndex, nodeIndex)
        ? 'drop-shadow(0 0 6px rgba(126, 190, 232, 0.5))'
        : null);
  };

  layerLayouts.forEach((layout, layoutIndex) => {
    layout.getNodeCenterY = (nodeIndex) => getNodeCenterY(layoutIndex, nodeIndex);
    layout.getFullNodeCenterY = (nodeIndex) =>
      getFullNodeCenterY(layoutIndex, nodeIndex);
  });

  layerLayouts.forEach((layout, layerIndex) => {
    let layerNodes = layout.layerNodes;
    let layerX = layout.x;
    let layerGroup = headGroup.append('g')
      .attr('class', `classifier-hidden-layer classifier-hidden-layer-${layerIndex}`)
      .classed('is-style-muted', isStyleTestMode())
      .classed('is-style-active', false);
    let labelY = getSlotY(layout, 0) - 32;

    layerGroup.append('text')
      .attr('class', 'classifier-hidden-layer-label')
      .attr('x', layerX + nodeSize / 2)
      .attr('y', labelY)
      .style('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('font-weight', 600)
      .style('letter-spacing', isStyleTestMode() ? '0' : '0.05em')
      .style('opacity', 0.8)
      .text(isStyleTestMode()
        ? formatLayerLabel(layerNodes[0].layerName)
        : layerNodes[0].layerName.toUpperCase());

    layerGroup.append('text')
      .attr('class', 'classifier-hidden-layer-count')
      .attr('x', layerX + nodeSize / 2)
      .attr('y', labelY + 14)
      .style('text-anchor', 'middle')
      .style('font-size', '8px')
      .style('opacity', 0.7)
      .text(`(${layerNodes.length})`);

    layout.slots.forEach((slot, slotIndex) => {
      let nodeY = getSlotY(layout, slotIndex);

      if (slot.isEllipsis) {
        [0, 1, 2].forEach((dotIndex) => {
          layerGroup.append('circle')
            .attr('cx', layerX + nodeSize / 2)
            .attr('cy', nodeY + nodeSize / 2 - 6 + dotIndex * 6)
            .attr('r', 1.8)
            .style('fill', '#5A5A5A')
            .style('opacity', 0.85);
        });
        return;
      }

      let node = slot.node;
      let activationStyle = getDenseActivationStyle(node);
      let fillColor = getNodeColor(node);
      let nodeGroup = layerGroup.append('g')
        .attr('class', `classifier-hidden-node classifier-hidden-node-${layerIndex}-${node.index}`)
        .style('cursor', 'crosshair')
        .on('mouseover', () => {
          hoverInfoStore.set({
            show: true,
            text: `${node.layerName}: ${formater(node.output)} (${activationStyle.isActive ? 'activated' : 'not activated'})`
          });

          if (isSelectedDenseNode(layerIndex, node.index)) {
            return;
          }

          svg.selectAll(`.classifier-head-node-link-${layerIndex}-${node.index}`)
            .filter(function () {
              return !denseDetailState;
            })
            .raise()
            .style('opacity', 1)
            .style('stroke-width', 1)
            .style('stroke', edgeHoverColor);

          svg.selectAll(`.classifier-hidden-node-border-${layerIndex}-${node.index}`)
            .style('opacity', 1)
            .style('stroke-width', 1.2)
            .style('stroke', intermediateColor);
        })
        .on('mouseleave', () => {
          hoverInfoStore.set({
            show: false,
            text: `${node.layerName}: ${formater(node.output)} (${activationStyle.isActive ? 'activated' : 'not activated'})`
          });

          restoreDenseNodeHoverState(layerIndex, node.index);
        })
        .on('click', () => {
          d3.event.stopPropagation();
          toggleDenseNeuronDetail({
            headGroup,
            layerLayouts,
            layerIndex,
            node,
            nodeX: layerX,
            nodeY,
            nodeSize,
            prevFeatureLayerIndex,
            flattenAnchorForFlatIndex,
          });
        });

      nodeGroup.append('image')
        .attr('x', layerX)
        .attr('y', nodeY)
        .attr('width', nodeSize)
        .attr('height', nodeSize)
        .style('filter', activationStyle.glow)
        .attr('xlink:href', getScalarNodeImageHref(fillColor));

      nodeGroup.append('rect')
        .attr('class', `classifier-hidden-node-border classifier-hidden-node-border-${layerIndex}-${node.index}`)
        .attr('x', layerX)
        .attr('y', nodeY)
        .attr('width', nodeSize)
        .attr('height', nodeSize)
        .attr('rx', 5)
        .attr('ry', 5)
        .style('fill', 'none')
        .style('stroke', '#8F8F8F')
        .style('stroke-width', 1)
        .style('opacity', 0);
    });
  });

  const sampleEndLinks = (links, countPerEnd = 12) => {
    let sortedLinks = [...links].sort((a, b) => a.source.index - b.source.index);
    if (sortedLinks.length <= countPerEnd * 2) {
      return sortedLinks;
    }
    return [
      ...sortedLinks.slice(0, countPerEnd),
      ...sortedLinks.slice(-countPerEnd),
    ];
  };

  let firstLayout = layerLayouts[0];
  firstLayout.slots.forEach((slot) => {
    if (!slot.node) {
      return;
    }

    let targetY = getNodeCenterY(0, slot.node.index);
    sampleEndLinks(slot.node.inputLinks).forEach((link) => {
      let flattenAnchor = flattenAnchorForFlatIndex(link.source.index);
      if (!flattenAnchor) {
        return;
      }

      let denseTarget = {
        x: firstLayout.x,
        y: targetY,
      };
      headGroup.append('path')
        .attr('class',
          `classifier-head-link classifier-head-flatten-link classifier-head-node-link-0-${slot.node.index}`)
        .attr('d', linkGen({
          source: flattenAnchor,
          target: denseTarget,
        }))
        .attr('data-source-x', flattenAnchor.x)
        .attr('data-source-y', flattenAnchor.y)
        .attr('data-source-layer', -1)
        .attr('data-target-layer', 0)
        .attr('data-target-index', slot.node.index)
        .attr('data-target-x', denseTarget.x)
        .attr('data-target-y', denseTarget.y)
        .attr('data-original-opacity', denseEdgeOpacity)
        .style('fill', 'none')
        .style('stroke', edgeInitColor)
        .style('stroke-width', edgeStrokeWidth)
        .style('opacity', denseEdgeOpacity)
        .style('pointer-events', 'none');
    });
  });

  for (let layerIndex = 1; layerIndex < orderedLayers.length; layerIndex++) {
    let prevLayout = layerLayouts[layerIndex - 1];
    let curLayout = layerLayouts[layerIndex];

    const resolveDisplayedSourceNode = (sourceNode) => {
      if (!isHiddenActivationLayer([{ layerName: sourceNode?.layerName }])) {
        return sourceNode;
      }

      return sourceNode?.inputLinks?.[0]?.source || sourceNode;
    };

    curLayout.slots.forEach((slot) => {
      if (!slot.node) {
        return;
      }

      let targetY = getNodeCenterY(layerIndex, slot.node.index);
      slot.node.inputLinks
        .map((link) => ({
          ...link,
          displayedSource: resolveDisplayedSourceNode(link.source),
        }))
        .filter((link) =>
          prevLayout.slotByNodeIndex.has(link.displayedSource.index) &&
          (link.displayedSource?.output || 0) > 0)
        .forEach((link) => {
          let sourceY = getNodeCenterY(layerIndex - 1, link.displayedSource.index);
          if (sourceY === undefined || targetY === undefined) {
            return;
          }

        headGroup.append('path')
          .attr('class',
            `classifier-head-link classifier-head-dense-link ` +
            `classifier-head-node-link-${layerIndex - 1}-${link.displayedSource.index} ` +
            `classifier-head-node-link-${layerIndex}-${slot.node.index}`)
          .attr('d', linkGen({
            source: {
              x: prevLayout.x + nodeSize,
              y: sourceY,
            },
            target: {
              x: curLayout.x,
              y: targetY,
            },
          }))
          .attr('data-source-layer', layerIndex - 1)
          .attr('data-source-index', link.displayedSource.index)
          .attr('data-source-x', prevLayout.x + nodeSize)
          .attr('data-source-y', sourceY)
          .attr('data-target-layer', layerIndex)
          .attr('data-target-index', slot.node.index)
          .attr('data-target-x', curLayout.x)
          .attr('data-target-y', targetY)
          .attr('data-original-opacity', denseEdgeOpacity)
          .style('fill', 'none')
          .style('stroke', edgeInitColor)
          .style('stroke-width', edgeStrokeWidth)
          .style('opacity', denseEdgeOpacity)
          .style('pointer-events', 'none');
      });
    });
  }

  return {
    layerXs,
    orderedLayers,
    orderMaps,
    layerLayouts,
    nodeSize,
    getNodeY,
    getNodeCenterY,
    getFullNodeY,
    getFullNodeCenterY,
    headGroup,
  };
}

const moveLegend = (d, i, g, moveX, duration, restore) => {
  let legend = d3.select(g[i]);
  legend.interrupt('softmax');

  if (!restore) {
    let { x: previousLegendX, y: previousLegendY } = getTranslateCoords(legend);

    legend.transition('softmax')
      .duration(duration)
      .ease(d3.easeCubicInOut)
      .attr('transform', `translate(${previousLegendX - moveX}, ${previousLegendY})`);

    // If not in restore mode, we register the previous location to the DOM element
    legend.attr('data-preX', previousLegendX);
    legend.attr('data-preY', previousLegendY);
  } else {
    // Restore the recorded location
    let previousLegendX = +legend.attr('data-preX');
    let previousLegendY = +legend.attr('data-preY');

    legend.transition('softmax')
      .duration(duration)
      .ease(d3.easeCubicInOut)
      .attr('transform', `translate(${previousLegendX}, ${previousLegendY})`);
  }
}

const logitCircleMouseOverHandler = (i) => {
  // Update the hover info UI
  hoverInfoStore.set({
    show: true,
    text: `Logit: ${formater(logits[i])}`
  })

  // Highlight the text in the detail view
  softmaxDetailViewInfo.highlightI = i;
  softmaxDetailViewStore.set(softmaxDetailViewInfo);

  let logitLayerLower = svg.select('.underneath');
  let intermediateLayer = svg.select('.intermediate-layer');

  // Highlight the circle
  svg.selectAll(`#logit-circle-${i}`)
    .style('stroke-width', 2);

  // Highlight the associated plus symbol
  intermediateLayer.select(`#plus-symbol-clone-${i}`)
    .style('opacity', 1)
    .select('circle')
    .style('fill', d => d.fill);

  // Raise the associated edge group
  logitLayerLower.select(`#logit-lower-${i}`).raise();
  svg.selectAll('.classifier-dense-detail-output-edge-active')
    .filter(function () {
      return getEdgeNumericAttr(d3.select(this), 'data-target-index') !== i;
    })
    .style('opacity', 0.12)
    .style('stroke-width', 0.75)
    .style('stroke-dasharray', '1.6 3.2')
    .style('filter', null);
  svg.selectAll('.classifier-dense-detail-output-edge-active')
    .filter(function () {
      return getEdgeNumericAttr(d3.select(this), 'data-target-index') === i;
    })
    .style('opacity', 1)
    .style('stroke-width', 1.35)
    .style('stroke-dasharray', null);

  // Highlight the associated edges
  logitLayerLower.selectAll(`.softmax-abstract-edge-${i}`)
    .style('stroke-width', 0.8)
    .style('stroke', '#E0E0E0');

  logitLayerLower.selectAll(`.softmax-edge-${i}`)
    .style('stroke-width', 1)
    .style('stroke', '#E0E0E0');

  logitLayerLower.selectAll(`.logit-output-edge-${i}`)
    .style('stroke-width', 3)
    .style('stroke', '#E0E0E0');

  svg.selectAll(`.logit-output-edge-${i}`)
    .style('stroke-width', 3)
    .style('stroke', '#E0E0E0');
}

const logitCircleMouseLeaveHandler = (i) => {
  // screenshot
  // return;

  // Update the hover info UI
  hoverInfoStore.set({
    show: false,
    text: `Logit: ${formater(logits[i])}`
  })

  // Dehighlight the text in the detail view
  softmaxDetailViewInfo.highlightI = -1;
  softmaxDetailViewStore.set(softmaxDetailViewInfo);

  let logitLayerLower = svg.select('.underneath');
  let intermediateLayer = svg.select('.intermediate-layer');

  // Restore the circle
  svg.selectAll(`#logit-circle-${i}`)
    .style('stroke-width', 1);

  // Restore the associated plus symbol
  intermediateLayer.select(`#plus-symbol-clone-${i}`)
    .style('opacity', 0.2);
  svg.selectAll('.classifier-dense-detail-output-edge-active')
    .style('opacity', 1)
    .style('stroke-width', 1.35)
    .style('stroke-dasharray', null);

  // Restore the associated edges
  logitLayerLower.selectAll(`.softmax-abstract-edge-${i}`)
    .style('stroke-width', 0.2)
    .style('stroke', '#EDEDED');

  logitLayerLower.selectAll(`.softmax-edge-${i}`)
    .style('stroke-width', 0.2)
    .style('stroke', '#F1F1F1');

  logitLayerLower.selectAll(`.logit-output-edge-${i}`)
    .style('stroke-width', 1.2)
    .style('stroke', '#E5E5E5');

  svg.selectAll(`.logit-output-edge-${i}`)
    .style('stroke-width', 1.2)
    .style('stroke', '#E5E5E5');
}

// This function is binded to the detail view in Overview.svelte
export const softmaxDetailViewMouseOverHandler = (event) => {
  logitCircleMouseOverHandler(event.detail.curI);
}

// This function is binded to the detail view in Overview.svelte
export const softmaxDetailViewMouseLeaveHandler = (event) => {
  logitCircleMouseLeaveHandler(event.detail.curI);
}

const drawLogitLayer = (arg) => {
  let {
    flatten: flattenLayerData,
    output: outputLayerData,
  } = getClassifierHead();
  let curLayerIndex = arg.curLayerIndex,
    outputLayerIndex = arg.outputLayerIndex,
    prevFeatureLayerIndex = arg.prevFeatureLayerIndex,
    moveX = arg.moveX,
    softmaxLeftMid = arg.softmaxLeftMid,
    selectedI = arg.selectedI,
    intermediateX1 = arg.intermediateX1,
    intermediateX2 = arg.intermediateX2,
    pixelWidth = arg.pixelWidth,
    pixelHeight = arg.pixelHeight,
    topY = arg.topY,
    bottomY = arg.bottomY,
    softmaxX = arg.softmaxX,
    middleGap = arg.middleGap,
    middleRectHeight = arg.middleRectHeight,
    symbolGroup = arg.symbolGroup,
    symbolX = arg.symbolX,
    flattenRange = arg.flattenRange,
    logitSpacingFactor = arg.logitSpacingFactor ?? 0.8,
    plusColumnX = arg.plusColumnX ?? symbolX - moveX,
    showSelectedPlusClone = arg.showSelectedPlusClone ?? true,
    hiddenLayout = arg.hiddenLayout;

  const prevFeatureLayerCount = cnn[prevFeatureLayerIndex].length;
  const firstChannelIndex = 0;
  const lastChannelIndex = prevFeatureLayerCount - 1;
  const middleChannelCount = Math.max(prevFeatureLayerCount - 2, 0);

  let logitLayer = svg.select('.intermediate-layer')
    .append('g')
    .attr('class', 'logit-layer')
    .raise();

  // Minotr layer ordering change
  let tempPlusSymbol = undefined;
  if (showSelectedPlusClone) {
    let tempCloneNode = symbolGroup.node().cloneNode(true);
    d3.select(tempCloneNode)
      .attr('class', 'temp-clone-plus-symbol')
      .attr('transform', `translate(${plusColumnX},
        ${nodeCoordinate[outputLayerIndex][selectedI].y + nodeLength / 2})`)
      // Keep the visible clone from intercepting events on the real symbol.
      .style('pointer-events', 'none');

    tempPlusSymbol = logitLayer.append(() => tempCloneNode);
  }

  svg.select('.softmax-symbol').raise();

  let logitLayerLower = svg.select('.underneath')
    .append('g')
    .attr('class', 'logit-layer-lower')
    .lower();

  // Use circles to encode logit values
  let centerX = softmaxLeftMid - moveX * logitSpacingFactor;
  let logitRadius = 8;
  let logitLeftX = centerX - logitRadius;
  let logitRightX = centerX + logitRadius;

  logits = getLogitsForOutputLayer(flattenLayerData, outputLayerData);
  let logitExtent = getSafeLogitExtent(logits);

  // Construct a color scale for the logit values
  let logitColorScale = d3.scaleLinear()
    .domain(logitExtent)
    .range([0.2, 1]);
  let linkGen = d3.linkHorizontal()
    .x(d => d.x)
    .y(d => d.y);

  logitLayer.append('circle')
    .attr('class', 'logit-circle')
    .attr('id', `logit-circle-${selectedI}`)
    .attr('cx', centerX)
    .attr('cy', nodeCoordinate[outputLayerIndex][selectedI].y + nodeLength / 2)
    .attr('r', logitRadius)
    .style('fill', layerColorScales.logit(logitColorScale(logits[selectedI])))
    .style('cursor', 'crosshair')
    .style('pointer-events', 'all')
    .style('stroke', intermediateColor)
    .on('mouseover', () => logitCircleMouseOverHandler(selectedI))
    .on('mouseleave', () => logitCircleMouseLeaveHandler(selectedI))
    .on('click', () => { d3.event.stopPropagation() });

  // Show the logit circle corresponding label
  let softmaxDetailAnnotation = svg.select('.intermediate-layer-annotation')
    .select('.softmax-detail-annoataion');

  softmaxDetailAnnotation.select(`#logit-text-${selectedI}`)
    .style('opacity', 1);

  if (tempPlusSymbol) {
    tempPlusSymbol.raise();
  } else {
    symbolGroup.raise();
  }

  // Draw the selected plus -> logit -> softmax links. Splitting at the circle
  // keeps the edge visually attached to the logit node instead of crossing it.
  logitLayer.append('line')
    .attr('class', `logit-output-edge-${selectedI}`)
    .attr('x1', plusColumnX + plusSymbolRadius * 2)
    .attr('x2', logitLeftX)
    .attr('y1', nodeCoordinate[outputLayerIndex][selectedI].y + nodeLength / 2)
    .attr('y2', nodeCoordinate[outputLayerIndex][selectedI].y + nodeLength / 2)
    .style('fill', 'none')
    .style('stroke', '#EAEAEA')
    .style('stroke-width', '1.2')
    .lower();

  logitLayer.append('line')
    .attr('class', `logit-output-edge-${selectedI}`)
    .attr('x1', logitRightX)
    .attr('x2', softmaxX)
    .attr('y1', nodeCoordinate[outputLayerIndex][selectedI].y + nodeLength / 2)
    .attr('y2', nodeCoordinate[outputLayerIndex][selectedI].y + nodeLength / 2)
    .style('fill', 'none')
    .style('stroke', '#EAEAEA')
    .style('stroke-width', '1.2')
    .lower();

  // Add the flatten to logit links
  let linkData = [];
  let flattenLength = flattenLayerData.length / prevFeatureLayerCount;
  let underneathIs = [...Array(outputLayerData.length).keys()]
    .filter(d => d != selectedI);
  let curIIndex = 0;

  const drawOneEdgeGroup = () => {
    // Only draw the new group if it is in the softmax mode
    if (!allowsSoftmaxAnimation) {
      svg.select('.underneath')
        .selectAll(`.logit-lower`)
        .remove();
      return;
    }

    let curI = underneathIs[curIIndex];

    let curEdgeGroup = svg.select('.underneath')
      .select(`#logit-lower-${curI}`);

    if (curEdgeGroup.empty()) {
      curEdgeGroup = svg.select('.underneath')
        .append('g')
        .attr('class', 'logit-lower')
        .attr('id', `logit-lower-${curI}`)
        .style('opacity', 0);

      if (hiddenLayout) {
        let lastDenseLayerIndex = hiddenLayout.orderedLayers.length - 1;
        outputLayerData[curI].inputLinks.forEach((link) => {
          let sourceY = hiddenLayout.getNodeCenterY(lastDenseLayerIndex, link.source.index);
          if (sourceY === undefined) {
            return;
          }

          linkData.push({
            source: {
              x: hiddenLayout.layerXs[lastDenseLayerIndex] + hiddenLayout.nodeSize,
              y: sourceY,
            },
            target: {
              x: plusColumnX - plusSymbolRadius,
              y: nodeCoordinate[outputLayerIndex][curI].y + nodeLength / 2,
            },
            index: link.source.index,
            weight: link.weight,
            sourceLayer: lastDenseLayerIndex,
            sourceIndex: link.source.index,
            targetIndex: link.dest.index,
            color: '#F1F1F1',
            width: 0.5,
            opacity: 1,
            class: `softmax-edge softmax-edge-${curI}`
          });
        });
      } else {
        // Hack: now show all edges, only draw 1/3 of the actual edges
        for (let f = 0; f < flattenLength; f += 3) {
          let loopFactors = [firstChannelIndex, lastChannelIndex];
          loopFactors.forEach(l => {
            let factoredF = f + l * flattenLength;

            // Flatten -> output
            linkData.push({
              source: {
                x: intermediateX1 + pixelWidth + 3 - moveX,
                y: l === 0 ? topY + f * pixelHeight : bottomY + f * pixelHeight
              },
              target: {
                x: intermediateX2 - moveX,
                y: nodeCoordinate[outputLayerIndex][curI].y + nodeLength / 2
              },
              index: factoredF,
              weight: flattenLayerData[factoredF].outputLinks[curI].weight,
              color: '#F1F1F1',
              width: 0.5,
              opacity: 1,
              class: `softmax-edge softmax-edge-${curI}`
            });
          });
        }

        // Draw middle rect to logits
        for (let vi = 0; vi < middleChannelCount; vi++) {
          linkData.push({
            source: {
              x: intermediateX1 + pixelWidth + 3 - moveX,
              y: topY + flattenLength * pixelHeight + middleGap * (vi + 1) +
                middleRectHeight * (vi + 0.5)
            },
            target: {
              x: intermediateX2 - moveX,
              y: nodeCoordinate[outputLayerIndex][curI].y + nodeLength / 2
            },
            index: -1,
            color: '#EDEDED',
            width: 0.5,
            opacity: 1,
            class: `softmax-abstract-edge-${curI}`
          });
        }
      }

      // Render the edges on the underneath layer
      curEdgeGroup.selectAll(`path.softmax-edge-${curI}`)
        .data(linkData)
        .enter()
        .append('path')
        .attr('class', d => d.class)
        .attr('id', d => `edge-${d.name}`)
        .attr('d', d => linkGen({ source: d.source, target: d.target }))
        .attr('data-source-layer', d => d.sourceLayer ?? null)
        .attr('data-source-index', d => d.sourceIndex ?? d.index ?? null)
        .attr('data-source-x', d => d.source.x)
        .attr('data-source-y', d => d.source.y)
        .attr('data-target-index', d => d.targetIndex ?? null)
        .attr('data-target-x', d => d.target.x)
        .attr('data-target-y', d => d.target.y)
        .style('fill', 'none')
        .style('stroke-width', d => d.width)
        .style('stroke', d => d.color === undefined ? intermediateColor : d.color)
        .style('opacity', d => d.opacity)
        .style('pointer-events', 'none');
    }

    let curNodeGroup = logitLayer.append('g')
      .attr('class', `logit-layer-${curI}`)
      .style('opacity', 0);

    // Draw the plus symbol
    let symbolClone = symbolGroup.clone(true)
      .style('opacity', 0);

    // Change the style of the clone
    symbolClone.attr('class', 'plus-symbol-clone')
      .attr('id', `plus-symbol-clone-${curI}`)
      .select('circle')
      .datum({
        fill: gappedColorScale(layerColorScales.weight,
          flattenRange, outputLayerData[curI].bias, 0.35)
      })
      .style('pointer-events', 'none')
      .style('fill', '#E5E5E5');

    symbolClone.attr('transform', `translate(${plusColumnX},
      ${nodeCoordinate[outputLayerIndex][curI].y + nodeLength / 2})`);

    // Draw the outter link using only merged path
    let outputEdgeD1 = linkGen({
      source: {
        x: plusColumnX + plusSymbolRadius * 2,
        y: nodeCoordinate[outputLayerIndex][curI].y + nodeLength / 2
      },
      target: {
        x: logitLeftX,
        y: nodeCoordinate[outputLayerIndex][curI].y + nodeLength / 2
      }
    });

    let outputEdgeD2 = linkGen({
      source: {
        x: logitRightX,
        y: nodeCoordinate[outputLayerIndex][curI].y + nodeLength / 2
      },
      target: {
        x: softmaxX,
        y: nodeCoordinate[outputLayerIndex][selectedI].y + nodeLength / 2
      }
    });

    // There are ways to combine these two paths into one. However, the animation
    // for merged path is not continuous, so we use two saperate paths here.

    let outputEdge1 = logitLayerLower.append('path')
      .attr('class', `logit-output-edge-${curI}`)
      .attr('d', outputEdgeD1)
      .style('fill', 'none')
      .style('stroke', '#EAEAEA')
      .style('stroke-width', '1.2');

    let outputEdge2 = logitLayerLower.append('path')
      .attr('class', `logit-output-edge-${curI}`)
      .attr('d', outputEdgeD2)
      .style('fill', 'none')
      .style('stroke', '#EAEAEA')
      .style('stroke-width', '1.2');

    let outputEdgeLength1 = outputEdge1.node().getTotalLength();
    let outputEdgeLength2 = outputEdge2.node().getTotalLength();
    let totalLength = outputEdgeLength1 + outputEdgeLength2;
    let totalDuration = hasInitialized ? 500 : 800;
    let opacityDuration = hasInitialized ? 400 : 600;

    outputEdge1.attr('stroke-dasharray', outputEdgeLength1 + ' ' + outputEdgeLength1)
      .attr('stroke-dashoffset', outputEdgeLength1);

    outputEdge2.attr('stroke-dasharray', outputEdgeLength2 + ' ' + outputEdgeLength2)
      .attr('stroke-dashoffset', outputEdgeLength2);

    outputEdge1.transition('softmax-output-edge')
      .duration(outputEdgeLength1 / totalLength * totalDuration)
      .attr('stroke-dashoffset', 0);

    outputEdge2.transition('softmax-output-edge')
      .delay(outputEdgeLength1 / totalLength * totalDuration)
      .duration(outputEdgeLength2 / totalLength * totalDuration)
      .attr('stroke-dashoffset', 0);

    curNodeGroup.append('circle')
      .attr('class', 'logit-circle')
      .attr('id', `logit-circle-${curI}`)
      .attr('cx', centerX)
      .attr('cy', nodeCoordinate[outputLayerIndex][curI].y + nodeLength / 2)
      .attr('r', 7)
      .style('fill', layerColorScales.logit(logitColorScale(logits[curI])))
      .style('stroke', intermediateColor)
      .style('cursor', 'crosshair')
      .on('mouseover', () => logitCircleMouseOverHandler(curI))
      .on('mouseleave', () => logitCircleMouseLeaveHandler(curI))
      .on('click', () => { d3.event.stopPropagation() });

    // Show the element in the detailed view
    softmaxDetailViewInfo.startAnimation = {
      i: curI,
      duration: opacityDuration,
      // Always show the animation
      hasInitialized: false
    };
    softmaxDetailViewStore.set(softmaxDetailViewInfo);

    // Show the elements with animation    
    curNodeGroup.transition('softmax-edge')
      .duration(opacityDuration)
      .style('opacity', 1);

    if ((selectedI < 3 && curI == 9) || (selectedI >= 3 && curI == 0)) {
      // Show the hover text
      softmaxDetailAnnotation.select('.softmax-detail-hover-annotation')
        .transition('softmax-edge')
        .duration(opacityDuration)
        .style('opacity', 1);
    }

    softmaxDetailAnnotation.select(`#logit-text-${curI}`)
      .transition('softmax-edge')
      .duration(opacityDuration)
      .style('opacity', 1);

    curEdgeGroup.transition('softmax-edge')
      .duration(opacityDuration)
      .style('opacity', 1)
      .on('end', () => {
        // Recursive animaiton
        curIIndex++;
        if (curIIndex < underneathIs.length) {
          linkData = [];
          drawOneEdgeGroup();
        } else {
          hasInitialized = true;
          softmaxDetailViewInfo.hasInitialized = true;
          softmaxDetailViewStore.set(softmaxDetailViewInfo);
        }
      });

    symbolClone.transition('softmax-edge')
      .duration(opacityDuration)
      .style('opacity', 0.2);
  }

  // Show the softmax detail view
  let anchorElement = svg.select('.intermediate-layer')
    .select('.layer-label').node();
  let pos = getMidCoords(svg, anchorElement);
  let wholeSvg = d3.select('#cnn-svg');
  let svgYMid = +wholeSvg.style('height').replace('px', '') / 2;
  let detailViewTop = svgYMid - 192 / 2;

  const detailview = document.getElementById('detailview');
  positionFloatingDetailView(detailview, detailViewTop, Math.max(24, pos.left - 490 - 50));

  denseDetailViewStore.set({ show: false });
  softmaxDetailViewStore.set({
    show: true,
    logits: logits,
    logitColors: logits.map(d => layerColorScales.logit(logitColorScale(d))),
    selectedI: selectedI,
    highlightI: -1,
    outputName: classList[selectedI],
    outputValue: outputLayerData[selectedI].output,
    startAnimation: { i: -1, duration: 0, hasInitialized: hasInitialized }
  })

  drawOneEdgeGroup();

  // Draw logit circle color scale
  drawIntermediateLayerLegend({
    legendHeight: 5,
    curLayerIndex: curLayerIndex,
    range: logitExtent[1] - logitExtent[0],
    minMax: { min: logitExtent[0], max: logitExtent[1] },
    group: logitLayer,
    width: softmaxX - (plusColumnX + plusSymbolRadius * 2 + 5),
    gradientAppendingName: 'flatten-logit-gradient',
    gradientGap: 0.1,
    colorScale: layerColorScales.logit,
    x: plusColumnX + plusSymbolRadius * 2 + 5,
    y: svgPaddings.top + vSpaceAroundGap * (10) + vSpaceAroundGap +
      nodeLength * 10
  });

  // Draw logit layer label
  let logitLabel = logitLayer.append('g')
    .attr('class', 'layer-label')
    .classed('hidden', detailedMode)
    .attr('transform', () => {
      let x = centerX;
      let y = (svgPaddings.top + vSpaceAroundGap) / 2 + 5;
      return `translate(${x}, ${y})`;
    });

  logitLabel.append('text')
    .style('text-anchor', 'middle')
    .style('dominant-baseline', 'middle')
    .style('opacity', 0.8)
    .style('font-weight', 800)
    .text('logit');
}

const removeLogitLayer = () => {
  svg.select('.logit-layer').remove();
  svg.select('.logit-layer-lower').remove();
  svg.selectAll('.plus-symbol-clone').remove();

  // Instead of removing the paths, we hide them, so it is faster to load in
  // the future
  svg.select('.underneath')
    .selectAll('.logit-lower')
    .style('opacity', 0);

  softmaxDetailViewStore.set({
    show: false,
    logits: []
  })
}

const softmaxClicked = (arg) => {
  if (isSoftmaxTransitioning) {
    d3.event.stopPropagation();
    return;
  }

  isSoftmaxTransitioning = true;
  let curLayerIndex = arg.curLayerIndex,
    outputLayerIndex = arg.outputLayerIndex,
    prevFeatureLayerIndex = arg.prevFeatureLayerIndex ?? curLayerIndex - 1,
    moveX = arg.moveX,
    symbolX = arg.symbolX,
    symbolY = arg.symbolY,
    outputX = arg.outputX,
    outputY = arg.outputY,
    softmaxLeftMid = arg.softmaxLeftMid,
    selectedI = arg.selectedI,
    intermediateX1 = arg.intermediateX1,
    intermediateX2 = arg.intermediateX2,
    pixelWidth = arg.pixelWidth,
    pixelHeight = arg.pixelHeight,
    topY = arg.topY,
    bottomY = arg.bottomY,
    middleGap = arg.middleGap,
    middleRectHeight = arg.middleRectHeight,
    softmaxX = arg.softmaxX,
    softmaxTextY = arg.softmaxTextY,
    softmaxWidth = arg.softmaxWidth,
    symbolGroup = arg.symbolGroup,
    flattenRange = arg.flattenRange,
    logitSpacingFactor = arg.logitSpacingFactor ?? 0.8,
    plusColumnX = arg.plusColumnX,
    showSelectedPlusClone = arg.showSelectedPlusClone,
    moveSymbolToPlusColumn = arg.moveSymbolToPlusColumn ?? false,
    hiddenLayout = arg.hiddenLayout,
    onReturnToOverview = arg.onReturnToOverview;
  let { output: outputLayerData } = getClassifierHead();

  let duration = 600;
  let centerX = softmaxLeftMid - moveX * logitSpacingFactor;
  let shouldReturnToOverview = isInSoftmax;
  d3.event.stopPropagation();

  // Clean up the logit elemends before moving anything
  if (isInSoftmax) {
    allowsSoftmaxAnimationStore.set(false);
    removeLogitLayer();
  } else {
    allowsSoftmaxAnimationStore.set(true);
  }

  // Move the overlay gradient
  svg.select('.intermediate-layer-overlay')
    .select('rect.overlay')
    .transition('softmax')
    .ease(d3.easeCubicInOut)
    .duration(duration)
    .attr('transform', `translate(${isInSoftmax ? 0 : -moveX}, ${0})`);

  let classifierHeadLinkGen = d3.linkHorizontal()
    .x(d => d.x)
    .y(d => d.y);

  svg.selectAll('.classifier-head-flatten-link')
    .attr('data-softmax-source-shift', isInSoftmax ? null : -moveX)
    .transition('softmax')
    .ease(d3.easeCubicInOut)
    .duration(duration)
    .attr('d', (d, i, g) => {
      let edge = d3.select(g[i]);
      let sourceX = +edge.attr('data-source-x');
      let sourceY = +edge.attr('data-source-y');
      let targetX = +edge.attr('data-target-x');
      let targetY = +edge.attr('data-target-y');
      return classifierHeadLinkGen({
        source: {
          x: isInSoftmax ? sourceX : sourceX - moveX,
          y: sourceY,
        },
        target: {
          x: targetX,
          y: targetY,
        },
      });
    });

  // Move the legends
  svg.selectAll(`.intermediate-legend-${prevFeatureLayerIndex}`)
    .each((d, i, g) => moveLegend(d, i, g, moveX, duration, isInSoftmax));

  svg.select('.intermediate-layer')
    .select(`.layer-label`)
    .each((d, i, g) => moveLegend(d, i, g, moveX, duration, isInSoftmax));

  svg.select('.intermediate-layer')
    .select(`.layer-detailed-label`)
    .each((d, i, g) => moveLegend(d, i, g, moveX, duration, isInSoftmax));

  // Also move all layers on the left
  for (let i = curLayerIndex - 1; i >= 0; i--) {
    let curLayer = svg.select(`g#cnn-layer-group-${i}`);
    let previousX = +curLayer.select('image').attr('x');
    let newX = isInSoftmax ? previousX + moveX : previousX - moveX;
    moveLayerX({
      layerIndex: i,
      targetX: newX,
      disable: true,
      delay: 0,
      transitionName: 'softmax',
      duration: duration
    });
  }

  // Hide the sum up annotation
  svg.select('.plus-annotation')
    .transition('softmax')
    .duration(duration)
    .style('opacity', isInSoftmax ? 1 : 0)
    .style('pointer-events', isInSoftmax ? 'all' : 'none');

  if (moveSymbolToPlusColumn && plusColumnX !== undefined) {
    symbolGroup.transition('softmax')
      .duration(duration)
      .ease(d3.easeCubicInOut)
      .attr('transform', `translate(${isInSoftmax ? symbolX : plusColumnX}, ${symbolY})`);

    svg.select('.bias-annotation')
      .transition('softmax')
      .duration(duration)
      .ease(d3.easeCubicInOut)
      .attr('x', (isInSoftmax ? symbolX : plusColumnX) - plusSymbolRadius - 8);

    svg.selectAll('[data-move-with-plus="true"]')
      .transition('softmax')
      .duration(duration)
      .ease(d3.easeCubicInOut)
      .style('opacity', (d, i, g) => {
        let edge = d3.select(g[i]);
        if (edge.classed('symbol-softmax')) {
          return isInSoftmax ? 1 : 0;
        }
        return edge.attr('data-original-opacity') || edge.style('opacity') || 1;
      })
      .attr('d', (d, i, g) => {
        let edge = d3.select(g[i]);
        let sourceX = +edge.attr('data-source-x');
        let sourceY = +edge.attr('data-source-y');
        let targetX = +edge.attr('data-target-x');
        let targetY = +edge.attr('data-target-y');
        let xDelta = (isInSoftmax ? symbolX : plusColumnX) - symbolX;
        let movingGroupCompensation = isInSoftmax ? 0 : moveX;

        if (edge.classed('dense-output')) {
          sourceX += movingGroupCompensation;
          targetX += xDelta + movingGroupCompensation;
        } else if (edge.classed('symbol-softmax')) {
          sourceX += xDelta + movingGroupCompensation;
          targetX += movingGroupCompensation;
        }

        return classifierHeadLinkGen({
          source: { x: sourceX, y: sourceY },
          target: { x: targetX, y: targetY }
        });
      });
  }

  // Hide the softmax annotation
  let softmaxAnnotation = svg.select('.softmax-annotation')
    .style('pointer-events', isInSoftmax ? 'all' : 'none');

  let softmaxDetailAnnotation = softmaxAnnotation.selectAll('.softmax-detail-annoataion')
    .data([0])
    .enter()
    .append('g')
    .attr('class', 'softmax-detail-annoataion');

  // Remove the detailed annoatioan when quitting the detail view
  if (isInSoftmax) {
    softmaxAnnotation.selectAll('.softmax-detail-annoataion').remove();
  }

  softmaxAnnotation.select('.arrow-group')
    .transition('softmax')
    .duration(duration)
    .style('opacity', isInSoftmax ? 1 : 0);

  softmaxAnnotation.select('.annotation-text')
    .style('cursor', 'help')
    .style('pointer-events', 'all')
    .on('click', () => {
      d3.event.stopPropagation();
      // Scroll to the article element
      document.querySelector(`#article-softmax`).scrollIntoView({
        behavior: 'smooth'
      });
    })
    .transition('softmax')
    .duration(duration)
    .style('opacity', isInSoftmax ? 1 : 0)
    .on('end', () => {
      if (!isInSoftmax) {
        // Add new annotation for the softmax button
        let textX = softmaxX + softmaxWidth / 2;
        let textY = softmaxTextY - 10;

        if (selectedI === 0) {
          textY = softmaxTextY + 70;
        }

        let text = softmaxDetailAnnotation.append('text')
          .attr('x', textX)
          .attr('y', textY)
          .attr('class', 'annotation-text softmax-detail-text')
          .style('dominant-baseline', 'baseline')
          .style('text-anchor', 'middle')
          .text('Normalize ');

        text.append('tspan')
          .attr('dx', 1)
          .style('fill', '#E56014')
          .text('logits');

        text.append('tspan')
          .attr('dx', 1)
          .text(' into');

        text.append('tspan')
          .attr('x', textX)
          .attr('dy', '1.1em')
          .text('class probabilities');

        if (selectedI === 0) {
          drawArrow({
            group: softmaxDetailAnnotation,
            sx: softmaxX + softmaxWidth / 2 - 5,
            sy: softmaxTextY + 44,
            tx: softmaxX + softmaxWidth / 2,
            ty: textY - 12,
            dr: 50,
            hFlip: true,
            marker: 'marker-alt'
          });
        } else {
          drawArrow({
            group: softmaxDetailAnnotation,
            sx: softmaxX + softmaxWidth / 2 - 5,
            sy: softmaxTextY + 4,
            tx: softmaxX + softmaxWidth / 2,
            ty: symbolY - plusSymbolRadius - 4,
            dr: 50,
            hFlip: true,
            marker: 'marker-alt'
          });
        }

        // Add annotation for the logit layer label
        textX = centerX + 45;
        textY = (svgPaddings.top + vSpaceAroundGap) / 2 + 5;
        let arrowTX = centerX + 20;
        let arrowTY = (svgPaddings.top + vSpaceAroundGap) / 2 + 5;

        softmaxDetailAnnotation.append('g')
          .attr('class', 'layer-detailed-label')
          .attr('transform', () => {
            let x = centerX;
            let y = (svgPaddings.top + vSpaceAroundGap) / 2 - 5;
            return `translate(${x}, ${y})`;
          })
          .classed('hidden', !detailedMode)
          .append('text')
          // .attr('x', centerX)
          // .attr('y',  (svgPaddings.top + vSpaceAroundGap) / 2 - 6)
          .style('opacity', 0.7)
          .style('dominant-baseline', 'middle')
          .style('font-size', '12px')
          .style('font-weight', '800')
          .append('tspan')
          .attr('x', 0)
          .text('logit')
          .append('tspan')
          .attr('x', 0)
          .style('font-size', '8px')
          .style('font-weight', 'normal')
          .attr('dy', '1.5em')
          .text('(10)');

        softmaxDetailAnnotation.append('text')
          .attr('class', 'annotation-text')
          .attr('x', textX)
          .attr('y', (svgPaddings.top + vSpaceAroundGap) / 2 + 3)
          .style('text-anchor', 'start')
          .text('Before')
          .append('tspan')
          .attr('x', textX)
          .attr('dy', '1em')
          .text('normalization')


        drawArrow({
          group: softmaxDetailAnnotation,
          tx: arrowTX,
          ty: arrowTY,
          sx: textX - 6,
          sy: textY + 2,
          dr: 60,
          hFlip: false,
          marker: 'marker-alt'
        });

        softmaxDetailAnnotation.append('text')
          .attr('class', 'annotation-text')
          .attr('x', nodeCoordinate[outputLayerIndex][0].x - 35)
          .attr('y', (svgPaddings.top + vSpaceAroundGap) / 2 + 3)
          .style('text-anchor', 'end')
          .text('After')
          .append('tspan')
          .attr('x', nodeCoordinate[outputLayerIndex][0].x - 35)
          .attr('dy', '1em')
          .text('normalization')

        drawArrow({
          group: softmaxDetailAnnotation,
          tx: nodeCoordinate[outputLayerIndex][0].x - 8,
          ty: arrowTY,
          sx: nodeCoordinate[outputLayerIndex][0].x - 27,
          sy: textY + 2,
          dr: 60,
          hFlip: true,
          marker: 'marker-alt'
        });

        for (let i = 0; i < outputLayerData.length; i++) {
          softmaxDetailAnnotation.append('text')
            .attr('x', centerX)
            .attr('y', nodeCoordinate[outputLayerIndex][i].y + nodeLength / 2 + 8)
            .attr('class', 'annotation-text softmax-detail-text')
            .attr('id', `logit-text-${i}`)
            .style('text-anchor', 'middle')
            .style('dominant-baseline', 'hanging')
            .style('opacity', i === selectedI ? 1 : 0)
            .text(`${classList[i]}`);
        }

        let hoverTextGroup = softmaxDetailAnnotation.append('g')
          .attr('class', 'softmax-detail-hover-annotation')
          .style('opacity', 0);

        textX = centerX + 50;
        textY = nodeCoordinate[outputLayerIndex][0].y + nodeLength / 2;

        if (selectedI < 3) {
          textY = nodeCoordinate[outputLayerIndex][outputLayerData.length - 1].y + nodeLength / 2;
        }

        // Add annotation to prompt user to check the logit value
        let hoverText = hoverTextGroup.append('text')
          .attr('x', textX)
          .attr('y', textY)
          .attr('class', 'annotation-text softmax-detail-text softmax-hover-text')
          .style('text-anchor', 'start')
          .style('dominant-baseline', 'baseline')
          .append('tspan')
          .style('font-weight', 700)
          .style('dominant-baseline', 'baseline')
          .text(`Hover over `)
          .append('tspan')
          .style('font-weight', 400)
          .style('dominant-baseline', 'baseline')
          .text('to see');

        hoverText.append('tspan')
          .style('dominant-baseline', 'baseline')
          .attr('x', textX)
          .attr('dy', '1em')
          .text('its ');

        hoverText.append('tspan')
          .style('dominant-baseline', 'baseline')
          .attr('dx', 1)
          .style('fill', '#E56014')
          .text('logit');

        hoverText.append('tspan')
          .style('dominant-baseline', 'baseline')
          .attr('dx', 1)
          .text(' value');

        drawArrow({
          group: hoverTextGroup,
          tx: centerX + 15,
          ty: textY,
          sx: textX - 8,
          sy: textY + 2,
          dr: 60,
          hFlip: false
        });
      }
    })

  // Hide the annotation
  svg.select('.flatten-annotation')
    .transition('softmax')
    .duration(duration)
    .style('opacity', isInSoftmax ? 1 : 0)
    .style('pointer-events', isInSoftmax ? 'all' : 'none');

  // Move the left part of faltten layer elements
  let flattenLeftPart = svg.select('.flatten-layer-left');
  flattenLeftPart.transition('softmax')
    .duration(duration)
    .ease(d3.easeCubicInOut)
    .attr('transform', `translate(${isInSoftmax ? 0 : -moveX}, ${0})`)
    .on('end', () => {
      // Add the logit layer
      if (!isInSoftmax) {
        let logitArg = {
          curLayerIndex: curLayerIndex,
          outputLayerIndex: outputLayerIndex,
          prevFeatureLayerIndex: prevFeatureLayerIndex,
          moveX: moveX,
          softmaxLeftMid: softmaxLeftMid,
          selectedI: selectedI,
          intermediateX1: intermediateX1,
          intermediateX2: intermediateX2,
          pixelWidth: pixelWidth,
          pixelHeight: pixelHeight,
          topY: topY,
          bottomY: bottomY,
          middleGap: middleGap,
          middleRectHeight: middleRectHeight,
          softmaxX: softmaxX,
          symbolGroup: symbolGroup,
          symbolX: symbolX,
          flattenRange: flattenRange,
          logitSpacingFactor: logitSpacingFactor,
          plusColumnX: plusColumnX,
          showSelectedPlusClone: showSelectedPlusClone,
          hiddenLayout: hiddenLayout
        };
        drawLogitLayer(logitArg);
      }

      // Redraw the line from the plus symbol to the output node
      if (!isInSoftmax) {
        let newLine = flattenLeftPart.select('.edge-group')
          .append('line')
          .attr('class', 'symbol-output-line')
          .attr('x1', moveSymbolToPlusColumn && plusColumnX !== undefined ?
            plusColumnX : symbolX)
          .attr('y1', symbolY)
          .attr('x2', outputX + moveX)
          .attr('y2', outputY)
          .style('stroke-width', 1.2)
          .style('stroke', '#E5E5E5')
          .style('opacity', 0);

        newLine.transition('softmax')
          .delay(duration / 3)
          .duration(duration * 2 / 3)
          .style('opacity', 1);
      } else {
        flattenLeftPart.select('.symbol-output-line').remove();
      }

      isInSoftmax = !isInSoftmax;
      isInSoftmaxStore.set(isInSoftmax);
      isSoftmaxTransitioning = false;
      if (denseDetailSelections.length) {
        redrawDenseDetailSelections();
      }
    })
}

/**
 * Draw the flatten layer before output layer
 * @param {number} curLayerIndex Index of the selected layer
 * @param {object} d Bounded d3 data
 * @param {number} i Index of the selected node
 * @param {number} width CNN group width
 * @param {number} height CNN group height
 */
export const drawFlatten = (curLayerIndex, d, i, width, height) => {
  isSoftmaxTransitioning = false;
  isInSoftmax = false;
  isInSoftmaxStore.set(false);
  allowsSoftmaxAnimationStore.set(false);
  softmaxDetailViewStore.set({
    show: false,
    logits: []
  });
  resetDenseNeuronDetailLayout();

  let classifierHead = getClassifierHead();
  let flattenLayerData = classifierHead.flatten;
  let flattenRawData = classifierHead.flattenRaw;
  let hiddenDenseLayers = classifierHead.denseLayers;
  let selectedOutputNode = classifierHead.output[i];
  const outputLayerIndex = curLayerIndex;
  const flattenSourceLayerName = flattenRawData[0]?.inputLinks[0]?.source?.layerName;
  const flattenSourceLayerIndex = cnn.findIndex((layer) =>
    layer[0]?.layerName === flattenSourceLayerName
  );
  const prevFeatureLayerIndex =
    flattenSourceLayerIndex >= 0 ? flattenSourceLayerIndex : curLayerIndex - 1;

  // Show the output legend
  svg.selectAll('.output-legend')
    .classed('hidden', false);

  let pixelWidth = nodeLength / 2;
  let pixelHeight = 1.1;
  let intermediateGap = Math.min((hSpaceAroundGap * gapRatio * 4) / 2, 390);
  let hiddenLayerWidth = hiddenDenseLayers.length ? 26 : 0;
  let hiddenLayerGap = 46;
  let outputLabelGutter = hiddenDenseLayers.length ? 165 : hiddenLayerGap;
  let originalFlattenLength = 2 * nodeLength +
    2.75 * intermediateGap + pixelWidth;
  let totalLength = originalFlattenLength;
  let sourceLayerMinX = svgPaddings.left + 170;
  let leftX = Math.max(
    nodeCoordinate[curLayerIndex][0].x - totalLength,
    sourceLayerMinX
  );
  const minimumGap = 20;
  let linkGen = d3.linkHorizontal()
    .x(d => d.x)
    .y(d => d.y);

  // Hide the edges
  svg.select('g.edge-group')
    .style('visibility', 'hidden');

  svg.select('.input-annotation')
    .style('opacity', 0);

  // Move the layer that actually feeds flatten.
  moveLayerX({
    layerIndex: prevFeatureLayerIndex, targetX: leftX,
    disable: true, delay: 0
  });

  // Disable the current layer (output layer)
  moveLayerX({
    layerIndex: curLayerIndex,
    targetX: nodeCoordinate[curLayerIndex][0].x, disable: true,
    delay: 0, opacity: 0.15, specialIndex: i
  });

  // Compute the gap in the left shrink region
  let leftEnd = leftX - hSpaceAroundGap;
  // Task 3 at lines 1035-1036: pack the left-side layers using the real count
  // of layers before output instead of the old fixed-depth assumption.
  let leftLayerCount = curLayerIndex - 2;
  let leftGap = (leftEnd - nodeCoordinate[0][0].x - leftLayerCount * nodeLength) / leftLayerCount;

  // Different from other intermediate view, we push the left part dynamically
  // 1. If there is enough space, we fix the first layer position and move all
  // other layers;
  // 2. If there is not enough space, we maintain the minimum gap and push all
  // left layers to the left (could be out-of-screen)
  if (leftGap > minimumGap) {
    // Move the left layers
    let packedIndex = 0;
    for (let i = 0; i < curLayerIndex; i++) {
      if (i === prevFeatureLayerIndex) {
        continue;
      }
      let curX = nodeCoordinate[0][0].x + i * (nodeLength + leftGap);
      curX = nodeCoordinate[0][0].x + packedIndex * (nodeLength + leftGap);
      moveLayerX({ layerIndex: i, targetX: curX, disable: true, delay: 0 });
      packedIndex++;
    }
  } else {
    leftGap = minimumGap;
    let curLeftBound = leftX - leftGap * 2 - nodeLength;
    // Move the left layers
    for (let i = curLayerIndex - 1; i >= 0; i--) {
      if (i === prevFeatureLayerIndex) {
        continue;
      }
      moveLayerX({ layerIndex: i, targetX: curLeftBound, disable: true, delay: 0 });
      curLeftBound = curLeftBound - leftGap - nodeLength;
    }
  }

  // Add an overlay
  let stops = [{ offset: '0%', color: 'rgb(250, 250, 250)', opacity: 1 },
  { offset: '50%', color: 'rgb(250, 250, 250)', opacity: 0.95 },
  { offset: '100%', color: 'rgb(250, 250, 250)', opacity: 0.85 }];
  addOverlayGradient('overlay-gradient-left', stops);

  let intermediateLayerOverlay = svg.append('g')
    .attr('class', 'intermediate-layer-overlay');

  intermediateLayerOverlay.append('rect')
    .attr('class', 'overlay')
    .style('fill', 'url(#overlay-gradient-left)')
    .style('stroke', 'none')
    .attr('width', leftX + svgPaddings.left - (leftGap * 2) + 3)
    .attr('height', height + svgPaddings.top + svgPaddings.bottom)
    .attr('x', -svgPaddings.left)
    .attr('y', 0)
    .style('opacity', 0);

  intermediateLayerOverlay.selectAll('rect.overlay')
    .transition('move')
    .duration(800)
    .ease(d3.easeCubicInOut)
    .style('opacity', 1);

  // Add the intermediate layer
  let intermediateLayer = svg.append('g')
    .attr('class', 'intermediate-layer')
    .style('opacity', 0);

  let intermediateX1 = leftX + nodeLength + intermediateGap;
  let intermediateX2 = intermediateX1 + intermediateGap + pixelWidth;
  let range = cnnLayerRanges[selectedScaleLevel][prevFeatureLayerIndex];
  let colorScale = layerColorScales.conv;
  let prevFeatureLayerCount = cnn[prevFeatureLayerIndex].length;
  let firstChannelIndex = 0;
  let lastChannelIndex = prevFeatureLayerCount - 1;
  let middleChannelCount = Math.max(prevFeatureLayerCount - 2, 0);
  let flattenLength = flattenLayerData.length / prevFeatureLayerCount;
  let middleGap = 5;
  const originalVisibleFlattenCount = 169;
  const maxVisibleFlattenHeight = originalVisibleFlattenCount * pixelHeight;
  const layerStackHeight = prevFeatureLayerCount * nodeLength +
    (prevFeatureLayerCount - 1) * vSpaceAroundGap;
  const maxGeometryFlattenHeight = middleChannelCount > 0 ?
    (layerStackHeight - middleGap * (middleChannelCount + 1)) / 2 :
    maxVisibleFlattenHeight;
  pixelHeight = Math.min(
    pixelHeight,
    Math.max(1, Math.min(maxVisibleFlattenHeight, maxGeometryFlattenHeight)) /
      flattenLength
  );
  let linkData = [];

  let flattenLayer = intermediateLayer.append('g')
    .attr('class', 'flatten-layer');

  let flattenLayerLeftPart = flattenLayer.append('g')
    .attr('class', 'flatten-layer-left');

  let topY = nodeCoordinate[prevFeatureLayerIndex][0].y;
  let bottomY = nodeCoordinate[prevFeatureLayerIndex][lastChannelIndex].y + nodeLength -
    flattenLength * pixelHeight;

  // Compute the pre-layer gap
  let preLayerDimension = cnn[prevFeatureLayerIndex][0].output.length;
  let preLayerGap = nodeLength / (2 * preLayerDimension);

  // Compute bounding box length
  let boundingBoxLength = nodeLength / preLayerDimension;

  // Compute the weight color scale
  let displayedFlattenChannelIndexes = Array.from(
    new Set([firstChannelIndex, lastChannelIndex])
  );
  let displayedFlattenWeights = [];
  let displayedFlattenValues = [];
  displayedFlattenChannelIndexes.forEach(channelIndex => {
    displayedFlattenValues = displayedFlattenValues.concat(
      flattenLayerData.slice(channelIndex * flattenLength,
        (channelIndex + 1) * flattenLength)
        .map(d => d.output)
    );
    displayedFlattenWeights = displayedFlattenWeights.concat(
      flattenLayerData.slice(channelIndex * flattenLength,
        (channelIndex + 1) * flattenLength)
        .map(d => d.outputLinks[i].weight)
    );
  });
  let flattenExtent = d3.extent(displayedFlattenWeights);

  let flattenRange = 2 * (Math.round(
    Math.max(...flattenExtent.map(Math.abs)) * 1000) / 1000);

  let flattenMouseOverHandler = (d) => {
    let index = d.index;
    // Screenshot
    // console.log(index);

    // Update the hover info UI
    if (d.weight === undefined) {
      hoverInfo = {
        show: true,
        text: `Pixel value: ${formater(flattenFactoredFDict[index])}`
      };
    } else {
      hoverInfo = {
        show: true,
        text: `Weight: ${formater(d.weight)}`
      };
    }
    hoverInfoStore.set(hoverInfo);

    flattenLayerLeftPart.select(`#edge-flatten-${index}`)
      .raise()
      .style('stroke', intermediateColor)
      .style('stroke-width', 1);

    flattenLayerLeftPart.select(`#edge-flatten-${index}-output`)
      .raise()
      .style('stroke-width', 1)
      .style('stroke', da => gappedColorScale(layerColorScales.weight,
        flattenRange, da.weight, 0.1));

    flattenLayerLeftPart.select(`#bounding-${index}`)
      .raise()
      .style('opacity', 1);
  }

  let flattenMouseLeaveHandler = (d) => {
    let index = d.index;

    // screenshot
    // if (index === 32) {return;}

    // Update the hover info UI
    if (d.weight === undefined) {
      hoverInfo = {
        show: false,
        text: `Pixel value: ${formater(flattenFactoredFDict[index])}`
      };
    } else {
      hoverInfo = {
        show: false,
        text: `Weight: ${formater(d.weight)}`
      };
    }
    hoverInfoStore.set(hoverInfo);

    flattenLayerLeftPart.select(`#edge-flatten-${index}`)
      .style('stroke-width', 0.6)
      .style('stroke', '#E5E5E5')

    flattenLayerLeftPart.select(`#edge-flatten-${index}-output`)
      .style('stroke-width', 0.6)
      .style('stroke', da => gappedColorScale(layerColorScales.weight,
        flattenRange, da.weight, 0.35));

    flattenLayerLeftPart.select(`#bounding-${index}`)
      .raise()
      .style('opacity', 0);
  }

  flattenFactoredFDict = {};
  for (let f = 0; f < flattenLength; f++) {
    let loopFactors = [firstChannelIndex, lastChannelIndex];
    loopFactors.forEach(l => {
      let factoredF = f + l * flattenLength;
      flattenFactoredFDict[factoredF] = flattenLayerData[factoredF].output;
      flattenLayerLeftPart.append('rect')
        .attr('x', intermediateX1)
        .attr('y', l === 0 ? topY + f * pixelHeight : bottomY + f * pixelHeight)
        .attr('width', pixelWidth)
        .attr('height', pixelHeight)
        .style('cursor', 'crosshair')
        .style('fill', colorScale((flattenLayerData[factoredF].output + range / 2) / range))
        .on('mouseover', () => flattenMouseOverHandler({ index: factoredF }))
        .on('mouseleave', () => flattenMouseLeaveHandler({ index: factoredF }))
        .on('click', () => { d3.event.stopPropagation() });

      if (!hiddenDenseLayers.length) {
        // Flatten -> output
        linkData.push({
          source: {
            x: intermediateX1 + pixelWidth + 3,
            y: l === 0 ? topY + f * pixelHeight : bottomY + f * pixelHeight
          },
          target: {
            x: intermediateX2,
            y: nodeCoordinate[curLayerIndex][i].y + nodeLength / 2
          },
          index: factoredF,
          weight: flattenLayerData[factoredF].outputLinks[i].weight,
          name: `flatten-${factoredF}-output`,
          color: gappedColorScale(layerColorScales.weight,
            flattenRange, flattenLayerData[factoredF].outputLinks[i].weight, 0.35),
          width: 0.6,
          opacity: 1,
          class: `flatten-output`
        });
      }

      // Pre-layer -> flatten
      let row = Math.floor(f / preLayerDimension);
      linkData.push({
        target: {
          x: intermediateX1 - 3,
          y: l === 0 ? topY + f * pixelHeight : bottomY + f * pixelHeight
        },
        source: {
          x: leftX + nodeLength + 3,
          y: nodeCoordinate[prevFeatureLayerIndex][l].y + (2 * row + 1) * preLayerGap
        },
        index: factoredF,
        name: `flatten-${factoredF}`,
        color: '#E5E5E5',
        // color: gappedColorScale(layerColorScales.conv,
        //   2 * Math.max(Math.abs(cnnLayerMinMax[10].max), Math.abs(cnnLayerMinMax[10].min)),
        //   cnn.flatten[factoredF].output, 0.2),
        width: 0.6,
        opacity: 1,
        class: `flatten`
      });

      // Add original pixel bounding box
      let loc = flattenRawData[factoredF].inputLinks[0].weight;
      flattenLayerLeftPart.append('rect')
        .attr('id', `bounding-${factoredF}`)
        .attr('class', 'flatten-bounding')
        .attr('x', leftX + loc[1] * boundingBoxLength)
        .attr('y', nodeCoordinate[prevFeatureLayerIndex][l].y + loc[0] * boundingBoxLength)
        .attr('width', boundingBoxLength)
        .attr('height', boundingBoxLength)
        .style('fill', 'none')
        .style('stroke', intermediateColor)
        .style('stroke-length', '0.5')
        .style('pointer-events', 'all')
        .style('cursor', 'crosshair')
        .style('opacity', 0)
        .on('mouseover', () => flattenMouseOverHandler({ index: factoredF }))
        .on('mouseleave', () => flattenMouseLeaveHandler({ index: factoredF }))
        .on('click', () => { d3.event.stopPropagation() });
    })
  }

  // Use abstract symbol to represent the flatten nodes in between (between
  // the first and the last nodes)
  // Compute the average value of input node and weights
  let meanValues = [];
  for (let n = 1; n < prevFeatureLayerCount - 1; n++) {
    /*
    let meanOutput = d3.mean(flattenLayerData.slice(flattenLength * n,
      flattenLength * (n + 1)).map(d => d.output));
    let meanWeight= d3.mean(flattenLayerData.slice(flattenLength * n,
      flattenLength * (n + 1)).map(d => d.outputLinks[i].weight));
    meanValues.push({index: n, output: meanOutput, weight: meanWeight});
    */
    meanValues.push({ index: n });
  }

  // Compute the middle gap
  let middleRectHeight = middleChannelCount > 0 ?
    (prevFeatureLayerCount * nodeLength + (prevFeatureLayerCount - 1) * vSpaceAroundGap -
      pixelHeight * flattenLength * 2 - middleGap * (middleChannelCount + 1)) /
    middleChannelCount : 0;

  // Add middle nodes
  meanValues.forEach((v, vi) => {
    // Add a small rectangle
    flattenLayerLeftPart.append('rect')
      .attr('x', intermediateX1 + pixelWidth / 4)
      .attr('y', topY + flattenLength * pixelHeight + middleGap * (vi + 1) +
        middleRectHeight * vi)
      .attr('width', pixelWidth / 2)
      .attr('height', middleRectHeight)
      // .style('fill', colorScale((v.output + range / 2) / range));
      .style('fill', '#E5E5E5');

    // Add a triangle next to the input node
    flattenLayerLeftPart.append('polyline')
      .attr('points',
        `${leftX + nodeLength + 3}
        ${nodeCoordinate[prevFeatureLayerIndex][v.index].y},
        ${leftX + nodeLength + 10}
        ${nodeCoordinate[prevFeatureLayerIndex][v.index].y + nodeLength / 2},
        ${leftX + nodeLength + 3}
        ${nodeCoordinate[prevFeatureLayerIndex][v.index].y + nodeLength}`)
      .style('fill', '#E5E5E5')
      .style('opacity', 1);

    // Input -> flatten
    linkData.push({
      source: {
        x: leftX + nodeLength + 10,
        y: nodeCoordinate[prevFeatureLayerIndex][v.index].y + nodeLength / 2
      },
      target: {
        x: intermediateX1 - 3,
        y: topY + flattenLength * pixelHeight + middleGap * (vi + 1) +
          middleRectHeight * (vi + 0.5)
      },
      index: -1,
      width: 1,
      opacity: 1,
      name: `flatten-abstract-${v.index}`,
      color: '#E5E5E5',
      class: `flatten-abstract`
    });

    if (!hiddenDenseLayers.length) {
      // Flatten -> output
      linkData.push({
        source: {
          x: intermediateX1 + pixelWidth + 3,
          y: topY + flattenLength * pixelHeight + middleGap * (vi + 1) +
            middleRectHeight * (vi + 0.5)
        },
        target: {
          x: intermediateX2,
          y: nodeCoordinate[curLayerIndex][i].y + nodeLength / 2
        },
        index: -1,
        name: `flatten-abstract-${v.index}-output`,
        color: '#E5E5E5',
        weight: v.weight,
        width: 1,
        opacity: 1,
        class: `flatten-abstract-output`
      });
    }
  })

  const getDisplayedFlattenAnchor = (flatIndex) => {
    let channelIndex = Math.floor(flatIndex / flattenLength);
    let localIndex = flatIndex % flattenLength;

    if (channelIndex === firstChannelIndex) {
      return {
        x: intermediateX1 + pixelWidth + 3,
        y: topY + localIndex * pixelHeight,
      };
    }

    if (channelIndex === lastChannelIndex) {
      return {
        x: intermediateX1 + pixelWidth + 3,
        y: bottomY + localIndex * pixelHeight,
      };
    }

    return undefined;
  };

  let hiddenLayout = undefined;
  let symbolEndX = intermediateX2 + plusSymbolRadius * 2;
  let denseAreaStartX = intermediateX1 + pixelWidth + 20;
  let denseAreaEndX = nodeCoordinate[curLayerIndex][i].x - outputLabelGutter;
  let denseAreaWidth = Math.max(denseAreaEndX - denseAreaStartX, 260);
  let hiddenLayerStartX = denseAreaStartX + Math.max(hiddenLayerGap, denseAreaWidth * 0.22);
  let hiddenLayerEndX = denseAreaStartX + denseAreaWidth * 0.58;

  hiddenLayout = drawHiddenDenseLayers({
    group: intermediateLayer,
    denseLayers: hiddenDenseLayers,
    startX: hiddenLayerStartX,
    endX: hiddenLayerEndX,
    anchorY: nodeCoordinate[curLayerIndex][i].y,
    outputTopY: nodeCoordinate[outputLayerIndex][0].y,
    outputBottomY:
      nodeCoordinate[outputLayerIndex][cnn[outputLayerIndex].length - 1].y +
      nodeLength,
    selectedOutputNode: selectedOutputNode,
    flattenAnchorForFlatIndex: getDisplayedFlattenAnchor,
    prevFeatureLayerIndex: prevFeatureLayerIndex,
  });

  let symbolX = intermediateX2 + plusSymbolRadius;
  let symbolY = nodeCoordinate[curLayerIndex][i].y + nodeLength / 2;

  if (hiddenLayout) {
    let lastDenseRightX = hiddenLayout.layerXs[hiddenLayout.layerXs.length - 1] +
      hiddenLayout.nodeSize;
    let outputInputX = getInputKnot({
      x: nodeCoordinate[curLayerIndex][i].x - 3,
      y: nodeCoordinate[curLayerIndex][i].y
    }).x;
    let hiddenSoftmaxWidth = 54;
    let hiddenOperationWidth = plusSymbolRadius * 2 + 20 + hiddenSoftmaxWidth;
    let hiddenOperationMidX = (lastDenseRightX + outputInputX) / 2;
    symbolX = hiddenOperationMidX - hiddenOperationWidth / 2 + plusSymbolRadius;
  }

  let symbolRectHeight = 1;
  let symbolGroup = intermediateLayer.append('g')
    .attr('class', 'plus-symbol')
    .attr('transform', `translate(${symbolX}, ${symbolY})`);

  symbolGroup.append('rect')
    .attr('x', -plusSymbolRadius)
    .attr('y', -plusSymbolRadius)
    .attr('width', plusSymbolRadius * 2)
    .attr('height', plusSymbolRadius * 2)
    .attr('rx', 3)
    .attr('ry', 3)
    .style('fill', 'none')
    .style('stroke', intermediateColor);

  symbolGroup.append('rect')
    .attr('x', -(plusSymbolRadius - 3))
    .attr('y', -symbolRectHeight / 2)
    .attr('width', 2 * (plusSymbolRadius - 3))
    .attr('height', symbolRectHeight)
    .style('fill', intermediateColor);

  symbolGroup.append('rect')
    .attr('x', -symbolRectHeight / 2)
    .attr('y', -(plusSymbolRadius - 3))
    .attr('width', symbolRectHeight)
    .attr('height', 2 * (plusSymbolRadius - 3))
    .style('fill', intermediateColor);

  symbolGroup.append('circle')
    .attr('cx', 0)
    .attr('cy', -nodeLength / 2 - 0.5 * kernelRectLength)
    .attr('r', kernelRectLength * 1.5)
    .style('stroke', intermediateColor)
    .style('cursor', 'crosshair')
    .style('fill', gappedColorScale(layerColorScales.weight,
      flattenRange, selectedOutputNode.bias, 0.35))
    .on('mouseover', () => {
      hoverInfoStore.set({ show: true, text: `Bias: ${formater(selectedOutputNode.bias)}` });
    })
    .on('mouseleave', () => {
      hoverInfoStore.set({ show: false, text: `Bias: ${formater(selectedOutputNode.bias)}` });
    })
    .on('click', () => { d3.event.stopPropagation(); });

  symbolGroup.append('path')
    .attr('d', linkGen({
      source: { x: 0, y: 0 },
      target: { x: 0, y: -nodeLength / 2 - 0.5 * kernelRectLength }
    }))
    .attr('id', 'bias-plus')
    .attr('stroke-width', 1.2)
    .attr('stroke', '#E5E5E5')
    .lower();

  if (!hiddenDenseLayers.length) {
    linkData.push({
      source: getOutputKnot({
        x: intermediateX2 + 2 * plusSymbolRadius - nodeLength,
        y: nodeCoordinate[curLayerIndex][i].y
      }),
      target: getInputKnot({
        x: nodeCoordinate[curLayerIndex][i].x - 3,
        y: nodeCoordinate[curLayerIndex][i].y
      }),
      name: `symbol-output`,
      width: 1.2,
      color: '#E5E5E5'
    });
  } else {
    let lastDenseLayerIndex = hiddenLayout.orderedLayers.length - 1;
    selectedOutputNode.inputLinks.forEach((link) => {
      let sourceY = hiddenLayout.getNodeCenterY(lastDenseLayerIndex, link.source.index);
      if (sourceY === undefined) {
        return;
      }

      linkData.push({
        source: {
          x: hiddenLayout.layerXs[lastDenseLayerIndex] + hiddenLayout.nodeSize,
          y: sourceY,
        },
        target: {
          x: symbolX - plusSymbolRadius,
          y: symbolY,
        },
        name: `dense-output-${link.source.index}`,
        width: edgeStrokeWidth,
        color: edgeInitColor,
        opacity: 0.28,
        class:
          `dense-output classifier-head-node-link-${lastDenseLayerIndex}-${link.source.index}`,
        sourceLayer: lastDenseLayerIndex,
        sourceIndex: link.source.index,
        targetIndex: link.dest.index,
        moveWithPlus: true,
      });
    });

    let softmaxWidth = 54;
    let softmaxX = symbolX + plusSymbolRadius + 20;
    let softmaxLeftMid = symbolX + plusSymbolRadius +
      (softmaxX - (symbolX + plusSymbolRadius)) / 2;
    let softmaxMoveX = Math.max(24, Math.min(90,
      (softmaxX - (intermediateX2 + pixelWidth + 3)) / 3));
    let hiddenLogitCenterX = softmaxLeftMid - softmaxMoveX * 1.75;
    let hiddenPlusColumnX = hiddenLogitCenterX -
      Math.max(52, Math.min(76, softmaxX - hiddenLogitCenterX));
    let softmaxArg = {
      curLayerIndex: curLayerIndex,
      outputLayerIndex: outputLayerIndex,
      prevFeatureLayerIndex: prevFeatureLayerIndex,
      moveX: softmaxMoveX,
      symbolX: symbolX,
      symbolY: symbolY,
      outputX: nodeCoordinate[curLayerIndex][i].x,
      outputY: symbolY,
      softmaxLeftMid: softmaxLeftMid,
      selectedI: i,
      intermediateX1: intermediateX1,
      intermediateX2: intermediateX2,
      pixelWidth: pixelWidth,
      pixelHeight: pixelHeight,
      topY: topY,
      bottomY: bottomY,
      middleGap: middleGap,
      middleRectHeight: middleRectHeight,
      softmaxX: softmaxX,
      softmaxWidth: softmaxWidth,
      softmaxTextY: nodeCoordinate[curLayerIndex][i].y - 2 * kernelRectLength - 6,
      symbolGroup: symbolGroup,
      flattenRange: flattenRange,
      logitSpacingFactor: 1.75,
      plusColumnX: hiddenPlusColumnX,
      showSelectedPlusClone: false,
      moveSymbolToPlusColumn: true,
      hiddenLayout: hiddenLayout,
      onReturnToOverview: () => {
        d3.select(`g#layer-${curLayerIndex}-node-${i}`).dispatch('click');
      }
    };
    let softmaxGroup = intermediateLayer.append('g')
      .attr('class', 'softmax-symbol')
      .attr('transform', `translate(${softmaxX}, ${symbolY})`)
      .style('pointer-events', 'all')
      .style('cursor', 'pointer')
      .on('click', () => softmaxClicked(softmaxArg));

    softmaxGroup.append('rect')
      .attr('x', 0)
      .attr('y', -plusSymbolRadius)
      .attr('width', softmaxWidth)
      .attr('height', plusSymbolRadius * 2)
      .attr('stroke', intermediateColor)
      .attr('rx', 2)
      .attr('ry', 2)
      .attr('fill', '#FAFAFA');

    softmaxGroup.append('text')
      .attr('x', 6)
      .attr('y', 1)
      .style('dominant-baseline', 'middle')
      .style('font-size', '11px')
      .style('opacity', 0.55)
      .text('softmax');

    linkData.push({
      source: {
        x: symbolX + plusSymbolRadius,
        y: symbolY,
      },
      target: {
        x: softmaxX,
        y: symbolY,
      },
      name: `symbol-softmax`,
      width: 1,
      color: '#94A3B8',
      opacity: 1,
      class: 'symbol-softmax',
      moveWithPlus: true,
    });

    linkData.push({
      source: {
        x: softmaxX + softmaxWidth,
        y: symbolY,
      },
      target: getInputKnot({
        x: nodeCoordinate[curLayerIndex][i].x - 3,
        y: nodeCoordinate[curLayerIndex][i].y
      }),
      name: `symbol-output`,
      width: 1.2,
      color: '#94A3B8',
      opacity: 1,
      class: 'symbol-output',
    });
  }

  let softmaxWidth = 55;
  let softmaxX = symbolX + plusSymbolRadius + 20;
  let softmaxLeftMid = symbolEndX;
  let softmaxTextY = nodeCoordinate[curLayerIndex][i].y - 2 * kernelRectLength - 6;

  if (!hiddenDenseLayers.length) {
    // Draw softmax operation symbol using the original flatten detail layout.
    let emptySpace = ((totalLength - 2 * nodeLength - 2 * intermediateGap)
      - softmaxWidth) / 2;
    softmaxX = emptySpace + symbolEndX;
    softmaxLeftMid = emptySpace / 2 + symbolEndX;
    let moveX = (intermediateX2 - (intermediateX1 + pixelWidth + 3)) * 2 / 3;

    let softmaxArg = {
      curLayerIndex: curLayerIndex,
      outputLayerIndex: outputLayerIndex,
      prevFeatureLayerIndex: prevFeatureLayerIndex,
      moveX: moveX,
      symbolX: symbolX,
      symbolY: symbolY,
      outputX: nodeCoordinate[curLayerIndex][i].x,
      outputY: symbolY,
      softmaxLeftMid: softmaxLeftMid,
      selectedI: i,
      intermediateX1: intermediateX1,
      intermediateX2: intermediateX2,
      pixelWidth: pixelWidth,
      pixelHeight: pixelHeight,
      topY: topY,
      bottomY: bottomY,
      middleGap: middleGap,
      middleRectHeight: middleRectHeight,
      softmaxX: softmaxX,
      softmaxWidth: softmaxWidth,
      softmaxTextY: softmaxTextY,
      symbolGroup: symbolGroup,
      flattenRange: flattenRange,
      onReturnToOverview: () => {
        d3.select(`g#layer-${curLayerIndex}-node-${i}`).dispatch('click');
      }
    };

    let softmaxSymbol = intermediateLayer.append('g')
      .attr('class', 'softmax-symbol')
      .attr('transform', `translate(${softmaxX}, ${symbolY})`)
      .style('pointer-events', 'all')
      .style('cursor', 'pointer')
      .on('click', () => softmaxClicked(softmaxArg));

    softmaxSymbol.append('rect')
      .attr('x', 0)
      .attr('y', -plusSymbolRadius)
      .attr('width', softmaxWidth)
      .attr('height', plusSymbolRadius * 2)
      .attr('stroke', intermediateColor)
      .attr('rx', 2)
      .attr('ry', 2)
      .attr('fill', '#FAFAFA');

    softmaxSymbol.append('text')
      .attr('x', 5)
      .attr('y', 1)
      .style('dominant-baseline', 'middle')
      .style('font-size', '12px')
      .style('opacity', 0.5)
      .text('softmax');
  }

  // Draw the layer label
  let layerLabel = intermediateLayer.append('g')
    .attr('class', 'layer-label')
    .classed('hidden', detailedMode)
    .attr('transform', () => {
      let x = leftX + nodeLength + intermediateGap + pixelWidth / 2;
      let y = (svgPaddings.top + vSpaceAroundGap) / 2 + 5;
      return `translate(${x}, ${y})`;
    })
    .style('cursor', 'help')
    .on('click', () => {
      d3.event.stopPropagation();
      // Scroll to the article element
      document.querySelector(`#article-flatten`).scrollIntoView({
        behavior: 'smooth'
      });
    });

  layerLabel.append('text')
    .style('dominant-baseline', 'middle')
    .style('opacity', 0.8)
    .style('font-weight', 800)
    .text(isStyleTestMode() ? 'Flatten' : 'FLATTEN');

  let svgHeight = Number(d3.select('#cnn-svg').style('height').replace('px', '')) + 150;
  let scroll = new SmoothScroll('a[href*="#"]', { offset: -svgHeight });

  let detailedLabelGroup = intermediateLayer.append('g')
    .attr('transform', () => {
      let x = leftX + nodeLength + intermediateGap + pixelWidth / 2;
      let y = (svgPaddings.top + vSpaceAroundGap) / 2 - 5;
      return `translate(${x}, ${y})`;
    })
    .attr('class', 'layer-detailed-label')
    .classed('hidden', !detailedMode)
    .style('cursor', 'help')
    .on('click', () => {
      d3.event.stopPropagation();
      // Scroll to the article element
      let anchor = document.querySelector(`#article-flatten`);
      scroll.animateScroll(anchor);
    });

  detailedLabelGroup.append('title')
    .text('Move to article section');

  let detailedLabelText = detailedLabelGroup.append('text')
    .style('text-anchor', 'middle')
    .style('dominant-baseline', 'middle')
    .style('opacity', '0.7')
    .style('font-weight', 800)
    .append('tspan')
    .text(isStyleTestMode() ? 'Flatten' : 'FLATTEN');

  let dimension = cnn[prevFeatureLayerIndex].length *
    cnn[prevFeatureLayerIndex][0].output.length *
    cnn[prevFeatureLayerIndex][0].output[0].length;

  detailedLabelText.append('tspan')
    .attr('x', 0)
    .attr('dy', '1.5em')
    .style('font-size', '8px')
    .style('font-weight', 'normal')
    .text(`(${dimension})`);

  // Add edges between nodes
  let edgeGroup = flattenLayerLeftPart.append('g')
    .attr('class', 'edge-group')
    .lower();

  edgeGroup.selectAll('path')
    .data(linkData)
    .enter()
    .append('path')
    .attr('class', d => d.class)
    .attr('id', d => `edge-${d.name}`)
    .attr('d', d => linkGen({ source: d.source, target: d.target }))
    .attr('data-source-x', d => d.source.x)
    .attr('data-source-y', d => d.source.y)
    .attr('data-source-layer', d => d.sourceLayer ?? null)
    .attr('data-source-index', d => d.sourceIndex ?? null)
    .attr('data-target-x', d => d.target.x)
    .attr('data-target-y', d => d.target.y)
    .attr('data-target-layer', d => d.targetLayer ?? null)
    .attr('data-target-index', d => d.targetIndex ?? null)
    .attr('data-move-with-plus', d => d.moveWithPlus ? 'true' : null)
    .attr('data-original-opacity', d => d.opacity)
    .style('fill', 'none')
    .style('stroke-width', d => d.width)
    .style('stroke', d => d.color === undefined ? intermediateColor : d.color)
    .style('opacity', d => d.opacity);

  edgeGroup.selectAll('path.flatten-abstract-output')
    .lower();

  edgeGroup.selectAll('path.flatten,path.flatten-output')
    .style('cursor', 'crosshair')
    .style('pointer-events', 'all')
    .on('mouseover', flattenMouseOverHandler)
    .on('mouseleave', flattenMouseLeaveHandler)
    .on('click', () => { d3.event.stopPropagation() });

  // Add legend
  let legendY = svgPaddings.top + vSpaceAroundGap * (10) + vSpaceAroundGap +
    nodeLength * 10;
  let flattenActivationMinMax = getSafeLegendMinMax(
    displayedFlattenValues,
    cnnLayerMinMax[prevFeatureLayerIndex]
  );
  drawIntermediateLayerLegend({
    legendHeight: 5,
    curLayerIndex: curLayerIndex,
    range: range,
    minMax: flattenActivationMinMax,
    group: intermediateLayer,
    width: intermediateGap + nodeLength - 3,
    legendLayerIndex: prevFeatureLayerIndex,
    gradientAppendingName: `flatten-activation-gradient-${prevFeatureLayerIndex}`,
    tickValues: [flattenActivationMinMax.min, flattenActivationMinMax.max],
    className: 'classifier-flatten-activation-legend',
    x: leftX,
    y: legendY
  });

  if (hiddenDenseLayers.length) {
    let displayedDenseValues = hiddenLayout.layerLayouts.flatMap((layout) =>
      layout.slots
        .filter((slot) => slot.node)
        .map((slot) => slot.node.output)
    );
    let denseLegendX = hiddenLayout.layerXs[0];
    let denseLegendWidth =
      hiddenLayout.layerXs[hiddenLayout.layerXs.length - 1] +
      hiddenLayout.nodeSize - denseLegendX;

    drawDenseActivationLegend({
      group: intermediateLayer,
      x: denseLegendX,
      y: legendY,
      width: denseLegendWidth,
      legendLayerIndex: prevFeatureLayerIndex,
      values: displayedDenseValues,
    });
  } else {
    drawIntermediateLayerLegend({
      legendHeight: 5,
      curLayerIndex: curLayerIndex,
      range: flattenRange,
      minMax: { min: flattenExtent[0], max: flattenExtent[1] },
      group: intermediateLayer,
      width: intermediateGap - 3 - 5,
      legendLayerIndex: prevFeatureLayerIndex,
      gradientAppendingName: `flatten-weight-gradient-${curLayerIndex}-${prevFeatureLayerIndex}-${i}`,
      gradientGap: 0.1,
      colorScale: layerColorScales.weight,
      x: leftX + intermediateGap + nodeLength + pixelWidth + 3,
      y: legendY
    });
  }

  // Add annotation to the intermediate layer
  let intermediateLayerAnnotation = svg.append('g')
    .attr('class', 'intermediate-layer-annotation')
    .style('opacity', 0);

  // Add annotation for the sum operation
  let plusAnnotation = intermediateLayerAnnotation.append('g')
    .attr('class', 'plus-annotation');

  // let textX = nodeCoordinate[curLayerIndex][i].x - 50;
  let textX = intermediateX2;
  let textY = nodeCoordinate[curLayerIndex][i].y + nodeLength +
    kernelRectLength * 3;
  let arrowSY = nodeCoordinate[curLayerIndex][i].y + nodeLength +
    kernelRectLength * 2;
  let arrowTY = nodeCoordinate[curLayerIndex][i].y + nodeLength / 2 +
    plusSymbolRadius;

  if (i == 9) {
    textY -= 110;
    arrowSY -= 70;
    arrowTY -= 18;
  }

  if (hiddenDenseLayers.length) {
    textX = symbolX - 34;
    textY = symbolY + 38;
    arrowSY = symbolY + 24;
    arrowTY = symbolY + plusSymbolRadius + 2;
  }

  let plusText = plusAnnotation.append('text')
    .attr('x', textX)
    .attr('y', textY)
    .attr('class', 'annotation-text')
    .style('dominant-baseline', 'hanging')
    .style('text-anchor', 'middle');

  plusText.append('tspan')
    .style('dominant-baseline', 'hanging')
    .text('Add up all products');

  plusText.append('tspan')
    .attr('x', textX)
    .attr('dy', '1em')
    .style('dominant-baseline', 'hanging')
    .text('(');

  plusText.append('tspan')
    .style('fill', '#66a3c8')
    .style('dominant-baseline', 'hanging')
    .text('element');

  plusText.append('tspan')
    .style('dominant-baseline', 'hanging')
    .text(' × ');

  plusText.append('tspan')
    .style('dominant-baseline', 'hanging')
    .style('fill', '#b58946')
    .text('weight');

  plusText.append('tspan')
    .style('dominant-baseline', 'hanging')
    .text(')');

  plusText.append('tspan')
    .attr('x', textX)
    .attr('dy', '1em')
    .style('dominant-baseline', 'hanging')
    .text('and then ');

  plusText.append('tspan')
    .style('dominant-baseline', 'hanging')
    .style('fill', '#479d94')
    .text('bias');

  drawArrow({
    group: plusAnnotation,
    sx: hiddenDenseLayers.length ? symbolX - plusSymbolRadius - 18 :
      intermediateX2 - 2 * plusSymbolRadius - 3,
    sy: arrowSY,
    tx: hiddenDenseLayers.length ? symbolX - plusSymbolRadius - 2 :
      intermediateX2 - 5,
    ty: arrowTY,
    dr: 30,
    hFlip: i === 9,
    marker: 'marker-alt'
  });

  // Add annotation for the bias
  let biasTextX = intermediateX2 + plusSymbolRadius;
  let biasTextY = nodeCoordinate[curLayerIndex][i].y - 2 * kernelRectLength - 4;

  if (hiddenDenseLayers.length) {
    biasTextX = symbolX - plusSymbolRadius - 8;
    biasTextY = symbolY - nodeLength / 2 - kernelRectLength * 1.5 - 8;
  }

  let biasTextGroup = hiddenDenseLayers.length ?
    intermediateLayerAnnotation : flattenLayerLeftPart;

  biasTextGroup.append('text')
    .attr('class', 'annotation-text bias-annotation')
    .attr('x', biasTextX)
    .attr('y', biasTextY)
    .style('text-anchor', 'middle')
    .style('dominant-baseline', 'baseline')
    .text('Bias');

  // Add annotation for the softmax symbol. The click animation reuses this
  // group to replace the prompt with the logit/normalization annotations.
  let softmaxAnnotation = intermediateLayerAnnotation.append('g')
    .attr('class', 'softmax-annotation');

  if (!hiddenDenseLayers.length) {
    let softmaxPromptX = nodeCoordinate[curLayerIndex][i].x -
      Math.max(70, (nodeCoordinate[curLayerIndex][i].x - softmaxX) / 2);

    softmaxAnnotation.append('text')
      .attr('x', softmaxPromptX)
      .attr('y', softmaxTextY)
      .attr('class', 'annotation-text')
      .style('dominant-baseline', 'baseline')
      .style('text-anchor', 'middle')
      .style('font-weight', 700)
      .text('Click ')
      .append('tspan')
      .attr('dx', 1)
      .style('font-weight', 400)
      .text('to learn more');

    drawArrow({
      group: softmaxAnnotation,
      sx: softmaxPromptX - 5,
      sy: softmaxTextY + 4,
      tx: softmaxX + softmaxWidth / 2,
      ty: symbolY - plusSymbolRadius - 4,
      dr: 50,
      hFlip: true
    });
  } else {
    softmaxAnnotation.append('text')
      .attr('x', softmaxX + softmaxWidth / 2)
      .attr('y', softmaxTextY)
      .attr('class', 'annotation-text')
      .style('dominant-baseline', 'baseline')
      .style('text-anchor', 'middle')
      .style('opacity', 0)
      .style('pointer-events', 'none')
      .text('Click to learn more');
  }

  // Add annotation for the flatten layer
  let flattenAnnotation = intermediateLayerAnnotation.append('g')
    .attr('class', 'flatten-annotation');

  textX = leftX - 48;
  textY = nodeCoordinate[prevFeatureLayerIndex][0].y;

  let flattenText = flattenAnnotation.append('text')
    .attr('x', textX)
    .attr('y', textY)
    .attr('class', 'annotation-text')
    .style('dominant-baseline', 'hanging')
    .style('text-anchor', 'middle');

  let tempTspan = flattenText.append('tspan')
    .style('dominant-baseline', 'hanging')
    .style('font-weight', 700)
    .text('Hover over ');

  tempTspan.append('tspan')
    .attr('dx', 1)
    .style('font-weight', 400)
    .style('dominant-baseline', 'hanging')
    .text('matrix to');

  flattenText.append('tspan')
    .style('dominant-baseline', 'hanging')
    .attr('x', textX)
    .attr('dy', '1em')
    .text('see how it is flattened');

  flattenText.append('tspan')
    .style('dominant-baseline', 'hanging')
    .attr('x', textX)
    .attr('dy', '1em')
    .text('into a 1D array!');

  drawArrow({
    group: flattenAnnotation,
    sx: textX + 36,
    sy: textY + nodeLength * 0.4 + 12,
    tx: leftX - 4,
    ty: textY + nodeLength / 2,
    dr: 80,
    hFlip: true
  });

  // Add annotation to explain the middle images
  textY = nodeCoordinate[prevFeatureLayerIndex][1].y;

  let middleText = flattenAnnotation.append('text')
    .attr('x', textX)
    .attr('y', textY)
    .attr('class', 'annotation-text')
    .style('dominant-baseline', 'hanging')
    .style('text-anchor', 'middle');

  middleText.append('tspan')
    .style('dominant-baseline', 'hanging')
    .text('Same flattening');

  middleText.append('tspan')
    .style('dominant-baseline', 'hanging')
    .attr('x', textX)
    .attr('dy', '1em')
    .text('operation for');

  middleText.append('tspan')
    .style('dominant-baseline', 'hanging')
    .attr('x', textX)
    .attr('dy', '1em')
    .text('each neuron');

  drawArrow({
    group: flattenAnnotation,
    sx: textX + 34,
    sy: textY + 25,
    tx: leftX - 4,
    ty: textY + nodeLength / 2 - 2,
    dr: 80,
    hFlip: true,
    marker: 'marker-alt'
  });


  // Add annotation for the output neuron
  let outputAnnotation = intermediateLayerAnnotation.append('g')
    .attr('class', 'output-annotation');

  outputAnnotation.append('text')
    .attr('x', nodeCoordinate[outputLayerIndex][i].x)
    .attr('y', nodeCoordinate[outputLayerIndex][i].y + 10)
    .attr('class', 'annotation-text')
    .text(`(${d3.format('.4f')(selectedOutputNode.output)})`);


  /* Prototype of using arc to represent the flatten layer (future)
  let pie = d3.pie()
    .padAngle(0)
    .sort(null)
    .value(d => d.output)
    .startAngle(0)
    .endAngle(-Math.PI);

  let radius = 490 / 2;
  let arc = d3.arc()
    .innerRadius(radius - 20)
    .outerRadius(radius);

  let arcs = pie(flattenLayerData);
  console.log(arcs);

  let test = svg.append('g')
    .attr('class', 'test')
    .attr('transform', 'translate(500, 250)');

  test.selectAll("path")
    .data(arcs)
    .join("path")
      .attr('class', 'arc')
      .attr("fill", d => colorScale((d.value + range/2) / range))
      .attr("d", arc);
  */

  // Show everything
  svg.selectAll('g.intermediate-layer, g.intermediate-layer-annotation')
    .transition()
    .delay(500)
    .duration(500)
    .ease(d3.easeCubicInOut)
    .style('opacity', 1);
}
