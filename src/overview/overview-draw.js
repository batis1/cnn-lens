/* global d3, SmoothScroll */

import {
  svgStore, vSpaceAroundGapStore, hSpaceAroundGapStore, cnnStore,
  nodeCoordinateStore, selectedScaleLevelStore, layerDisplayOrderStore, cnnLayerRangesStore,
  detailedModeStore, cnnLayerMinMaxStore, hoverInfoStore, manualLayerRevealStore
} from '../stores.js';
import {
  getExtent, getInputKnot, getLinkData, getOutputKnot
} from './draw-utils.js';
import { overviewConfig } from '../config.js';

// Configs
const layerColorScales = overviewConfig.layerColorScales;
const nodeLength = overviewConfig.nodeLength;
const numLayers = overviewConfig.numLayers;
const edgeOpacity = overviewConfig.edgeOpacity;
const edgeInitColor = overviewConfig.edgeInitColor;
const edgeStrokeWidth = overviewConfig.edgeStrokeWidth;
const svgPaddings = overviewConfig.svgPaddings;
const gapRatio = overviewConfig.gapRatio;
const classLists = overviewConfig.classLists;
const formater = d3.format('.4f');
const edgeRevealDuration = 900;
const edgeRevealLayerDelay = 550;
const edgeRevealStagger = 18;
const packetTravelDuration = 1400;
const packetLayerOffset = 120;
const packetRadius = 2.8;
const nodePulseDuration = 240;
const nodePulseSettleDuration = 260;
const winnerLabelDelayOffset = 180;
const winnerLabelDuration = 380;
const layerRevealDuration = 520; // Task 6.3
const layerRevealStagger = 380; //Task 6.3
const manualRevealPacketOffset = 80;
const ENABLE_EDGE_PACKETS = false;
const packetColors = {
  green: '#BDE4B2',
  pink: '#F472B6',
  violet: '#A855F7',
  amber: '#F59E0B',
  cyan: '#22C7F0',
  coral: '#FB7185',
  lightPurple: '#E9D5FF'
};

const isStyleTestMode = () =>
  Boolean(document.querySelector('.overview.style-test-mode'));

const formatLayerLabel = (name) => {
  if (!name) {
    return '';
  }

  let normalized = String(name).toLowerCase();
  if (normalized === 'output') {
    return 'Output';
  }
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

const updateStyleTestLayerHierarchy = (activeLayerIndex = null) => {
  if (!svg || !isStyleTestMode()) {
    return;
  }

  let hasActiveLayer = Number.isFinite(activeLayerIndex);
  svg.selectAll('g.layer-label, g.layer-detailed-label')
    .classed('is-style-active', function () {
      let labelIndex = Number(String(this.id || '').match(/-(\d+)$/)?.[1]);
      return hasActiveLayer && labelIndex === activeLayerIndex;
    })
    .classed('is-style-muted', function () {
      let labelIndex = Number(String(this.id || '').match(/-(\d+)$/)?.[1]);
      return hasActiveLayer && Number.isFinite(labelIndex) && labelIndex !== activeLayerIndex;
    });
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

let layerDisplayOrder = 'model';
layerDisplayOrderStore.subscribe(value => { layerDisplayOrder = value; })

let cnnLayerRanges = undefined;
cnnLayerRangesStore.subscribe(value => { cnnLayerRanges = value; })

let cnnLayerMinMax = undefined;
cnnLayerMinMaxStore.subscribe(value => { cnnLayerMinMax = value; })

let detailedMode = undefined;
detailedModeStore.subscribe(value => { detailedMode = value; })

let manualRevealState = { //Task 6.3
  displayOrder: [],
  nextDisplayIndex: 1,
  cnnGroup: undefined,
  controlGroup: undefined,
};

export const applyCurrentEdgeVisibility = (overrideFilter = null) => {
  if (!svg) {
    return;
  }

  let edgeSelection = svg.select('g.edge-group').selectAll('path.edge');
  if (edgeSelection.empty()) {
    return;
  }

  edgeSelection.each(function (d) {
    let isVisibleByReveal = d.displayTargetLayerIndex < manualRevealState.nextDisplayIndex;
    let isVisible = overrideFilter ? overrideFilter(d, isVisibleByReveal) : isVisibleByReveal;

    d3.select(this)
      .interrupt('edge-reveal')
      .style('visibility', isVisible ? 'visible' : 'hidden')
      .style('opacity', isVisible ? edgeOpacity : 0)
      .style('pointer-events', isVisible ? 'stroke' : 'none')
      .style('stroke-dasharray', null)
      .style('stroke-dashoffset', null);
  });
}

export const setLayerHoverState = (layerIndex, isHovered) => {
  if (!svg || layerIndex === undefined || layerIndex < 0) {
    return;
  }

  let layerGroup = svg.select(`#cnn-layer-group-${layerIndex}`);
  let outline = layerGroup.select('rect.layer-hover-outline');
  let labels = svg.selectAll(`#layer-label-${layerIndex}, #layer-detailed-label-${layerIndex}`);

  if (isHovered) {
    outline
      .classed('active', true)
      .style('opacity', 1)
      .style('stroke-dashoffset', 0)
      .transition('layer-hover-dash')
      .duration(900)
      .ease(d3.easeLinear)
      .style('stroke-dashoffset', -18)
      .on('end', function repeat() {
        d3.select(this)
          .style('stroke-dashoffset', 0)
          .transition('layer-hover-dash')
          .duration(900)
          .ease(d3.easeLinear)
          .style('stroke-dashoffset', -18)
          .on('end', repeat);
      });

    labels.select('rect.layer-label-highlight')
      .style('opacity', 0.88);
    labels.select('text')
      .style('fill', '#475569');
    return;
  }

  outline
    .classed('active', false)
    .interrupt('layer-hover-dash')
    .transition('layer-hover-out')
    .duration(180)
    .style('opacity', 0);

  labels.select('rect.layer-label-highlight')
    .transition('layer-label-highlight-out')
    .duration(160)
    .style('opacity', 0);
  labels.select('text')
    .style('fill', null);
}

/**
 * Use bounded d3 data to draw one canvas
 * @param {object} d d3 data
 * @param {index} i d3 data index
 * @param {[object]} g d3 group
 * @param {number} range color range map (max - min)
 */
export const drawOutput = (d, i, g, range) => {
  let image = g[i];
  let colorScale = layerColorScales[d.type];

  if (d.type === 'input') {
    colorScale = colorScale[d.index];
  }

  // Set up a second convas in order to resize image
  let imageLength = d.output.length === undefined ? 1 : d.output.length;
  let bufferCanvas = document.createElement("canvas");
  let bufferContext = bufferCanvas.getContext("2d");
  bufferCanvas.width = imageLength;
  bufferCanvas.height = imageLength;

  // Fill image pixel array
  let imageSingle = bufferContext.getImageData(0, 0, imageLength, imageLength);
  let imageSingleArray = imageSingle.data;

  if (imageLength === 1) {
    imageSingleArray[0] = d.output;
  } else {
    for (let i = 0; i < imageSingleArray.length; i += 4) {
      let pixeIndex = Math.floor(i / 4);
      let row = Math.floor(pixeIndex / imageLength);
      let column = pixeIndex % imageLength;
      let color = undefined;
      if (d.type === 'input' || d.type === 'fc') {
        color = d3.rgb(colorScale(1 - d.output[row][column]))
      } else {
        color = d3.rgb(colorScale((d.output[row][column] + range / 2) / range));
      }

      imageSingleArray[i] = color.r;
      imageSingleArray[i + 1] = color.g;
      imageSingleArray[i + 2] = color.b;
      imageSingleArray[i + 3] = 255;
    }
  }

  // canvas.toDataURL() only exports image in 96 DPI, so we can hack it to have
  // higher DPI by rescaling the image using canvas magic
  let largeCanvas = document.createElement('canvas');
  largeCanvas.width = nodeLength * 3;
  largeCanvas.height = nodeLength * 3;
  let largeCanvasContext = largeCanvas.getContext('2d');

  // Use drawImage to resize the original pixel array, and put the new image
  // (canvas) into corresponding canvas
  bufferContext.putImageData(imageSingle, 0, 0);
  largeCanvasContext.drawImage(bufferCanvas, 0, 0, imageLength, imageLength,
    0, 0, nodeLength * 3, nodeLength * 3);

  let imageDataURL = largeCanvas.toDataURL();
  d3.select(image).attr('xlink:href', imageDataURL);

  // Destory the buffer canvas
  bufferCanvas.remove();
  largeCanvas.remove();
}

/**
 * Draw bar chart to encode the output value
 * @param {object} d d3 data
 * @param {index} i d3 data index
 * @param {[object]} g d3 group
 * @param {function} scale map value to length
 */
const drawOutputScore = (d, i, g, scale) => {
  let group = d3.select(g[i]);
  group.select('rect.output-rect')
    .transition('output')
    .delay(500)
    .duration(800)
    .ease(d3.easeCubicIn)
    .attr('width', scale(d.output))
}

export const drawCustomImage = (image, inputLayer) => {

  let imageWidth = image.width;
  // Set up a second convas in order to resize image
  let imageLength = inputLayer[0].output.length;
  let bufferCanvas = document.createElement("canvas");
  let bufferContext = bufferCanvas.getContext("2d");
  bufferCanvas.width = imageLength;
  bufferCanvas.height = imageLength;

  // Fill image pixel array
  let imageSingle = bufferContext.getImageData(0, 0, imageLength, imageLength);
  let imageSingleArray = imageSingle.data;

  for (let i = 0; i < imageSingleArray.length; i += 4) {
    let pixeIndex = Math.floor(i / 4);
    let row = Math.floor(pixeIndex / imageLength);
    let column = pixeIndex % imageLength;

    let red = inputLayer[0].output[row][column];
    let green = inputLayer[1].output[row][column];
    let blue = inputLayer[2].output[row][column];

    imageSingleArray[i] = red * 255;
    imageSingleArray[i + 1] = green * 255;
    imageSingleArray[i + 2] = blue * 255;
    imageSingleArray[i + 3] = 255;
  }

  // canvas.toDataURL() only exports image in 96 DPI, so we can hack it to have
  // higher DPI by rescaling the image using canvas magic
  let largeCanvas = document.createElement('canvas');
  largeCanvas.width = imageWidth * 3;
  largeCanvas.height = imageWidth * 3;
  let largeCanvasContext = largeCanvas.getContext('2d');

  // Use drawImage to resize the original pixel array, and put the new image
  // (canvas) into corresponding canvas
  bufferContext.putImageData(imageSingle, 0, 0);
  largeCanvasContext.drawImage(bufferCanvas, 0, 0, imageLength, imageLength,
    0, 0, imageWidth * 3, imageWidth * 3);

  let imageDataURL = largeCanvas.toDataURL();
  // d3.select(image).attr('xlink:href', imageDataURL);
  image.src = imageDataURL;

  // Destory the buffer canvas
  bufferCanvas.remove();
  largeCanvas.remove();
}

/**
 * Create color gradient for the legend
 * @param {[object]} g d3 group
 * @param {function} colorScale Colormap
 * @param {string} gradientName Label for gradient def
 * @param {number} min Min of legend value
 * @param {number} max Max of legend value
 */
const getLegendGradient = (g, colorScale, gradientName, min, max) => {
  if (min === undefined) { min = 0; }
  if (max === undefined) { max = 1; }
  let gradient = g.append('defs')
    .append('svg:linearGradient')
    .attr('id', `${gradientName}`)
    .attr('x1', '0%')
    .attr('y1', '100%')
    .attr('x2', '100%')
    .attr('y2', '100%')
    .attr('spreadMethod', 'pad');
  let interpolation = 10
  for (let i = 0; i < interpolation; i++) {
    let curProgress = i / (interpolation - 1);
    let curColor = colorScale(curProgress * (max - min) + min);
    gradient.append('stop')
      .attr('offset', `${curProgress * 100}%`)
      .attr('stop-color', curColor)
      .attr('stop-opacity', 1);
  }
}

/**
 * Draw all legends
 * @param {object} legends Parent group
 * @param {number} legendHeight Height of the legend element
 */
const drawLegends = (legends, legendHeight) => {
  let visibleLayerCount = cnn.length || numLayers;
  let numOfComponent = Math.max(1, Math.ceil((visibleLayerCount - 2) / 5));

  // Add local legends
  for (let i = 0; i < numOfComponent; i++) {
    let start = 1 + i * 5;
    let range1 = cnnLayerRanges.local[start];
    let range2 = cnnLayerRanges.local[start + 2];

    let localLegendScale1 = d3.scaleLinear()
      .range([0, 2 * nodeLength + hSpaceAroundGap - 1.2])
      .domain([-range1 / 2, range1 / 2]);

    let localLegendScale2 = d3.scaleLinear()
      .range([0, 3 * nodeLength + 2 * hSpaceAroundGap - 1.2])
      .domain([-range2 / 2, range2 / 2]);

    let localLegendAxis1 = d3.axisBottom()
      .scale(localLegendScale1)
      .tickFormat(d3.format('.2f'))
      .tickValues([-range1 / 2, 0, range1 / 2]);

    let localLegendAxis2 = d3.axisBottom()
      .scale(localLegendScale2)
      .tickFormat(d3.format('.2f'))
      .tickValues([-range2 / 2, 0, range2 / 2]);

    let localLegend1 = legends.append('g')
      .attr('class', 'legend local-legend')
      .attr('id', `local-legend-${i}-1`)
      .classed('hidden', !detailedMode || selectedScaleLevel !== 'local')
      .attr('transform', `translate(${nodeCoordinate[start][0].x}, ${0})`);

    localLegend1.append('g')
      .attr('transform', `translate(0, ${legendHeight - 3})`)
      .call(localLegendAxis1)

    localLegend1.append('rect')
      .attr('width', 2 * nodeLength + hSpaceAroundGap)
      .attr('height', legendHeight)
      .style('fill', 'url(#convGradient)');

    let localLegend2 = legends.append('g')
      .attr('class', 'legend local-legend')
      .attr('id', `local-legend-${i}-2`)
      .classed('hidden', !detailedMode || selectedScaleLevel !== 'local')
      .attr('transform', `translate(${nodeCoordinate[start + 2][0].x}, ${0})`);

    localLegend2.append('g')
      .attr('transform', `translate(0, ${legendHeight - 3})`)
      .call(localLegendAxis2)

    localLegend2.append('rect')
      .attr('width', 3 * nodeLength + 2 * hSpaceAroundGap)
      .attr('height', legendHeight)
      .style('fill', 'url(#convGradient)');
  }

  // Add module legends
  for (let i = 0; i < numOfComponent; i++) {
    let start = 1 + i * 5;
    let range = cnnLayerRanges.module[start];

    let moduleLegendScale = d3.scaleLinear()
      .range([0, 5 * nodeLength + 3 * hSpaceAroundGap +
        1 * hSpaceAroundGap * gapRatio - 1.2])
      .domain([-range / 2, range / 2]);

    let moduleLegendAxis = d3.axisBottom()
      .scale(moduleLegendScale)
      .tickFormat(d3.format('.2f'))
      .tickValues([-range / 2, -(range / 4), 0, range / 4, range / 2]);

    let moduleLegend = legends.append('g')
      .attr('class', 'legend module-legend')
      .attr('id', `module-legend-${i}`)
      .classed('hidden', !detailedMode || selectedScaleLevel !== 'module')
      .attr('transform', `translate(${nodeCoordinate[start][0].x}, ${0})`);

    moduleLegend.append('g')
      .attr('transform', `translate(0, ${legendHeight - 3})`)
      .call(moduleLegendAxis)

    moduleLegend.append('rect')
      .attr('width', 5 * nodeLength + 3 * hSpaceAroundGap +
        1 * hSpaceAroundGap * gapRatio)
      .attr('height', legendHeight)
      .style('fill', 'url(#convGradient)');
  }

  // Add global legends
  let start = 1;
  let range = cnnLayerRanges.global[start];

  let globalLegendScale = d3.scaleLinear()
    .range([0, 10 * nodeLength + 6 * hSpaceAroundGap +
      3 * hSpaceAroundGap * gapRatio - 1.2])
    .domain([-range / 2, range / 2]);

  let globalLegendAxis = d3.axisBottom()
    .scale(globalLegendScale)
    .tickFormat(d3.format('.2f'))
    .tickValues([-range / 2, -(range / 4), 0, range / 4, range / 2]);

  let globalLegend = legends.append('g')
    .attr('class', 'legend global-legend')
    .attr('id', 'global-legend')
    .classed('hidden', !detailedMode || selectedScaleLevel !== 'global')
    .attr('transform', `translate(${nodeCoordinate[start][0].x}, ${0})`);

  globalLegend.append('g')
    .attr('transform', `translate(0, ${legendHeight - 3})`)
    .call(globalLegendAxis)

  globalLegend.append('rect')
    .attr('width', 10 * nodeLength + 6 * hSpaceAroundGap +
      3 * hSpaceAroundGap * gapRatio)
    .attr('height', legendHeight)
    .style('fill', 'url(#convGradient)');


  // Add output legend
  let outputRectScale = d3.scaleLinear()
    .domain(cnnLayerRanges.output)
    .range([0, nodeLength - 1.2]);

  let outputLegendAxis = d3.axisBottom()
    .scale(outputRectScale)
    .tickFormat(d3.format('.1f'))
    .tickValues([0, cnnLayerRanges.output[1]])

  let outputLegend = legends.append('g')
    .attr('class', 'legend output-legend')
    .attr('id', 'output-legend')
    .classed('hidden', !detailedMode)
    .attr('transform', `translate(${nodeCoordinate[visibleLayerCount - 1][0].x}, ${0})`);

  outputLegend.append('g')
    .attr('transform', `translate(0, ${legendHeight - 3})`)
    .call(outputLegendAxis);

  outputLegend.append('rect')
    .attr('width', nodeLength)
    .attr('height', legendHeight)
    .style('fill', 'gray');

  // Add input image legend
  let inputScale = d3.scaleLinear()
    .range([0, nodeLength - 1.2])
    .domain([0, 1]);

  let inputLegendAxis = d3.axisBottom()
    .scale(inputScale)
    .tickFormat(d3.format('.1f'))
    .tickValues([0, 0.5, 1]);

  let inputLegend = legends.append('g')
    .attr('class', 'legend input-legend')
    .classed('hidden', !detailedMode)
    .attr('transform', `translate(${nodeCoordinate[0][0].x}, ${0})`);

  inputLegend.append('g')
    .attr('transform', `translate(0, ${legendHeight - 3})`)
    .call(inputLegendAxis);

  inputLegend.append('rect')
    .attr('x', 0.3)
    .attr('width', nodeLength - 0.3)
    .attr('height', legendHeight)
    .attr('transform', `rotate(180, ${nodeLength / 2}, ${legendHeight / 2})`)
    .style('stroke', 'rgb(20, 20, 20)')
    .style('stroke-width', 0.3)
    .style('fill', 'url(#inputGradient)');
}

/**
 * Match packet colors to the current displayed layer order, not only the
 * original model index order.
 * @param {object|number} edgeOrDisplayLayerIndex
 */
const getPacketColor = (edgeOrDisplayLayerIndex) => {
  let displayTargetLayerIndex = typeof edgeOrDisplayLayerIndex === 'object' ?
    edgeOrDisplayLayerIndex.displayTargetLayerIndex : edgeOrDisplayLayerIndex;
  let targetLayerType = typeof edgeOrDisplayLayerIndex === 'object' ?
    edgeOrDisplayLayerIndex.targetLayerType : undefined;

  if (targetLayerType === 'fc' || displayTargetLayerIndex >= cnn.length - 1) {
    return packetColors.lightPurple;
  }

  let palette = [
    packetColors.green,
    packetColors.pink,
    packetColors.violet,
    packetColors.amber,
    packetColors.cyan,
    packetColors.coral
  ];

  let featureLayerCount = Math.max(cnn.length - 2, 1);
  let featureDisplayIndex = Math.max(displayTargetLayerIndex - 1, 0);
  let bucketSize = Math.max(1, Math.ceil(featureLayerCount / palette.length));
  let paletteIndex = Math.min(
    palette.length - 1,
    Math.floor(featureDisplayIndex / bucketSize),
  );

  return palette[paletteIndex];
}

/**
 * D3 in this project is older and does not provide d3.maxIndex.
 * Find the winning output index manually.
 * @param {[object]} outputLayer
 */
const getWinningOutputIndex = (outputLayer) => {
  let maxIndex = 0;
  let maxValue = -Infinity;

  outputLayer.forEach((node, index) => {
    if (node.output > maxValue) {
      maxValue = node.output;
      maxIndex = index;
    }
  });

  return maxIndex;
}

const getFeatureLayerDisplayOrder = () => {
  // Task 3 at lines 479-506: compute the feature-layer display order from the
  // current model so conv, relu, and pool layers can be rearranged in the UI.
  let featureLayerIndices = cnn
    .map((layer, index) => ({ index: index, type: layer[0].type }))
    .filter((entry) => entry.index > 0 && entry.index < cnn.length - 1);

  if (layerDisplayOrder === 'model') {
    return featureLayerIndices.map((entry) => entry.index);
  }

  let typePriority = {
    'conv-relu-pool': ['conv', 'relu', 'pool', 'fc'],
    'conv-pool-relu': ['conv', 'pool', 'relu', 'fc'],
    'relu-conv-pool': ['relu', 'conv', 'pool', 'fc'],
  }[layerDisplayOrder];

  if (!typePriority) {
    return featureLayerIndices.map((entry) => entry.index);
  }

  let orderLookup = new Map(typePriority.map((type, i) => [type, i]));
  return featureLayerIndices
    .sort((a, b) => {
      let aOrder = orderLookup.has(a.type) ? orderLookup.get(a.type) : 999;
      let bOrder = orderLookup.has(b.type) ? orderLookup.get(b.type) : 999;
      if (aOrder !== bOrder) {
        return aOrder - bOrder;
      }
      return a.index - b.index;
    })
    .map((entry) => entry.index);
}

const getDisplayOrder = () => {
  let featureOrder = getFeatureLayerDisplayOrder();
  return [0, ...featureOrder, cnn.length - 1];
}

const getDisplayLayerIndexLookup = () => {
  let lookup = new Map();
  getDisplayOrder().forEach((layerIndex, displayIndex) => {
    lookup.set(layerIndex, displayIndex);
  });
  return lookup;
}
// Task 5 at lines 525-527
const getDisplayGapWeight = (layer) => {
  if (layer[0].type === 'conv') {
    return gapRatio;
  }
  return layer[0].layerName === 'output' ? Math.max(1.4, gapRatio / 2) : 1;
}
// Task 4 at lines 525-575: build link data based on the display order instead of model order so that the animation and hover interactions are consistent with the current layer arrangement.
const buildDisplayOrderLinkData = (nodeCoordinate, cnn, displayOrder) => {
  let linkData = [];

  for (let di = 1; di < displayOrder.length; di++) {
    let sourceLayerIndex = displayOrder[di - 1];
    let targetLayerIndex = displayOrder[di];
    let sourceLayer = cnn[sourceLayerIndex];
    let targetLayer = cnn[targetLayerIndex];
    let targetType = targetLayer[0].type;

    if (targetType === 'conv' || targetType === 'fc') {
      for (let targetNodeIndex = 0; targetNodeIndex < targetLayer.length; targetNodeIndex++) {
        let curTarget = getInputKnot(nodeCoordinate[targetLayerIndex][targetNodeIndex]);
        for (let sourceNodeIndex = 0; sourceNodeIndex < sourceLayer.length; sourceNodeIndex++) {
          let curSource = getOutputKnot(nodeCoordinate[sourceLayerIndex][sourceNodeIndex]);
          linkData.push({
            source: curSource,
            target: curTarget,
            weight: null,
            sourceLayerIndex: sourceLayerIndex,
            sourceNodeIndex: sourceNodeIndex,
            targetLayerIndex: targetLayerIndex,
            targetNodeIndex: targetNodeIndex,
            displaySourceLayerIndex: di - 1,
            displayTargetLayerIndex: di,
            targetLayerType: targetType,
          });
        }
      }
      continue;
    }

    let pairCount = Math.min(sourceLayer.length, targetLayer.length);
    for (let nodeIndex = 0; nodeIndex < pairCount; nodeIndex++) {
      linkData.push({
        source: getOutputKnot(nodeCoordinate[sourceLayerIndex][nodeIndex]),
        target: getInputKnot(nodeCoordinate[targetLayerIndex][nodeIndex]),
        weight: null,
        sourceLayerIndex: sourceLayerIndex,
        sourceNodeIndex: nodeIndex,
        targetLayerIndex: targetLayerIndex,
        targetNodeIndex: nodeIndex,
        displaySourceLayerIndex: di - 1,
        displayTargetLayerIndex: di,
        targetLayerType: targetType,
      });
    }
  }

  return linkData;
}

const drawStageGroups = (cnnGroup, height) => {
  if (layerDisplayOrder !== 'model') {
    let existing = cnnGroup.select('g.stage-grouping');
    if (!existing.empty()) {
      existing.remove();
    }
    return;
  }

  let existing = cnnGroup.select('g.stage-grouping');
  if (!existing.empty()) {
    existing.remove();
  }
}


/**
 * Reveal edges progressively from earlier layers to later layers.
 * This keeps the overview calmer and matches the slide-style flow animation.
 * @param {object} edgeSelection D3 selection of path.edge elements
 */
const animateEdgeReveal = (edgeSelection, baseDelay = 0) => {
  edgeSelection
    .sort((a, b) => {
      if (a.displayTargetLayerIndex !== b.displayTargetLayerIndex) {
        return a.displayTargetLayerIndex - b.displayTargetLayerIndex;
      }
      if (a.targetNodeIndex !== b.targetNodeIndex) {
        return a.targetNodeIndex - b.targetNodeIndex;
      }
      return a.sourceNodeIndex - b.sourceNodeIndex;
    })
    .each(function (d) {
      let path = d3.select(this);
      let totalLength = this.getTotalLength();
      let intraLayerDelay =
        (d.targetNodeIndex * 4 + d.sourceNodeIndex) * edgeRevealStagger;

      path
        .style('opacity', 0)
        .style('visibility', 'visible')
        .style('stroke-dasharray', `${totalLength} ${totalLength}`)
        .style('stroke-dashoffset', totalLength)
        .transition('edge-reveal')
        .delay(baseDelay + (d.displayTargetLayerIndex - 1) * edgeRevealLayerDelay + intraLayerDelay)
        .duration(edgeRevealDuration)
        .ease(d3.easeLinear)
        .style('opacity', edgeOpacity)
        .style('stroke-dashoffset', 0)
        .on('end', function () {
          d3.select(this)
            .style('stroke-dasharray', null)
            .style('stroke-dashoffset', null);
        });
    });
}

/**
 * Animate small packets along the revealed edges.
 * @param {object} packetSelection D3 selection of circle elements
 */
const animateEdgePackets = (packetSelection, baseDelay = 0) => {
  if (!ENABLE_EDGE_PACKETS) {
    packetSelection.remove();
    return;
  }

  packetSelection.each(function (d) {
    let circle = d3.select(this);
    let path = d.path;
    let totalLength = path.getTotalLength();
    let intraLayerDelay =
      (d.targetNodeIndex * 4 + d.sourceNodeIndex) * edgeRevealStagger;
    let revealDelay =
      (d.displayTargetLayerIndex - 1) * edgeRevealLayerDelay + intraLayerDelay;
    let packetDelay = baseDelay + revealDelay + packetLayerOffset;

    circle
      .style('opacity', 0)
      .transition('packet-fade-in')
      .delay(packetDelay)
      .duration(120)
      .style('opacity', 1)
      .transition('packet-travel')
      .duration(packetTravelDuration)
      .ease(d3.easeLinear)
      .attrTween('transform', () => (t) => {
        let point = path.getPointAtLength(t * totalLength);
        return `translate(${point.x}, ${point.y})`;
      })
      .transition('packet-fade-out')
      .duration(180)
      .style('opacity', 0)
      .remove();
  });
}
//Task 6.3
const animateEdgeRevealForStep = (edgeSelection) => {
  edgeSelection
    .sort((a, b) => {
      if (a.targetNodeIndex !== b.targetNodeIndex) {
        return a.targetNodeIndex - b.targetNodeIndex;
      }
      return a.sourceNodeIndex - b.sourceNodeIndex;
    })
    .each(function (d) {
      let path = d3.select(this);
      let totalLength = this.getTotalLength();
      let intraLayerDelay =
        (d.targetNodeIndex * 4 + d.sourceNodeIndex) * edgeRevealStagger;

      path
        .interrupt('edge-reveal')
        .style('opacity', 0)
        .style('visibility', 'visible')
        .style('pointer-events', 'none')
        .style('stroke-dasharray', `${totalLength} ${totalLength}`)
        .style('stroke-dashoffset', totalLength)
        .transition('edge-reveal')
        .delay(intraLayerDelay)
        .duration(edgeRevealDuration)
        .ease(d3.easeLinear)
        .style('opacity', edgeOpacity)
        .style('stroke-dashoffset', 0)
        .on('end', function () {
          d3.select(this)
            .style('pointer-events', 'stroke')
            .style('stroke-dasharray', null)
            .style('stroke-dashoffset', null);
        });
    });
}

// Task 6.3
const animateEdgePacketsForStep = (cnnGroup, edgeSelection) => {
  let edges = edgeSelection.nodes();
  let edgeAnimationData = edgeSelection.data().map((d, i) => {
    let intraLayerDelay =
      (d.targetNodeIndex * 4 + d.sourceNodeIndex) * edgeRevealStagger;

    return {
      ...d,
      path: edges[i],
      revealDelay: intraLayerDelay,
      packetDelay: intraLayerDelay + manualRevealPacketOffset,
    };
  });

  if (!ENABLE_EDGE_PACKETS) {
    cnnGroup.select('g.edge-packet-group').remove();
    animateNodeArrivalPulses(cnnGroup, edgeAnimationData);
    return;
  }

  let packetGroup = cnnGroup.select('g.edge-packet-group');
  if (packetGroup.empty()) {
    packetGroup = cnnGroup.append('g')
      .attr('class', 'edge-packet-group')
      .style('pointer-events', 'none');
  }

  let packets = packetGroup.selectAll('circle.edge-packet-step')
    .data(edgeAnimationData)
    .enter()
    .append('circle')
    .attr('class', 'edge-packet edge-packet-step')
    .attr('r', packetRadius)
    .attr('transform', (d) => `translate(${d.source.x}, ${d.source.y})`)
    .style('fill', (d) => getPacketColor(d))
    .style('stroke', 'none')
    .style('opacity', 0);

  packets.each(function (d) {
    let circle = d3.select(this);
    let totalLength = d.path.getTotalLength();

    circle
      .transition('packet-fade-in')
      .delay(d.packetDelay)
      .duration(120)
      .style('opacity', 1)
      .transition('packet-travel')
      .duration(packetTravelDuration)
      .ease(d3.easeLinear)
      .attrTween('transform', () => (t) => {
        let point = d.path.getPointAtLength(t * totalLength);
        return `translate(${point.x}, ${point.y})`;
      })
      .transition('packet-fade-out')
      .duration(180)
      .style('opacity', 0)
      .remove();
  });

  animateNodeArrivalPulses(cnnGroup, edgeAnimationData);
}

/**
 * Pulse destination nodes when packets arrive.
 * @param {object} cnnGroup D3 group containing the network
 * @param {[object]} edgeData Edge metadata with path timing fields
 */
const animateNodeArrivalPulses = (cnnGroup, edgeData) => {
  let nodePulseData = [];
  let pulseLookup = {};

  edgeData.forEach((edge) => {
    let key = `${edge.targetLayerIndex}-${edge.targetNodeIndex}`;
    let arrivalDelay = edge.packetDelay + packetTravelDuration;

    if (!pulseLookup[key] || arrivalDelay < pulseLookup[key].delay) {
      let layerNode = cnn[edge.targetLayerIndex][edge.targetNodeIndex];
      let nodePoint = nodeCoordinate[edge.targetLayerIndex][edge.targetNodeIndex];
      let isOutput = layerNode.layerName === 'output';

      pulseLookup[key] = {
        key: key,
        delay: arrivalDelay,
        x: nodePoint.x,
        y: nodePoint.y,
        width: isOutput ? nodeLength + 88 : nodeLength + 8,
        height: isOutput ? nodeLength + 10 : nodeLength + 8,
        radius: isOutput ? 7 : 4,
        color: getPacketColor(edge)
      };
    }
  });

  Object.keys(pulseLookup).forEach((key) => {
    nodePulseData.push(pulseLookup[key]);
  });

  nodePulseData.sort((a, b) => a.delay - b.delay);

  cnnGroup.select('g.node-pulse-group').remove();

  let pulseGroup = cnnGroup.append('g')
    .attr('class', 'node-pulse-group')
    .style('pointer-events', 'none');

  let pulses = pulseGroup.selectAll('rect.node-pulse')
    .data(nodePulseData)
    .enter()
    .append('rect')
    .attr('class', 'node-pulse')
    .attr('x', (d) => d.x - 4)
    .attr('y', (d) => d.y - 4)
    .attr('rx', (d) => d.radius)
    .attr('ry', (d) => d.radius)
    .attr('width', (d) => d.width)
    .attr('height', (d) => d.height)
    .style('fill', (d) => d.color)
    .style('opacity', 0);

  pulses.each(function (d) {
    d3.select(this)
      .transition('node-pulse-in')
      .delay(d.delay)
      .duration(nodePulseDuration)
      .ease(d3.easeCubicOut)
      .style('opacity', 0.22)
      .attr('x', d.x - 6)
      .attr('y', d.y - 6)
      .attr('width', d.width + 4)
      .attr('height', d.height + 4)
      .transition('node-pulse-out')
      .duration(nodePulseSettleDuration)
      .ease(d3.easeCubicOut)
      .style('opacity', 0)
      .attr('x', d.x - 8)
      .attr('y', d.y - 8)
      .attr('width', d.width + 8)
      .attr('height', d.height + 8)
      .remove();
  });
}

/**
 * Highlight the predicted output label after the packet flow reaches output.
 */
const animateWinningOutputLabel = (baseDelay = 0) => {
  let outputLayer = cnn[cnn.length - 1];
  let winningIndex = getWinningOutputIndex(outputLayer);
  let styleTestMode = isStyleTestMode();
  let outputLayerRevealDelay = manualRevealState.displayOrder.length ?
    0 : (cnn.length - 2) * edgeRevealLayerDelay;
  let winnerDelay =
    baseDelay + outputLayerRevealDelay + winnerLabelDelayOffset;
  let winnerCoords = nodeCoordinate[cnn.length - 1][winningIndex];

  let allLabels = svg.selectAll('text.output-text');
  let winnerLabel = svg.select(`#layer-${cnn.length - 1}-node-${winningIndex}`)
    .select('text.output-text');
  let winnerBar = svg.select(`#layer-${cnn.length - 1}-node-${winningIndex}`)
    .select('rect.output-rect');
  let winnerGroup = svg.select(`#layer-${cnn.length - 1}-node-${winningIndex}`);

  svg.selectAll('g.node-output')
    .interrupt('winner-node')
    .interrupt('winner-node-settle')
    .attr('transform', null);

  allLabels
    .interrupt('winner-reset')
    .interrupt('winner-highlight')
    .interrupt('winner-settle');
  svg.selectAll('rect.output-rect')
    .interrupt('winner-bar')
    .interrupt('winner-bar-settle');
  winnerGroup.interrupt('winner-node');

  allLabels
    .transition('winner-reset')
    .duration(250)
    .style('fill', styleTestMode ? '#d8eef8' : 'black')
    .style('opacity', styleTestMode ? 0.72 : 0.58)
    .style('font-size', '11px')
    .style('font-weight', '400')
    .style('text-decoration', 'none');

  svg.selectAll('rect.output-rect')
    .transition('winner-bar-reset')
    .duration(200)
    .style('fill', '#9AA4B2')
    .style('opacity', 0.72)
    .attr('height', nodeLength / 4)
    .attr('y', (d, i) => nodeCoordinate[cnn.length - 1][i].y + nodeLength / 2 + 8);

  svg.selectAll('g.output-winner-overlay').remove();

  let winnerText = classLists[winningIndex];
  let winnerCardWidth = styleTestMode
    ? Math.max(104, winnerText.length * 7.5 + 36)
    : nodeLength + 52;
  let overlay = svg.append('g')
    .attr('class', 'output-winner-overlay')
    .style('pointer-events', 'none')
    .style('opacity', 0);

  overlay.append('rect')
    .attr('class', styleTestMode ? 'output-winner-card' : null)
    .attr('x', winnerCoords.x - (styleTestMode ? 10 : 8))
    .attr('y', winnerCoords.y - (styleTestMode ? 8 : 5))
    .attr('rx', 6)
    .attr('ry', 6)
    .attr('width', winnerCardWidth)
    .attr('height', styleTestMode ? nodeLength + 16 : nodeLength + 10)
    .style('fill', styleTestMode ? 'rgba(8, 31, 70, 0.94)' : packetColors.lightPurple)
    .style('stroke', styleTestMode ? 'rgba(14, 252, 255, 0.58)' : packetColors.purple)
    .style('stroke-width', styleTestMode ? 1.1 : 1.5);

  if (styleTestMode) {
    overlay.append('rect')
      .attr('class', 'output-winner-accent')
      .attr('x', winnerCoords.x - 10)
      .attr('y', winnerCoords.y - 4)
      .attr('rx', 2)
      .attr('ry', 2)
      .attr('width', 4)
      .attr('height', nodeLength + 8)
      .style('fill', packetColors.cyan);
  }

  overlay.append('text')
    .attr('class', styleTestMode ? 'output-winner-text' : null)
    .attr('x', winnerCoords.x + (styleTestMode ? 8 : 0))
    .attr('y', winnerCoords.y + nodeLength / 2)
    .style('dominant-baseline', 'middle')
    .style('font-size', styleTestMode ? '13px' : '18px')
    .style('font-weight', styleTestMode ? '700' : '900')
    .style('fill', styleTestMode ? '#f8fdff' : packetColors.purple)
    .text(winnerText);

  winnerLabel
    .transition('winner-highlight')
    .delay(winnerDelay)
    .duration(220)
    .ease(d3.easeCubicOut)
    .style('fill', styleTestMode ? '#f8fdff' : packetColors.purple)
    .style('opacity', 1)
    .style('font-size', styleTestMode ? '12px' : '16px')
    .style('font-weight', styleTestMode ? '600' : '800')
    .style('text-decoration', styleTestMode ? 'none' : 'underline')
    .transition('winner-settle')
    .duration(180)
    .ease(d3.easeCubicOut)
    .style('font-size', styleTestMode ? '12px' : '15px');

  winnerBar
    .transition('winner-bar')
    .delay(winnerDelay - 40)
    .duration(220)
    .ease(d3.easeCubicOut)
    .style('fill', packetColors.purple)
    .style('opacity', 1)
    .attr('height', nodeLength / 2.8)
    .attr('y', nodeCoordinate[cnn.length - 1][winningIndex].y + nodeLength / 2 + 6)
    .transition('winner-bar-settle')
    .duration(260)
    .ease(d3.easeCubicInOut)
    .attr('height', nodeLength / 3.5)
    .attr('y', nodeCoordinate[cnn.length - 1][winningIndex].y + nodeLength / 2 + 7);

  winnerGroup
    .transition('winner-node')
    .delay(winnerDelay - 20)
    .duration(180)
    .ease(d3.easeBackOut.overshoot(1.4))
    .attr('transform', styleTestMode ? 'translate(4, 0)' : 'translate(10, 0)')
    .transition('winner-node-settle')
    .duration(220)
    .ease(d3.easeCubicOut)
    .attr('transform', styleTestMode ? 'translate(2, 0)' : 'translate(6, 0)');

  winnerGroup.select('text.output-rank')
    .transition('winner-rank')
    .delay(winnerDelay - 40)
    .duration(220)
    .style('fill', packetColors.violet)
    .style('opacity', 1);

  overlay
    .transition('winner-overlay')
    .delay(winnerDelay - 10)
    .duration(winnerLabelDuration)
    .ease(d3.easeCubicOut)
    .style('opacity', 0.96)
    .transition('winner-overlay-settle')
    .duration(260)
    .style('opacity', 0.82);

}

const resetOutputWinnerAnimation = () => {
  let styleTestMode = isStyleTestMode();
  svg.selectAll('g.node-output')
    .interrupt('winner-node')
    .interrupt('winner-node-settle')
    .attr('transform', null);

  svg.selectAll('text.output-text')
    .interrupt('winner-reset')
    .interrupt('winner-highlight')
    .interrupt('winner-settle')
    .style('fill', styleTestMode ? '#d8eef8' : 'black')
    .style('opacity', styleTestMode ? 0.72 : 0.58)
    .style('font-size', '11px')
    .style('font-weight', '400')
    .style('text-decoration', 'none');

  svg.selectAll('rect.output-rect')
    .interrupt('winner-bar-reset')
    .interrupt('winner-bar')
    .interrupt('winner-bar-settle')
    .style('fill', '#9AA4B2')
    .style('opacity', 0.72)
    .attr('height', nodeLength / 4)
    .attr('y', (d, i) => nodeCoordinate[cnn.length - 1][i].y + nodeLength / 2 + 8);

  svg.selectAll('text.output-rank')
    .interrupt('winner-rank')
    .style('fill', null)
    .style('opacity', null);

  svg.selectAll('g.output-winner-overlay')
    .interrupt('winner-overlay')
    .interrupt('winner-overlay-settle')
    .remove();
}

/**
 * Replay the overview edge reveal and packet flow.
 * Used on first draw and when the selected input image changes.
 * @param {object} cnnGroup D3 group containing the network
 */
const replayEdgeAnimations = (cnnGroup, baseDelay = 0) => {
  let edges = cnnGroup.select('g.edge-group').selectAll('path.edge');
  let edgeAnimationData = edges.data().map((d, i) => {
    let intraLayerDelay =
      (d.targetNodeIndex * 4 + d.sourceNodeIndex) * edgeRevealStagger;
    let revealDelay =
      (d.displayTargetLayerIndex - 1) * edgeRevealLayerDelay + intraLayerDelay;

    return {
      ...d,
      path: edges.nodes()[i],
      revealDelay: revealDelay,
      packetDelay: baseDelay + revealDelay + packetLayerOffset
    };
  });

  edges.interrupt('edge-reveal');
  edges
    .style('opacity', edgeOpacity)
    .style('stroke-dasharray', null)
    .style('stroke-dashoffset', null);

  animateEdgeReveal(edges, baseDelay);

  cnnGroup.select('g.edge-packet-group').remove();

  let packetGroup = cnnGroup.append('g')
    .attr('class', 'edge-packet-group')
    .style('pointer-events', 'none');

  let packets = packetGroup.selectAll('circle.edge-packet')
    .data(edgeAnimationData)
    .enter()
    .append('circle')
    .attr('class', 'edge-packet')
    .attr('r', packetRadius)
    .attr('transform', (d) => `translate(${d.source.x}, ${d.source.y})`)
    .style('fill', (d) => getPacketColor(d))
    .style('stroke', 'none')
    .style('opacity', 0);

  animateEdgePackets(packets, baseDelay);
  animateNodeArrivalPulses(cnnGroup, edgeAnimationData);
  animateWinningOutputLabel(baseDelay);
}

const setLayerRevealVisibility = (cnnGroup, layerIndex, visible, duration = 0) => {
  let compactTargetOpacity = detailedMode ? 0 : 0.8;
  let detailedTargetOpacity = detailedMode ? 0.7 : 0;

  let layerGroup = cnnGroup.select(`#cnn-layer-group-${layerIndex}`);
  let compactLabel = svg.select(`#layer-label-${layerIndex}`);
  let detailedLabel = svg.select(`#layer-detailed-label-${layerIndex}`);

  layerGroup.interrupt('manual-layer-reveal');
  compactLabel.interrupt('manual-label-reveal');
  detailedLabel.interrupt('manual-detail-label-reveal');

  if (duration > 0) {
    layerGroup
      .transition('manual-layer-reveal')
      .duration(duration)
      .ease(d3.easeCubicOut)
      .style('opacity', visible ? 1 : 0)
      .style('pointer-events', visible ? 'all' : 'none');

    compactLabel
      .transition('manual-label-reveal')
      .duration(duration)
      .style('opacity', visible ? compactTargetOpacity : 0);

    detailedLabel
      .transition('manual-detail-label-reveal')
      .duration(duration)
      .style('opacity', visible ? detailedTargetOpacity : 0);
    return;
  }

  layerGroup
    .style('opacity', visible ? 1 : 0)
    .style('pointer-events', visible ? 'all' : 'none');
  compactLabel.style('opacity', visible ? compactTargetOpacity : 0);
  detailedLabel.style('opacity', visible ? detailedTargetOpacity : 0);
}

const updateRevealLayerHierarchy = () => {
  if (!isStyleTestMode() || !manualRevealState.displayOrder.length) {
    return;
  }

  let activeDisplayIndex = Math.max(0, manualRevealState.nextDisplayIndex - 1);
  let activeLayerIndex = manualRevealState.displayOrder[activeDisplayIndex];
  updateStyleTestLayerHierarchy(activeLayerIndex);
}

const updateRevealControl = (controlGroup) => {
  let hasMore = manualRevealState.nextDisplayIndex < manualRevealState.displayOrder.length;
  let nextLayerIndex = hasMore ?
    manualRevealState.displayOrder[manualRevealState.nextDisplayIndex] : null;
  let nextLabel = hasMore ? cnn[nextLayerIndex][0].layerName : 'complete';
  let shownCount = Math.min(manualRevealState.nextDisplayIndex, manualRevealState.displayOrder.length);
  let progressText = `${shownCount}/${manualRevealState.displayOrder.length}`;
  manualLayerRevealStore.set({
    hasMore,
    nextLabel,
    progressText,
  });

  controlGroup.select('text.reveal-control-main')
    .text(hasMore ? `Show ${nextLabel}` : 'All layers shown')
    .style('font-size', '14px')
    .style('font-weight', 600)
    .style('fill', '#9aa0a6')
    .style('opacity', 1);

  controlGroup.select('text.reveal-control-progress')
    .text(progressText)
    .style('fill', '#9aa0a6')
    .style('opacity', 1);

  controlGroup.select('text.reveal-control-icon')
    .text(hasMore ? '›' : '✓')
    .style('font-size', '16px')
    .style('fill', '#9aa0a6')
    .style('opacity', 1);

  controlGroup.select('rect.reveal-control-bg')
    .attr('rx', 8)
    .attr('ry', 8)
    .style('fill', '#f8f8f8')
    .style('stroke', '#d9d9d9')
    .style('stroke-width', 1);

  controlGroup
    .style('cursor', 'pointer')
    .style('pointer-events', 'all')
    .style('opacity', 1);
}

const resetManualLayerReveal = (cnnGroup, controlGroup) => {
  cnnGroup.select('g.edge-packet-group').remove();
  cnnGroup.select('g.node-pulse-group').remove();
  resetOutputWinnerAnimation();

  cnnGroup.select('g.edge-group').selectAll('path.edge')
    .interrupt('edge-reveal')
    .style('opacity', 0)
    .style('visibility', 'hidden')
    .style('pointer-events', 'none')
    .style('stroke-dasharray', null)
    .style('stroke-dashoffset', null)
    .style('stroke', edgeInitColor)
    .style('stroke-width', edgeStrokeWidth);

  manualRevealState.nextDisplayIndex = 1;

  manualRevealState.displayOrder.forEach((layerIndex, displayIndex) => {
    setLayerRevealVisibility(cnnGroup, layerIndex, displayIndex === 0);
  });
  updateRevealLayerHierarchy();

  cnnGroup.select('g.input-annotation')
    .interrupt()
    .style('opacity', 1);

  applyCurrentEdgeVisibility();
  updateRevealControl(controlGroup);
}

const revealNextLayerStep = (cnnGroup, controlGroup) => {
  if (manualRevealState.nextDisplayIndex >= manualRevealState.displayOrder.length) {
    resetManualLayerReveal(cnnGroup, controlGroup);
    return;
  }

  let displayIndex = manualRevealState.nextDisplayIndex;
  let layerIndex = manualRevealState.displayOrder[displayIndex];

  setLayerRevealVisibility(cnnGroup, layerIndex, true, layerRevealDuration);

  let edgeSelection = cnnGroup.select('g.edge-group')
    .selectAll('path.edge')
    .filter((d) => d.displayTargetLayerIndex === displayIndex);

  animateEdgeRevealForStep(edgeSelection);
  animateEdgePacketsForStep(cnnGroup, edgeSelection);

  if (layerIndex === cnn.length - 1) {
    animateWinningOutputLabel(packetTravelDuration + winnerLabelDelayOffset);
  }

  manualRevealState.nextDisplayIndex += 1;
  updateRevealLayerHierarchy();
  updateRevealControl(controlGroup);
}

export const revealNextOverviewLayer = () => {
  if (!manualRevealState.cnnGroup || !manualRevealState.controlGroup) {
    return;
  }
  revealNextLayerStep(manualRevealState.cnnGroup, manualRevealState.controlGroup);
}

export const canRevealNextOverviewLayer = () =>
  Boolean(
    manualRevealState.cnnGroup &&
    manualRevealState.controlGroup &&
    manualRevealState.nextDisplayIndex < manualRevealState.displayOrder.length,
  );

export const resetOverviewLayerReveal = () => {
  if (!manualRevealState.cnnGroup || !manualRevealState.controlGroup) {
    return;
  }

  resetManualLayerReveal(manualRevealState.cnnGroup, manualRevealState.controlGroup);
}

export const revealOverviewThroughLayer = (layerIndex) => {
  if (!manualRevealState.cnnGroup || !manualRevealState.controlGroup) {
    return;
  }

  let targetDisplayIndex = manualRevealState.displayOrder.indexOf(layerIndex);
  if (targetDisplayIndex < 0) {
    return;
  }

  let nextDisplayIndex = Math.max(
    manualRevealState.nextDisplayIndex,
    targetDisplayIndex + 1,
  );
  for (let displayIndex = 0; displayIndex < nextDisplayIndex; displayIndex++) {
    setLayerRevealVisibility(
      manualRevealState.cnnGroup,
      manualRevealState.displayOrder[displayIndex],
      true,
    );
  }

  manualRevealState.nextDisplayIndex = nextDisplayIndex;
  updateRevealLayerHierarchy();
  applyCurrentEdgeVisibility();
  updateRevealControl(manualRevealState.controlGroup);
}

const initializeManualLayerReveal = (cnnGroup) => {
  let displayOrder = getDisplayOrder();
  manualRevealState = {
    displayOrder,
    nextDisplayIndex: 1,
    cnnGroup,
    controlGroup: undefined,
  };

  cnnGroup.select('g.edge-packet-group').remove();
  cnnGroup.select('g.node-pulse-group').remove();
  resetOutputWinnerAnimation();

  cnnGroup.select('g.edge-group').selectAll('path.edge')
    .interrupt('edge-reveal')
    .style('opacity', 0)
    .style('visibility', 'hidden')
    .style('pointer-events', 'none')
    .style('stroke-dasharray', null)
    .style('stroke-dashoffset', null);

  applyCurrentEdgeVisibility();

  displayOrder.forEach((layerIndex, displayIndex) => {
    setLayerRevealVisibility(cnnGroup, layerIndex, displayIndex === 0);
  });
  updateRevealLayerHierarchy();

  cnnGroup.select('g.input-annotation')
    .style('opacity', 1);

  // The Svelte toolbar owns the visible reveal control. Keep an empty D3
  // selection for the existing reveal-state helpers without drawing a second
  // button inside the CNN SVG.
  let controlGroup = d3.select(null);

  manualRevealState.controlGroup = controlGroup;
  updateRevealControl(controlGroup);
}

const animateOverviewReveal = (cnnGroup) => {
  // Task 3 at lines 922-971: reveal the overview layer-by-layer first, then
  // replay the edge and packet animations after the layers are visible.
  let displayOrder = getDisplayOrder();
  let stageGroups = cnnGroup.select('g.stage-grouping').selectAll('g.stage-group');
  let inputAnnotation = cnnGroup.select('g.input-annotation');
  let totalRevealDelay = Math.max(cnn.length - 1, 0) * layerRevealStagger + layerRevealDuration;

  stageGroups
    .style('opacity', 0)
    .transition('stage-reveal')
    .delay(layerRevealStagger)
    .duration(layerRevealDuration * 2)
    .style('opacity', 1);

  displayOrder.forEach((layerIndex, i) => {
    let layerGroup = cnnGroup.select(`#cnn-layer-group-${layerIndex}`);
    layerGroup
      .style('opacity', 0)
      .transition('layer-reveal')
      .delay(i * layerRevealStagger)
      .duration(layerRevealDuration)
      .ease(d3.easeCubicOut)
      .style('opacity', 1);

    let compactLabel = svg.select(`#layer-label-${layerIndex}`);
    let detailedLabel = svg.select(`#layer-detailed-label-${layerIndex}`);
    let compactTargetOpacity = detailedMode ? 0 : 0.8;
    let detailedTargetOpacity = detailedMode ? 0.7 : 0;

    compactLabel
      .style('opacity', 0)
      .transition('label-reveal')
      .delay(i * layerRevealStagger + 80)
      .duration(layerRevealDuration)
      .style('opacity', compactTargetOpacity);

    detailedLabel
      .style('opacity', 0)
      .transition('detail-label-reveal')
      .delay(i * layerRevealStagger + 80)
      .duration(layerRevealDuration)
      .style('opacity', detailedTargetOpacity);
  });

  if (isStyleTestMode()) {
    updateStyleTestLayerHierarchy(displayOrder[displayOrder.length - 1]);
  }

  inputAnnotation
    .style('opacity', 0)
    .transition('input-annotation-reveal')
    .delay(layerRevealStagger)
    .duration(layerRevealDuration)
    .style('opacity', 1);

  replayEdgeAnimations(cnnGroup, totalRevealDelay);
}

/**
 * Draw the overview
 * @param {number} width Width of the cnn group
 * @param {number} height Height of the cnn group
 * @param {object} cnnGroup Group to appen cnn elements to
 * @param {function} nodeMouseOverHandler Callback func for mouseOver
 * @param {function} nodeMouseLeaveHandler Callback func for mouseLeave
 * @param {function} nodeClickHandler Callback func for click
 */
export const drawCNN = (width, height, cnnGroup, nodeMouseOverHandler,
  nodeMouseLeaveHandler, nodeClickHandler) => {
  // Task 3 at lines 990-1018: compute x positions from the selected display
  // order instead of assuming one fixed historical architecture.
  // Draw the CNN
  // Compute spacing from the actual displayed architecture instead of assuming
  // the original 2-block network.
  // let displayOrder = getDisplayOrder();
  // let totalGapUnits = 0;
  // for (let di = 1; di < displayOrder.length; di++) {
  //   let curLayer = cnn[displayOrder[di]];
  //   let isLongGap = curLayer[0].layerName === 'output' ||
  //     curLayer[0].type === 'conv';
  //   totalGapUnits += isLongGap ? gapRatio : 1;
  // }
  // Task 5 at lines 1020-1025
  let displayOrder = getDisplayOrder();
  let totalGapUnits = 0;
  for (let di = 1; di < displayOrder.length; di++) {
    let curLayer = cnn[displayOrder[di]];
    totalGapUnits += getDisplayGapWeight(curLayer); //totalGapUnits = 1 + 1 + 1 + 2 = 5
  }

  hSpaceAroundGap = (width - nodeLength * cnn.length) / totalGapUnits;
  hSpaceAroundGapStore.set(hSpaceAroundGap);
  let leftByLayerIndex = new Map();
  let leftAccuumulatedSpace = 0;

  for (let di = 0; di < displayOrder.length; di++) {
    let layerIndex = displayOrder[di];
    let curLayer = cnn[layerIndex];
    // let isOutput = curLayer[0].layerName === 'output';

    // if (di > 0) {
    //   if (isOutput || curLayer[0].type === 'conv') {
    //     leftAccuumulatedSpace += hSpaceAroundGap * gapRatio;
    //   } else {
    //     leftAccuumulatedSpace += hSpaceAroundGap;
    //   }
    // }
    // Task 5 at lines 1045-1049
    let isOutput = curLayer[0].layerName === 'output';

    if (di > 0) {
      leftAccuumulatedSpace += hSpaceAroundGap * getDisplayGapWeight(curLayer);
    }

    leftByLayerIndex.set(layerIndex, leftAccuumulatedSpace);
    leftAccuumulatedSpace += nodeLength;
  }

  // Iterate through the cnn to draw nodes in each layer
  for (let l = 0; l < cnn.length; l++) {
    let curLayer = cnn[l];
    let isOutput = curLayer[0].layerName === 'output';

    nodeCoordinate.push([]);

    // All nodes share the same x coordiante (left in div style)
    let left = leftByLayerIndex.get(l);

    let layerGroup = cnnGroup.append('g')
      .attr('class', 'cnn-layer-group')
      .attr('id', `cnn-layer-group-${l}`);

    vSpaceAroundGap = (height - nodeLength * curLayer.length) /
      (curLayer.length + 1);
    vSpaceAroundGapStore.set(vSpaceAroundGap);

    let nodeGroups = layerGroup.selectAll('g.node-group')
      .data(curLayer, d => d.index)
      .enter()
      .append('g')
      .attr('class', 'node-group')
      .style('cursor', 'pointer')
      .style('pointer-events', 'all')
      .on('click', nodeClickHandler)
      .on('mouseover', nodeMouseOverHandler)
      .on('mouseleave', nodeMouseLeaveHandler)
      .classed('node-output', isOutput)
      .attr('id', (d, i) => {
        // Compute the coordinate
        // Not using transform on the group object because of a decade old
        // bug on webkit (safari)
        // https://bugs.webkit.org/show_bug.cgi?id=23113
        let top = i * nodeLength + (i + 1) * vSpaceAroundGap;
        top += svgPaddings.top;
      nodeCoordinate[l].push({ x: left, y: top });
      return `layer-${l}-node-${i}`
    });

    let layerTop = svgPaddings.top - 16;
    let layerHeight = height + 34;
    layerGroup.insert('rect', ':first-child')
      .attr('class', 'layer-hover-outline')
      .attr('x', left - 14)
      .attr('y', layerTop)
      .attr('width', nodeLength + 28)
      .attr('height', layerHeight)
      .attr('rx', 16)
      .attr('ry', 16)
      .style('fill', 'rgba(248, 250, 252, 0.18)')
      .style('stroke', '#94a3b8')
      .style('stroke-width', 1.2)
      .style('stroke-dasharray', '7 5')
      .style('opacity', 0)
      .style('pointer-events', 'none');

    // Overwrite the mouseover and mouseleave function for output nodes to show
    // hover info in the UI
    layerGroup.selectAll('g.node-output')
      .on('mouseover', (d, i, g) => {
        nodeMouseOverHandler(d, i, g);
        hoverInfoStore.set({ show: true, text: `Output value: ${formater(d.output)}` });
      })
      .on('mouseleave', (d, i, g) => {
        nodeMouseLeaveHandler(d, i, g);
        hoverInfoStore.set({ show: false, text: `Output value: ${formater(d.output)}` });
      });

    if (curLayer[0].layerName !== 'output') {
      // Embed raster image in these groups
      nodeGroups.append('image')
        .attr('class', 'node-image')
        .attr('width', nodeLength)
        .attr('height', nodeLength)
        .attr('x', left)
        .attr('y', (d, i) => nodeCoordinate[l][i].y);

      // Add a rectangle to show the border
      nodeGroups.append('rect')
        .attr('class', 'bounding')
        .attr('width', nodeLength)
        .attr('height', nodeLength)
        .attr('x', left)
        .attr('y', (d, i) => nodeCoordinate[l][i].y)
        .style('fill', 'none')
        .style('stroke', 'gray')
        .style('stroke-width', 1)
        .classed('hidden', true);
    } else {
      nodeGroups.append('rect')
        .attr('class', 'output-rect')
        .attr('x', left)
        .attr('y', (d, i) => nodeCoordinate[l][i].y + nodeLength / 2 + 8)
        .attr('height', nodeLength / 4)
        .attr('width', 0)
        .style('fill', '#9AA4B2');
      nodeGroups.append('text')
        .attr('class', 'output-text')
        .attr('x', left)
        .attr('y', (d, i) => nodeCoordinate[l][i].y + nodeLength / 2)
        .style('dominant-baseline', 'middle')
        .style('font-size', '11px')
        .style('fill', 'black')
        .style('opacity', 0.5)
        .text((d, i) => classLists[i]);

      // Add annotation text to tell readers the exact output probability
      // nodeGroups.append('text')
      //   .attr('class', 'annotation-text')
      //   .attr('id', (d, i) => `output-prob-${i}`)
      //   .attr('x', left)
      //   .attr('y', (d, i) => nodeCoordinate[l][i].y + 10)
      //   .text(d => `(${d3.format('.4f')(d.output)})`);
    }
  }

  // Share the nodeCoordinate
  nodeCoordinateStore.set(nodeCoordinate)

  // Compute the scale of the output score width (mapping the the node
  // width to the max output score)
  let outputRectScale = d3.scaleLinear()
    .domain(cnnLayerRanges.output)
    .range([0, nodeLength]);

  // Draw the canvas
  for (let l = 0; l < cnn.length; l++) {
    let range = cnnLayerRanges[selectedScaleLevel][l];
    svg.select(`g#cnn-layer-group-${l}`)
      .selectAll('image.node-image')
      .each((d, i, g) => drawOutput(d, i, g, range));
  }

  svg.selectAll('g.node-output').each(
    (d, i, g) => drawOutputScore(d, i, g, outputRectScale)
  );

  drawStageGroups(cnnGroup, height);

  // Add layer label
  let layerNames = cnn.map(d => {
    if (d[0].layerName === 'output') {
      return {
        name: d[0].layerName,
        dimension: `(${d.length})`
      }
    } else {
      return {
        name: d[0].layerName,
        dimension: d[0].output.length === undefined ?
          `(${d.length})` :
          `(${d[0].output.length}, ${d[0].output.length}, ${d.length})`
      }
    }
  });

  // Task 5 at line 1196-1202
  let getLayerNameParts = (name) => {
    if (isStyleTestMode()) {
      return [formatLayerLabel(name)];
    }

    if (name.includes('max_pool')) {
      return ['max', name.replace('max_', '')];
    }
    if (name.includes('avg_pool')) {
      return ['avg', name.replace('avg_', '')];
    }

    return [name];
  };

  let detailedLabels = svg.selectAll('g.layer-detailed-label')
    .data(layerNames)
    .enter()
    .append('g')
    .attr('class', 'layer-detailed-label')
    .attr('id', (d, i) => `layer-detailed-label-${i}`)
    .classed('hidden', !detailedMode)
    .attr('transform', (d, i) => {
      let x = nodeCoordinate[i][0].x + nodeLength / 2;
      let y = (svgPaddings.top + vSpaceAroundGap) / 2 - 8;
      return `translate(${x}, ${y})`;
    })
    .style('cursor', 'default')
    .on('mouseenter', (d, i) => setLayerHoverState(i, true))
    .on('mouseleave', (d, i) => setLayerHoverState(i, false));

  detailedLabels.insert('rect', ':first-child')
    .attr('class', 'layer-label-highlight')
    .attr('x', -34)
    .attr('y', -15)
    .attr('rx', 3)
    .attr('ry', 3)
    .attr('width', 68)
    .attr('height', 18)
    .style('fill', '#FEF08A')
    .style('opacity', 0)
    .style('pointer-events', 'all');

  let detailedLabelText = detailedLabels.append('text')
    .style('opacity', 0.7)
    .style('dominant-baseline', 'middle');

  detailedLabelText.each(function (d) {
    let text = d3.select(this);
    let nameParts = getLayerNameParts(d.name);
    nameParts.forEach((part, partIndex) => {
      text.append('tspan')
        .style('font-size', '10px')
        .style('font-weight', 800)
        .attr('x', 0)
        .attr('dy', partIndex === 0 ? 0 : '1em')
        .text(part);
    });

    text.append('tspan')
      .style('font-size', '7px')
      .style('font-weight', 'normal')
      .attr('x', 0)
      .attr('dy', '1.2em')
      .text(d.dimension);

  });

  /*
    Previous single-line label version. Kept here as reference because long
    names such as MAX_POOL_1 overlap when many trained-order layers are visible.
    detailedLabels.append('text')
      .style('opacity', 0.7)
      .style('dominant-baseline', 'middle')
      .append('tspan')
      .style('font-size', '12px')
      .text(d => d.name)
      .append('tspan')
    .style('font-size', '8px')
    .style('font-weight', 'normal')
    .attr('x', 0)
    .attr('dy', '1.5em')
    .text(d => d.dimension);
  */

  let labels = svg.selectAll('g.layer-label')
    .data(layerNames)
    .enter()
    .append('g')
    .attr('class', 'layer-label')
    .attr('id', (d, i) => `layer-label-${i}`)
    .classed('hidden', detailedMode)
    .attr('transform', (d, i) => {
      let x = nodeCoordinate[i][0].x + nodeLength / 2;
      let y = (svgPaddings.top + vSpaceAroundGap) / 2 + 5;
      return `translate(${x}, ${y})`;
    })
    .style('cursor', 'default')
    .on('mouseenter', (d, i) => setLayerHoverState(i, true))
    .on('mouseleave', (d, i) => setLayerHoverState(i, false));

  labels.insert('rect', ':first-child')
    .attr('class', 'layer-label-highlight')
    .attr('x', -30)
    .attr('y', -12)
    .attr('rx', 3)
    .attr('ry', 3)
    .attr('width', 60)
    .attr('height', 18)
    .style('fill', '#FEF08A')
    .style('opacity', 0)
    .style('pointer-events', 'all');

  labels.append('text')
    .style('dominant-baseline', 'middle')
    .style('opacity', 0.8)
    .text(d => {
      if (isStyleTestMode()) { return formatLayerLabel(d.name) }
      if (d.name.includes('conv')) { return 'conv' }
      if (d.name.includes('relu')) { return 'relu' }
      if (d.name.includes('sigmoid')) { return 'sigmoid' } //Task 6.2
      if (d.name.includes('avg_pool')) { return 'avg_pool' }
      if (d.name.includes('max_pool')) { return 'pool' }
      return d.name
    });

  // Add layer color scale legends
  getLegendGradient(svg, layerColorScales.conv, 'convGradient');
  getLegendGradient(svg, layerColorScales.input[0], 'inputGradient');

  let legendHeight = 5;
  let legends = svg.append('g')
    .attr('class', 'color-legend')
    .attr('transform', `translate(${0}, ${svgPaddings.top + vSpaceAroundGap * (10) + vSpaceAroundGap +
      nodeLength * 10
      })`);

  drawLegends(legends, legendHeight);

  // Add edges between nodes
  let linkGen = d3.linkHorizontal()
    .x(d => d.x)
    .y(d => d.y);
  //Task 4 at lines 1322-1333: compute the link data based on the actual displayed architecture and the node coordinates, instead of assuming a fixed architecture and coordinate pattern.
  let displayIndexLookup = getDisplayLayerIndexLookup();
  let isModelDisplayOrder = displayOrder.every((layerIndex, index) => layerIndex === index);
  let rawLinkData = isModelDisplayOrder ?
    getLinkData(nodeCoordinate, cnn) :
    buildDisplayOrderLinkData(nodeCoordinate, cnn, displayOrder);
  let linkData = rawLinkData.map((link) => ({
    ...link,
    sourceLayerIndex: link.sourceLayerIndex ?? (link.targetLayerIndex - 1),
    targetLayerType: link.targetLayerType ?? cnn[link.targetLayerIndex][0].type,
    displaySourceLayerIndex: link.displaySourceLayerIndex ?? displayIndexLookup.get(link.sourceLayerIndex ?? (link.targetLayerIndex - 1)),
    displayTargetLayerIndex: link.displayTargetLayerIndex ?? displayIndexLookup.get(link.targetLayerIndex),
  }));

  let edgeGroup = cnnGroup.append('g')
    .attr('class', 'edge-group');

  let edges = edgeGroup.selectAll('path.edge')
    .data(linkData)
    .enter()
    .append('path')
    .attr('class', d =>
      `edge edge-${d.targetLayerIndex} edge-${d.targetLayerIndex}-${d.targetNodeIndex}`)
    .attr('id', d =>
      `edge-${d.targetLayerIndex}-${d.targetNodeIndex}-${d.sourceNodeIndex}`)
    .attr('d', d => linkGen({ source: d.source, target: d.target }))
    .style('fill', 'none')
    .style('stroke-width', edgeStrokeWidth)
    .style('opacity', 0)
    .style('visibility', 'hidden')
    .style('pointer-events', 'none')
    .style('stroke', edgeInitColor);

  // Add input channel annotations
  let inputAnnotation = cnnGroup.append('g')
    .attr('class', 'input-annotation');

  // PYTORCH_BACKEND_INTEGRATION:
  // The old TensorFlow.js demos use RGB inputs. PyTorch MNIST models use one
  // grayscale channel, so the active code below adapts to the actual number of
  // input nodes. The original RGB-only code is preserved here for reference:
  /*
  let redChannel = inputAnnotation.append('text')
    .attr('x', nodeCoordinate[0][0].x + nodeLength / 2)
    .attr('y', nodeCoordinate[0][0].y + nodeLength + 5)
    .attr('class', 'annotation-text')
    .style('dominant-baseline', 'hanging')
    .style('text-anchor', 'middle');

  redChannel.append('tspan')
    .style('dominant-baseline', 'hanging')
    .style('fill', '#C95E67')
    .text('Red');

  redChannel.append('tspan')
    .style('dominant-baseline', 'hanging')
    .text(' channel');

  inputAnnotation.append('text')
    .attr('x', nodeCoordinate[0][1].x + nodeLength / 2)
    .attr('y', nodeCoordinate[0][1].y + nodeLength + 5)
    .attr('class', 'annotation-text')
    .style('dominant-baseline', 'hanging')
    .style('text-anchor', 'middle')
    .style('fill', '#3DB665')
    .text('Green');

  inputAnnotation.append('text')
    .attr('x', nodeCoordinate[0][2].x + nodeLength / 2)
    .attr('y', nodeCoordinate[0][2].y + nodeLength + 5)
    .attr('class', 'annotation-text')
    .style('dominant-baseline', 'hanging')
    .style('text-anchor', 'middle')
    .style('fill', '#3F7FBC')
    .text('Blue');
  */
  let inputChannelLabels = cnn[0].length === 1
    ? [{ text: 'Input', color: '#475569' }]
    : [
      { text: 'Red channel', color: '#C95E67' },
      { text: 'Green', color: '#3DB665' },
      { text: 'Blue', color: '#3F7FBC' },
    ];

  inputChannelLabels.slice(0, cnn[0].length).forEach((label, channelIndex) => {
    inputAnnotation.append('text')
      .attr('x', nodeCoordinate[0][channelIndex].x + nodeLength / 2)
      .attr('y', nodeCoordinate[0][channelIndex].y + nodeLength + 5)
      .attr('class', 'annotation-text')
      .style('dominant-baseline', 'hanging')
      .style('text-anchor', 'middle')
      .style('fill', label.color)
      .text(label.text);
  });

  initializeManualLayerReveal(cnnGroup);
}

/**
 * Update canvas values when user changes input image
 */
export const updateCNN = () => {
  // Compute the scale of the output score width (mapping the the node
  // width to the max output score)
  let outputRectScale = d3.scaleLinear()
    .domain(cnnLayerRanges.output)
    .range([0, nodeLength]);

  // Rebind the cnn data to layer groups layer by layer
  for (let l = 0; l < cnn.length; l++) {
    let curLayer = cnn[l];
    let range = cnnLayerRanges[selectedScaleLevel][l];
    let layerGroup = svg.select(`g#cnn-layer-group-${l}`);

    let nodeGroups = layerGroup.selectAll('g.node-group')
      .data(curLayer);

    if (l < cnn.length - 1) {
      // Redraw the canvas and output node
      nodeGroups.transition('disappear')
        .duration(300)
        .ease(d3.easeCubicOut)
        .style('opacity', 0)
        .on('end', function () {
          d3.select(this)
            .select('image.node-image')
            .each((d, i, g) => drawOutput(d, i, g, range));
          d3.select(this).transition('appear')
            .duration(700)
            .ease(d3.easeCubicIn)
            .style('opacity', 1);
        });
    } else {
      nodeGroups.each(
        (d, i, g) => drawOutputScore(d, i, g, outputRectScale)
      );
    }
  }

  let visibleLayerCount = cnn.length || numLayers;
  let numOfComponent = Math.max(1, Math.ceil((visibleLayerCount - 2) / 5));

  // Update the color scale legend
  // Local legends
  for (let i = 0; i < numOfComponent; i++) {
    let start = 1 + i * 5;
    let range1 = cnnLayerRanges.local[start];
    let range2 = cnnLayerRanges.local[start + 2];

    let localLegendScale1 = d3.scaleLinear()
      .range([0, 2 * nodeLength + hSpaceAroundGap])
      .domain([-range1 / 2, range1 / 2]);

    let localLegendScale2 = d3.scaleLinear()
      .range([0, 3 * nodeLength + 2 * hSpaceAroundGap])
      .domain([-range2 / 2, range2 / 2]);

    let localLegendAxis1 = d3.axisBottom()
      .scale(localLegendScale1)
      .tickFormat(d3.format('.2f'))
      .tickValues([-range1 / 2, 0, range1 / 2]);

    let localLegendAxis2 = d3.axisBottom()
      .scale(localLegendScale2)
      .tickFormat(d3.format('.2f'))
      .tickValues([-range2 / 2, 0, range2 / 2]);

    svg.select(`g#local-legend-${i}-1`).select('g').call(localLegendAxis1);
    svg.select(`g#local-legend-${i}-2`).select('g').call(localLegendAxis2);
  }

  // Module legend
  for (let i = 0; i < numOfComponent; i++) {
    let start = 1 + i * 5;
    let range = cnnLayerRanges.local[start];

    let moduleLegendScale = d3.scaleLinear()
      .range([0, 5 * nodeLength + 3 * hSpaceAroundGap +
        1 * hSpaceAroundGap * gapRatio - 1.2])
      .domain([-range, range]);

    let moduleLegendAxis = d3.axisBottom()
      .scale(moduleLegendScale)
      .tickFormat(d3.format('.2f'))
      .tickValues([-range, -(range / 2), 0, range / 2, range]);

    svg.select(`g#module-legend-${i}`).select('g').call(moduleLegendAxis);
  }

  // Global legend
  let start = 1;
  let range = cnnLayerRanges.global[start];

  let globalLegendScale = d3.scaleLinear()
    .range([0, 10 * nodeLength + 6 * hSpaceAroundGap +
      3 * hSpaceAroundGap * gapRatio - 1.2])
    .domain([-range, range]);

  let globalLegendAxis = d3.axisBottom()
    .scale(globalLegendScale)
    .tickFormat(d3.format('.2f'))
    .tickValues([-range, -(range / 2), 0, range / 2, range]);

  svg.select(`g#global-legend`).select('g').call(globalLegendAxis);

  // Output legend
  let outputLegendAxis = d3.axisBottom()
    .scale(outputRectScale)
    .tickFormat(d3.format('.1f'))
    .tickValues([0, cnnLayerRanges.output[1]]);

  svg.select('g#output-legend').select('g').call(outputLegendAxis);

  let cnnGroup = svg.select('g.cnn-group');
  cnnGroup.selectAll('g.cnn-layer-group')
    .interrupt()
    .style('opacity', 0);
  cnnGroup.selectAll('g.stage-grouping g.stage-group')
    .interrupt()
    .style('opacity', 0);
  svg.selectAll('g.layer-label, g.layer-detailed-label')
    .interrupt()
    .style('opacity', 0);
  cnnGroup.select('g.input-annotation')
    .interrupt()
    .style('opacity', 0);

  initializeManualLayerReveal(cnnGroup);
}

/**
 * Update the ranges for current CNN layers
 */
export const updateCNNLayerRanges = () => {
  // Iterate through all nodes to find a output ranges for each layer
  let cnnLayerRangesLocal = [1];
  let curRange = undefined;

  // Also track the min/max of each layer (avoid computing during intermediate
  // layer)
  cnnLayerMinMax = [];

  for (let l = 0; l < cnn.length - 1; l++) {
    let curLayer = cnn[l];

    // Compute the min max
    let outputExtents = curLayer.map(l => getExtent(l.output));
    let aggregatedExtent = outputExtents.reduce((acc, cur) => {
      return [Math.min(acc[0], cur[0]), Math.max(acc[1], cur[1])];
    })
    cnnLayerMinMax.push({ min: aggregatedExtent[0], max: aggregatedExtent[1] });

    // conv layer refreshes curRange counting
    if (curLayer[0].type === 'conv' || curLayer[0].type === 'fc') {
      aggregatedExtent = aggregatedExtent.map(Math.abs);
      // Plus 0.1 to offset the rounding error (avoid black color)
      curRange = 2 * (0.1 +
        Math.round(Math.max(...aggregatedExtent) * 1000) / 1000);
    }

    if (curRange !== undefined) {
      cnnLayerRangesLocal.push(curRange);
    }
  }

  // Finally, add the output layer range
  cnnLayerRangesLocal.push(1);
  cnnLayerMinMax.push({ min: 0, max: 1 });

  // Support different levels of scales (1) lcoal, (2) component, (3) global
  let visibleLayerCount = cnn.length || numLayers;
  let cnnLayerRangesComponent = [1];
  let numOfComponent = Math.max(1, Math.ceil((visibleLayerCount - 2) / 5));
  for (let i = 0; i < numOfComponent; i++) {
    let curArray = cnnLayerRangesLocal.slice(1 + 5 * i, 1 + 5 * i + 5);
    let maxRange = Math.max(...curArray);
    for (let j = 0; j < 5; j++) {
      cnnLayerRangesComponent.push(maxRange);
    }
  }
  cnnLayerRangesComponent.push(1);

  let cnnLayerRangesGlobal = [1];
  let maxRange = Math.max(...cnnLayerRangesLocal.slice(1,
    cnnLayerRangesLocal.length - 1));
  for (let i = 0; i < visibleLayerCount - 2; i++) {
    cnnLayerRangesGlobal.push(maxRange);
  }
  cnnLayerRangesGlobal.push(1);

  // Update the ranges dictionary
  cnnLayerRanges.local = cnnLayerRangesLocal;
  cnnLayerRanges.module = cnnLayerRangesComponent;
  cnnLayerRanges.global = cnnLayerRangesGlobal;
  cnnLayerRanges.output = [0, d3.max(cnn[cnn.length - 1].map(d => d.output))];

  cnnLayerRangesStore.set(cnnLayerRanges);
  cnnLayerMinMaxStore.set(cnnLayerMinMax);
}
