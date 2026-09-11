/* global d3 */

const layerColorScales = {
  input: [d3.interpolateGreys, d3.interpolateGreys, d3.interpolateGreys],
  conv: d3.interpolateRdBu,
  relu: d3.interpolateRdBu,
  pool: d3.interpolateRdBu,
  fc: d3.interpolateGreys,
  weight: d3.interpolateBrBG,
  logit: d3.interpolateOranges
};

let nodeLength = 40;

export const overviewConfig = {
  nodeLength: nodeLength,
  plusSymbolRadius: nodeLength / 5,
  // Task 3 at line 20: keep a safe maximum layout default while the real
  // visible layer count is now derived at runtime from the selected model.
  numLayers: 17,
  edgeOpacity: 0.8,
  edgeInitColor: 'rgb(230, 230, 230)',
  edgeHoverColor: 'rgb(130, 130, 130)',
  edgeHoverOuting: false,
  edgeStrokeWidth: 0.9,
  intermediateColor: 'gray',
  layerColorScales: layerColorScales,
  svgPaddings: { top: 25, bottom: 25, left: 50, right: 50 },
  kernelRectLength: 8 / 3,
  gapRatio: 4,
  overlayRectOffset: 12,
  // Task 3 at lines 33-34: shared class labels reused by all 7-layer, 12-layer,
  // and 17-layer demo models when rendering the output layer.
  classLists: ['goldfish', 'tabby cat', 'German shepherd', 'monarch butterfly',
    'banana', 'pomegranate', 'bullet train', 'lighthouse', 'sunglasses',
    'refrigerator']
};
