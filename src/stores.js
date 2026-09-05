import { writable } from 'svelte/store';

export const cnnStore = writable([]);
export const svgStore = writable(undefined);

export const vSpaceAroundGapStore = writable(undefined);
export const hSpaceAroundGapStore = writable(undefined);

export const nodeCoordinateStore = writable([]);
export const selectedScaleLevelStore = writable(undefined);
// Task 3 at line 13: stores the user's chosen visual layer order so the same
// trained network can be redrawn in a different order without changing the model.
export const layerDisplayOrderStore = writable('model');
export const manualLayerRevealStore = writable({
  hasMore: false,
  nextLabel: '',
  progressText: '',
});

export const cnnLayerRangesStore = writable({});
export const cnnLayerMinMaxStore = writable([]);

export const needRedrawStore = writable([undefined, undefined]);

export const detailedModeStore = writable(true);

export const shouldIntermediateAnimateStore = writable(false);

export const isInSoftmaxStore = writable(false);
export const softmaxDetailViewStore = writable({});
export const denseDetailViewStore = writable({});
export const allowsSoftmaxAnimationStore = writable(false);

// PYTORCH_BACKEND_INTEGRATION:
// Keep an explicit empty text value so the overview toolbar does not render
// "undefined" before the first hover event. This is runtime-neutral; no
// TensorFlow.js fallback code needs to be restored here.
// Previous default:
// export const hoverInfoStore = writable({});
export const hoverInfoStore = writable({ show: false, text: '' });

export const modalStore = writable({});

export const intermediateLayerPositionStore = writable({});
