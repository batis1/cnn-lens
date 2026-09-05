import * as d3 from 'd3';
import * as tf from '@tensorflow/tfjs';

// LOCAL_DEV_RUNTIME_BOOTSTRAP:
// The original project reads d3/tf as browser globals from CDN scripts.
// Keep that API, but provide local npm-backed globals so the app also works
// when those CDN scripts are blocked or unavailable.
globalThis.d3 = d3;
globalThis.tf = tf;
globalThis.SmoothScroll = globalThis.SmoothScroll || class {
	animateScroll(anchor) {
		anchor?.scrollIntoView?.({ behavior: 'smooth', block: 'start' });
	}
};

let app;

import('./App.svelte').then(({ default: App }) => {
	app = new App({
		target: document.body,
		props: {}
	});
});

export default app;
