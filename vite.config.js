import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
  base: process.env.VERCEL ? '/' : '/CNN-Visualizer/',
  plugins: [svelte()]
});
