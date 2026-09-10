<script>
  import { createEventDispatcher, tick } from "svelte";
  export let disabled = false;
  let open = false;
  let root;
  let trigger;
  let restart;
  const dispatch = createEventDispatcher();
  async function close() { open = false; await tick(); trigger?.focus(); }
  async function toggle() { open = !open; await tick(); if (open) restart?.focus(); }
  $: if (disabled) open = false;
</script>

<svelte:window on:pointerdown={(event) => { if (open && !root?.contains(event.target)) open = false; }} on:keydown={(event) => { if (open && event.key === "Escape") { event.preventDefault(); close(); } }} />
<div class="reveal-split" bind:this={root}>
  <slot />
  <button class="toggle" bind:this={trigger} {disabled} aria-label="Visualization actions" aria-expanded={open} title="Visualization actions" on:click={toggle}><i class="fas fa-chevron-down" aria-hidden="true"></i></button>
  {#if open}
    <div class="actions">
      <button bind:this={restart} on:click={() => { dispatch("restart"); close(); }}><i class="fas fa-rotate-left" aria-hidden="true"></i> Restart visualization</button>
    </div>
  {/if}
</div>

<style>
  .reveal-split { display: flex; position: relative; flex: 0 0 auto; }
  .reveal-split :global(.reveal-layer-btn) { border-radius: 8px 0 0 8px; }
  .toggle { width: 30px; height: 36px; padding: 0; border: 1px solid var(--primary-border); border-left: 0; border-radius: 0 8px 8px 0; background: var(--primary-surface); color: var(--primary-text); box-shadow: 0 3px 0 var(--primary-shadow); cursor: pointer; font-size: 10px; }
  .toggle:hover:not(:disabled) { background: var(--primary-surface-hover); }
  .toggle:disabled { opacity: .5; cursor: not-allowed; }
  button:focus-visible { outline: 2px solid var(--primary-border); outline-offset: 2px; }
  .actions { position: absolute; top: calc(100% + 8px); right: 0; z-index: 100; padding: 4px; width: max-content; background: #fffaf0; border: 1px solid #d3ccc1; border-radius: 10px; box-shadow: 0 10px 28px #3029251f, 0 2px 6px #30292514; }
  .actions button { display: flex; gap: 10px; align-items: center; min-height: 36px; padding: 8px 12px; border: 0; border-radius: 6px; background: transparent; color: var(--control-text); font: inherit; font-size: 13px; cursor: pointer; }
  .actions button:hover, .actions button:focus-visible { background: #f0ece4; }
</style>
