<script>
  import { tick } from "svelte";

  export let images = [];
  export let selected = "";
  export let customURL = null;
  export let disabled = false;
  let open = false;
  let chosen = false;
  let root;
  let trigger;
  $: current = images.find((image) => image.file === selected);
  $: source = selected === "custom" ? customURL : current?.src || `assets/img/${selected}`;

  async function close() {
    open = false;
    await tick();
    trigger?.focus();
  }

  function choose(event) {
    if (disabled || !event.target.closest(".image-container")) return;
    chosen = true;
    close();
  }

  function outside(event) {
    if (open && !root?.contains(event.target)) open = false;
  }

  function selectionEvents(node) {
    const keydown = (event) => {
      if (event.key === "Enter" || event.key === " ") choose(event);
    };
    node.addEventListener("click", choose);
    node.addEventListener("keydown", keydown);
    return { destroy() {
      node.removeEventListener("click", choose);
      node.removeEventListener("keydown", keydown);
    } };
  }
</script>

<svelte:window on:pointerdown={outside} on:keydown={(event) => {
  if (open && event.key === "Escape") { event.preventDefault(); close(); }
}} />

<div class="bubble-picker" bind:this={root}>
  <button
    bind:this={trigger}
    class="bubble-trigger"
    class:stacked={!chosen}
    {disabled}
    aria-label={`Select an image. Current: ${current?.class || "custom image"}`}
    aria-expanded={open}
    title="Choose input image"
    on:click={() => open = !open}
  >
    <span class="bubble-art" aria-hidden="true">
    {#if !chosen}
      {#each images.filter((image) => image.file !== selected).slice(0, 2).reverse() as image, i}
        <img class="stack-image" style={`--stack: ${2 - i}`} src={image.src || `assets/img/${image.file}`} alt="" />
      {/each}
    {/if}
    <img class="current-image" src={source} alt="" />
    <span class="picker-indicator" aria-hidden="true"><i class={`fas fa-${open ? "chevron-up" : "chevron-down"}`}></i></span>
    </span>
    <span class="picker-label">Select an image</span>
  </button>
  {#if open}
    <div class="bubble-options" role="group" aria-label="Input images" use:selectionEvents>
      <slot />
    </div>
  {/if}
</div>

<style>
  .bubble-picker { position: relative; flex: 0 0 auto; height: 68px; display: flex; align-items: center; z-index: 20; }
  .bubble-trigger { display: flex; align-items: center; gap: 8px; height: 64px; padding: 0 8px 0 0; border: 0; background: transparent; cursor: pointer; border-radius: 8px; font: inherit; color: var(--control-text); }
  .bubble-art { position: relative; display: block; width: 68px; height: 64px; flex: 0 0 auto; }
  .bubble-trigger.stacked .bubble-art { width: 104px; }
  .picker-label { font-size: 14px; font-weight: 500; white-space: nowrap; }
  .bubble-trigger img { position: absolute; width: 48px; height: 48px; object-fit: cover; border-radius: 50%; top: 8px; left: 7px; box-shadow: 0 0 0 3px var(--page-canvas), 0 3px 8px #0002; }
  .bubble-trigger .current-image { outline: 2px solid var(--primary-border); outline-offset: 3px; }
  .bubble-trigger .stack-image { transform: translateX(calc(var(--stack) * 19px)) scale(calc(1 - var(--stack) * .09)); }
  .picker-indicator { position: absolute; left: 44px; bottom: 3px; width: 19px; height: 19px; display: grid; place-items: center; border-radius: 50%; background: var(--page-canvas); color: var(--primary-border); font-size: 10px; box-shadow: 0 0 0 1px #0002; }
  .bubble-trigger:focus-visible { outline: 2px solid var(--primary-border); outline-offset: 4px; }
  .bubble-trigger:disabled { opacity: .5; cursor: wait; }
  .bubble-options { position: absolute; top: 72px; left: 0; display: flex; flex-wrap: wrap; gap: 8px; width: max-content; max-width: min(760px, calc(100vw - 48px)); box-sizing: border-box; padding: 14px; border: 1px solid #d3ccc1; border-radius: 10px; background: #fffaf0; box-shadow: 0 10px 28px rgba(48, 41, 37, 0.12), 0 2px 6px rgba(48, 41, 37, 0.08); }
  .bubble-options :global(.image-container) { animation: bubble-arrive 280ms cubic-bezier(.2,.8,.2,1) both; animation-delay: calc(var(--bubble-order, 0) * 28ms); }
  @keyframes bubble-arrive { from { opacity: 0; transform: translateY(-10px) scale(.65); } to { opacity: 1; transform: translateY(0) scale(1); } }
  @media (prefers-reduced-motion: reduce) { .bubble-options :global(.image-container) { animation: none; } }
</style>
