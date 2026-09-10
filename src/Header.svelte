<script>
  import { createEventDispatcher } from "svelte";
  const dispatch = createEventDispatcher();
</script>

<header class="app-header">
  <div class="top-bar">
    <div class="brand-block">
      <div class="brand-row">
        <button class="brand-mark" aria-label="CNN LENS home" title="Return to welcome" on:click={() => dispatch("home")}>CNN LENS</button>
      </div>
      <div class="brand-subtitle">
        Explore trained convolutional networks, layer by layer.
      </div>
    </div>
  </div>
</header>

<style>
  :global(*) {
    box-sizing: border-box;
  }

  .app-header {
    width: 100%;
    background: var(--page-chrome);
    font-family: "Public Sans", -apple-system, BlinkMacSystemFont, "Segoe UI",
      sans-serif;
  }
  .top-bar {
    padding: 16px 24px;
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 20px;
  }
  .brand-block {
    display: flex;
    flex-direction: column;
    gap: 4px;
    min-width: 0;
  }
  .brand-row {
    display: flex;
    align-items: baseline;
    gap: 16px;
    flex-wrap: wrap;
    margin-bottom: 2px;
  }
  .brand-mark {
    padding: 0;
    border: 0;
    background: transparent;
    cursor: pointer;
    font-family: "Public Sans", -apple-system, BlinkMacSystemFont, "Segoe UI",
      sans-serif;
    font-size: 22px;
    font-weight: 800;
    line-height: 1;
    color: var(--control-text);
    letter-spacing: 0;
  }
  .brand-mark:focus-visible { outline: 2px solid var(--primary-border); outline-offset: 5px; }
  .brand-tag {
    color: #93c5fd; /* Soft blue */
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
  }
  .brand-subtitle {
    color: #766f68;
    font-size: 15px;
    font-weight: 400;
    line-height: 1.3;
  }
  @media (max-width: 768px) {
    .top-bar {
      flex-direction: column;
      align-items: flex-start;
    }
  }
</style>
