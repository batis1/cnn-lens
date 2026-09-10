<script>
  export let count = 0;
  export let summary = "";
  export let kernels = "";
  const labels = { Conv2d: "Convolution", AvgPool2d: "Average pool", MaxPool2d: "Max pool", Linear: "Dense" };
  $: operations = summary.split(/\s*->\s*/).filter(Boolean);
</script>

<details class="architecture">
  <summary>
    <span class="count">{operations.length || count} layers</span>
    <span class="overview">Feature extraction <span aria-hidden="true">→</span> Classification</span>
    <span class="toggle"><span class="toggle-label">Architecture</span><i class="fas fa-chevron-down" aria-hidden="true"></i></span>
  </summary>
  <ol aria-label="Model operations">
    {#each operations as operation, i}
      <li class:pool={operation.includes("Pool")} class:classifier={operation === "Flatten" || operation === "Linear"}>
        <span class="number">{i + 1}</span><span title={operation}>{labels[operation] || operation}</span>
      </li>
    {/each}
  </ol>
  {#if kernels}<p>{kernels}</p>{/if}
</details>

<style>
  .architecture { color: #71675f; font-size: 12px; width: 100%; }
  summary { display: flex; align-items: center; flex-wrap: wrap; gap: 10px 16px; cursor: pointer; list-style: none; min-height: 30px; border-radius: 4px; }
  summary::-webkit-details-marker { display: none; }
  summary:focus-visible { outline: 2px solid #528ca7; outline-offset: 3px; }
  .count { color: #38332e; font-weight: 600; padding-right: 16px; border-right: 1px solid #d3ccc1; }
  .overview { display: flex; gap: 12px; align-items: center; }
  .overview > span { color: #aaa198; }
  .toggle { display: flex; gap: 8px; align-items: center; margin-left: auto; color: #71675f; }
  .toggle :global(svg) { width: 10px; transition: transform 150ms ease; }
  details[open] .toggle :global(svg) { transform: rotate(180deg); }
  ol { display: flex; flex-wrap: wrap; gap: 12px 20px; list-style: none; padding: 16px 0 10px; margin: 6px 0 0; border-top: 1px solid #d3ccc1; }
  li { display: flex; gap: 7px; align-items: center; min-height: 26px; color: #514c47; }
  .number { width: 24px; min-width: 24px; height: 24px; flex: 0 0 24px; display: grid; place-items: center; white-space: nowrap; overflow-wrap: normal; border-radius: 50%; background: #dceef4; color: #426575; font-size: 10px; }
  .pool .number { background: #e1eadc; color: #52654b; }
  .classifier .number { background: #eee2d8; color: #796152; }
  p { margin: 8px 0 0; font-size: 11px; }
  @media (max-width: 600px) { summary { gap: 8px; } .count { padding-right: 8px; } .overview { font-size: 11px; gap: 6px; } .toggle-label { display: none; } ol { gap: 8px 14px; } }
  @media (prefers-reduced-motion: reduce) { .toggle :global(svg) { transition: none; } }
</style>
