<script>
  import { createEventDispatcher } from "svelte";
  export let intro = true;
  export let active = false;
  export let busy = false;
  export let title = "Choose an image";
  export let description = "Start with a sample image. Next selects espresso and reveals its red, green, and blue channels.";
  export let progress = 0;
  export let total = 1;
  export let finished = false;
  const dispatch = createEventDispatcher();
</script>

{#if intro}
  <section class="welcome" aria-label="Welcome to CNN Lens">
    <div class="introduction">
      <div class="journey" aria-hidden="true">
        <img src={`${import.meta.env.BASE_URL}assets/img/sample_val_1471.JPEG`} alt="" />
        <span class="flow"><i class="fas fa-arrow-right"></i></span>
        <span class="network"><i class="fas fa-microchip"></i></span>
        <span class="flow"><i class="fas fa-arrow-right"></i></span>
        <span class="prediction">Espresso<span class="prediction-bar"></span></span>
      </div>
      <h1>See how a neural network sees.</h1>
      <div class="actions"><button class="primary" disabled={busy} on:click={() => dispatch("start")}>Take a guided tour <i class="fas fa-arrow-right"></i></button><button disabled={busy} on:click={() => dispatch("skip")}>Explore on my own</button></div>
    </div>
  </section>
{:else if active}
  <aside class="helper" aria-label="Guided tour">
    <button class="close" aria-label="Close guided tour" title="Close guided tour" on:click={() => dispatch("skip")}><i class="fas fa-times"></i></button>
    <span class="eyebrow">CNN Lens / Guided tour</span>
    <h2>{title}</h2>
    <p aria-live="polite">{description}</p>
    <footer><span class="count">{progress + 1} / {total}</span><progress value={progress + 1} max={total} aria-label="Tour progress"></progress><button class="primary" disabled={busy} on:click={() => dispatch("next")}>{busy ? "Working..." : finished ? "Finish" : "Next"} <i class="fas fa-arrow-right"></i></button></footer>
  </aside>
{/if}

<style>
  .welcome { position: absolute; inset: 0; z-index: 25; display: grid; place-items: center; background: #fffcf0; border-radius: inherit; padding: 24px; overflow: auto; }
  .introduction { text-align: center; max-width: 620px; }
  h1 { font-size: 24px; line-height: 1.2; color: #36312d; margin: 26px 0 14px; letter-spacing: 0; }
  p { color: #71675f; font-size: 13px; line-height: 1.6; margin: 0; }
  .journey { display: flex; align-items: center; justify-content: center; gap: 22px; min-height: 90px; }
  .journey img { width: 72px; height: 72px; border-radius: 50%; object-fit: cover; }
  .network { font-size: 64px; color: #7bcde8; animation: breathe 2.4s ease-in-out infinite; }
  .flow { color: #b8afa5; animation: breathe 2.4s ease-in-out infinite; }
  .flow:nth-of-type(3) { animation-delay: .6s; }
  .prediction { display: grid; gap: 10px; color: #514c47; font-size: 14px; }
  .prediction-bar { width: 72px; height: 6px; border-radius: 3px; background: #89bfab; transform-origin: left; animation: confidence 2.4s ease-in-out infinite; }
  .actions { display: flex; justify-content: center; flex-wrap: wrap; gap: 12px; margin-top: 28px; }
  button { font: inherit; font-size: 13px; color: #30221e; background: #fffcf0; border: 1px solid #c5beb5; border-radius: 8px; min-height: 36px; padding: 8px 12px; cursor: pointer; box-shadow: 0 4px 0 #d4cdc4; }
  button.primary { background: #7aceff; border-color: #619fbd; box-shadow: 0 4px 0 #61a5cc; }
  button:not(.close):active:not(:disabled) { transform: translateY(3px); box-shadow: 0 1px 0 #d4cdc4; }
  button.primary:active:not(:disabled) { box-shadow: 0 1px 0 #61a5cc; }
  button:disabled { opacity: .5; cursor: wait; }
  button:focus-visible { outline: 3px solid #82cbea; outline-offset: 3px; }
  .helper { position: absolute; z-index: 40; right: 18px; bottom: 18px; width: 320px; max-width: calc(100% - 36px); box-sizing: border-box; padding: 18px; background: #fffaf0; border: 1px solid #d3ccc1; border-radius: 10px; box-shadow: 0 10px 28px #3029251f, 0 2px 6px #30292514; }
  .close { position: absolute; top: 10px; right: 10px; border: 0; box-shadow: none; background: transparent; min-height: 28px; width: 28px; padding: 0; }
  .eyebrow, .count { color: #80766c; font-size: 11px; }
  h2 { color: #36312d; font-size: 17px; margin: 10px 24px 8px 0; }
  .helper p { font-size: 13px; }
  footer { display: flex; align-items: center; gap: 12px; margin-top: 18px; }
  progress { width: 60px; height: 4px; accent-color: #65b9d9; margin-right: auto; }
  @keyframes breathe { 0%,100% { transform: translateY(0); opacity: .65; } 50% { transform: translateY(-5px); opacity: 1; } }
  @keyframes confidence { 0%,100% { transform: scaleX(.65); } 50% { transform: scaleX(1); } }
  @media (max-width: 600px) { h1 { font-size: 21px; } .journey { gap: 12px; } .journey img { width: 52px; height: 52px; } .network { font-size: 48px; } .helper { bottom: 10px; right: 10px; max-width: calc(100% - 20px); } }
  @media (prefers-reduced-motion: reduce) { .network, .flow, .prediction-bar { animation: none; } }
</style>
