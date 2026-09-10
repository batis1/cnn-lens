<div class="loading-stage" role="status" aria-live="polite" aria-label="Loading visualization">
  <div class="loading-content">
    <div class="character" aria-hidden="true">
      <div class="chip">
        <span class="chip-icon"><i class="fas fa-microchip"></i></span>
        <div class="face"><span class="eye"></span><span class="eye"></span><span class="smile"></span></div>
        <div class="neurons"><span></span><span></span><span></span></div>
      </div>
      <div class="shadow"></div>
    </div>
    <p>Loading...</p>
  </div>
</div>

<style>
  .loading-stage { position: absolute; inset: 0; z-index: 30; display: grid; place-items: center; background: #fffdf0; border-radius: inherit; cursor: progress; }
  .loading-content { display: flex; flex-direction: column; align-items: center; gap: 24px; }
  .character { width: 112px; height: 112px; position: relative; }
  .chip { position: absolute; top: 0; left: 8px; width: 96px; height: 96px; animation: hop 1100ms ease-in-out infinite; }
  .chip-icon { display: block; color: #8ddff0; font-size: 96px; line-height: 1; filter: drop-shadow(0 5px 0 #65b9cb); }
  .chip-icon :global(svg) { width: 96px; height: 96px; }
  .face { position: absolute; top: 34px; left: 29px; display: flex; gap: 7px; }
  .eye { position: relative; width: 15px; height: 19px; border-radius: 50%; background: #fff; }
  .eye::after { content: ""; position: absolute; top: 5px; right: 2px; width: 7px; height: 9px; border-radius: 50%; background: #30434a; }
  .smile { position: absolute; top: 22px; left: 14px; width: 9px; height: 5px; border: 2px solid #397582; border-top: 0; border-radius: 0 0 8px 8px; }
  .neurons { position: absolute; top: 19px; left: 32px; display: flex; gap: 7px; }
  .neurons::before { content: ""; position: absolute; top: 3px; left: 3px; right: 3px; height: 2px; background: #52aec4; }
  .neurons span { width: 6px; height: 6px; border-radius: 50%; background: #229ac4; animation: signal 1100ms ease-in-out infinite; }
  .neurons span:nth-child(2) { animation-delay: 120ms; }
  .neurons span:nth-child(3) { animation-delay: 240ms; }
  .shadow { position: absolute; bottom: 0; left: 37px; width: 38px; height: 6px; border-radius: 50%; background: #d7d5c9; animation: shadow 1100ms ease-in-out infinite; }
  p { margin: 0; color: #71675f; font: inherit; font-size: 16px; }
  @keyframes hop { 0%, 100% { transform: translateY(0) rotate(-4deg); } 50% { transform: translateY(-12px) rotate(4deg); } }
  @keyframes shadow { 0%, 100% { transform: scaleX(1); opacity: .7; } 50% { transform: scaleX(.65); opacity: .35; } }
  @keyframes signal { 0%, 70%, 100% { opacity: .4; } 35% { opacity: 1; } }
  @media (prefers-reduced-motion: reduce) { .chip, .shadow, .neurons span { animation: none; } }
</style>
