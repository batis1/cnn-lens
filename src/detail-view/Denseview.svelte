<script>
  import { createEventDispatcher, onDestroy, tick } from 'svelte';

  export let layerName;
  export let nodeIndex;
  export let terms = [];
  export let restTerm = null;
  export let bias = 0;
  export let preActivation = 0;
  export let output = 0;
  export let isActive = false;

  const dispatch = createEventDispatcher();
  const format = d3.format('.4f');
  const termStepMs = 360;
  const sumMoveMs = 780;
  const pairColors = [
    '#1f9bcf',
    '#df6f38',
    '#2f80ed',
    '#16a085',
    '#9b59b6',
    '#c27c10',
    '#d14b8f',
    '#5867c8',
  ];
  const aiPairColors = [
    '#0efcff',
    '#ffd166',
    '#73d2ff',
    '#69f0ae',
    '#f9a8ff',
    '#ffb86b',
    '#ff7ab6',
    '#a5b4fc',
  ];

  $: displayTerms = restTerm ? [...terms, restTerm] : terms;

  let revealedTermCount = 0;
  let activeTermIndex = -1;
  let showBias = false;
  let showSum = false;
  let showMovingSum = false;
  let showReluInput = false;
  let showReluOutput = false;
  let movingSumStyle = '';
  let animationTimers = [];
  let animationSignature = '';
  let formulaElement;
  let resultValueElement;
  let reluInputElement;

  const clearAnimationTimers = () => {
    animationTimers.forEach(clearTimeout);
    animationTimers = [];
  };

  const queueAnimationStep = (delay, callback) => {
    animationTimers.push(setTimeout(callback, delay));
  };

  const launchSumMove = async () => {
    showMovingSum = false;
    await tick();

    if (!formulaElement || !resultValueElement || !reluInputElement) {
      showReluInput = true;
      return;
    }

    let formulaRect = formulaElement.getBoundingClientRect();
    let startRect = resultValueElement.getBoundingClientRect();
    let endRect = reluInputElement.getBoundingClientRect();
    movingSumStyle = [
      `left: ${startRect.left - formulaRect.left}px`,
      `top: ${startRect.top - formulaRect.top}px`,
      `width: ${startRect.width}px`,
      `--sum-dx: ${endRect.left - startRect.left}px`,
      `--sum-dy: ${endRect.top - startRect.top}px`,
      `--sum-move-ms: ${sumMoveMs}ms`
    ].join('; ');
    showMovingSum = true;
  };

  const startDenseAnimation = (signature) => {
    animationSignature = signature;
    clearAnimationTimers();
    revealedTermCount = 0;
    activeTermIndex = -1;
    showBias = false;
    showSum = false;
    showMovingSum = false;
    showReluInput = false;
    showReluOutput = false;
    movingSumStyle = '';

    displayTerms.forEach((term, i) => {
      queueAnimationStep(180 + i * termStepMs, () => {
        revealedTermCount = i + 1;
        activeTermIndex = i;
      });
    });

    let delay = 180 + displayTerms.length * termStepMs + 160;
    queueAnimationStep(delay, () => {
      activeTermIndex = -1;
      showBias = true;
    });

    delay += 360;
    queueAnimationStep(delay, () => {
      showSum = true;
    });

    delay += 420;
    queueAnimationStep(delay, () => {
      launchSumMove();
    });

    delay += sumMoveMs + 80;
    queueAnimationStep(delay, () => {
      showReluInput = true;
      showMovingSum = false;
    });

    delay += 320;
    queueAnimationStep(delay, () => {
      showReluOutput = true;
    });
  };

  $: {
    let signature = JSON.stringify({
      layerName,
      nodeIndex,
      terms: displayTerms.map((term) => [
        term.sourceIndex,
        term.weight,
        term.input,
        term.contribution,
        term.count
      ]),
      bias,
      preActivation,
      output
    });

    if (signature !== animationSignature) {
      startDenseAnimation(signature);
    }
  }

  onDestroy(clearAnimationTimers);

  const handleClickX = () => {
    dispatch('xClicked', {});
  };

  const restLabel = (term) => {
    if (term.isRest) {
      return `${term.count} more`;
    }

    return '';
  };

  const termColor = (index, term) => {
    if (term.isRest) {
      return isAiTestUiMode() ? '#8edce7' : '#8f9aa6';
    }

    let colors = isAiTestUiMode() ? aiPairColors : pairColors;
    return colors[index % colors.length];
  };

  const isAiTestUiMode = () =>
    typeof document !== 'undefined' &&
    document.body.classList.contains('ai-test-ui-mode');

  const termTitle = (term) => {
    if (term.isRest) {
      return `${term.count} smaller contributions: ${format(term.contribution)}`;
    }

    return `input ${format(term.input)} × weight ${format(term.weight)} = ${format(term.contribution)}`;
  };

  const weightLabel = (term) => format(term.weight);
  const inputLabel = (term) => format(term.input);
</script>

<style>
  .buttons {
    cursor: pointer;
    position: absolute;
    top: 0px;
    right: 0px;
  }

  .control-button {
    color: gray;
    font-size: 15px;
    opacity: 0.4;
    cursor: pointer;
  }

  .control-button:hover {
    opacity: 0.8;
  }

  .title-text {
    color: #4a4a4a;
    font-size: 1.2em;
    font-weight: 500;
  }

  .box {
    align-items: center;
    display: flex;
    flex-direction: column;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
      Oxygen-Sans, Ubuntu, Cantarell, "Helvetica Neue", sans-serif;
    padding: 10px 14px 16px 14px;
    position: relative;
  }

  .formula {
    align-items: center;
    display: grid;
    grid-template-columns: auto;
    grid-template-rows: auto auto;
    margin: 14px 0 12px 0;
    max-width: calc(100vw - 40px);
    position: relative;
    row-gap: 0;
    width: fit-content;
  }

  .left {
    display: contents;
  }

  .terms {
    border-bottom: 1.2px solid #8d8d8d;
    color: #202020;
    font-size: 10.8px;
    grid-column: 1;
    grid-row: 1;
    justify-self: center;
    line-height: 1.35;
    max-width: 100%;
    padding: 0 8px 9px 8px;
    width: fit-content;
  }

  .relu {
    color: #505050;
    font-size: 12px;
    grid-column: 1;
    grid-row: 2;
    justify-self: center;
    padding-top: 7px;
    text-align: center;
    width: 100%;
  }

  .term {
    cursor: help;
    color: var(--pair-color);
    display: inline-block;
    font-weight: 600;
    opacity: 1;
    transition: opacity 220ms ease, transform 220ms ease, text-shadow 220ms ease;
    white-space: nowrap;
  }

  .term sub {
    font-size: 0.68em;
    line-height: 0;
    margin-left: 1px;
  }

  .term-rest {
    align-items: center;
    color: #8f9aa6;
    display: inline-flex;
    flex-direction: column;
    font-style: italic;
    font-weight: 500;
    line-height: 1.05;
    vertical-align: middle;
  }

  .rest-contribution {
    display: block;
    font-size: 9px;
    font-style: normal;
    font-weight: 500;
    line-height: 1.1;
  }

  .term-entry {
    opacity: 1;
    transform: translateY(0);
    transition: opacity 220ms ease, transform 220ms ease;
  }

  .term-hidden {
    opacity: 0;
    pointer-events: none;
    transform: translateY(3px);
  }

  .term.term-active,
  .term-active .term,
  .bias-active {
    text-shadow: 0 0 6px color-mix(in srgb, var(--pair-color, #67bce7), transparent 55%);
    transform: translateY(-1px);
  }

  .matrix-product {
    align-items: center;
    display: flex;
    gap: 6px;
    justify-content: center;
    min-height: 126px;
  }

  .weight-operand {
    align-items: center;
    display: flex;
    flex-direction: column;
    gap: 3px;
  }

  .input-operand {
    align-items: center;
    display: flex;
    gap: 4px;
  }

  .operand-label {
    color: #6a737d;
    font-size: 11px;
    font-weight: 700;
    line-height: 1;
  }

  .weight-label {
    padding-left: 1px;
  }

  .input-label {
    min-width: 8px;
    text-align: right;
  }

  .vector {
    color: #202020;
    position: relative;
  }

  .vector::before,
  .vector::after {
    border-bottom: 1px solid #777;
    border-top: 1px solid #777;
    bottom: 0;
    content: "";
    position: absolute;
    top: 0;
    width: 6px;
  }

  .vector::before {
    border-left: 1px solid #777;
    left: 0;
  }

  .vector::after {
    border-right: 1px solid #777;
    right: 0;
  }

  .weight-vector {
    align-items: center;
    display: flex;
    gap: 2px;
    max-width: 595px;
    padding: 7px 10px;
  }

  .weight-vector > span {
    align-items: center;
    display: inline-flex;
  }

  .input-vector {
    align-items: center;
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: 58px;
    padding: 7px 8px;
  }

  .operator {
    color: #666;
    font-size: 14px;
    font-weight: 600;
  }

  .comma {
    color: #777;
    margin-left: -2px;
  }

  .bias {
    cursor: help;
    font-weight: 600;
    opacity: 1;
    transition: opacity 220ms ease, transform 220ms ease, text-shadow 220ms ease;
    white-space: nowrap;
  }

  .bias-positive {
    color: #2b93bd;
  }

  .bias-negative {
    color: #df6f38;
  }

  .equals {
    color: gray;
    font-size: 15px;
    font-weight: 700;
  }

  .result {
    color: #202020;
    font-size: 14px;
    min-width: 64px;
    position: relative;
    text-align: right;
  }

  .result-group {
    align-items: center;
    display: flex;
    gap: 10px;
    margin-left: 0;
    opacity: 1;
    padding-left: 4px;
    transition: opacity 240ms ease, transform 240ms ease;
    white-space: nowrap;
  }

  .sum-runner {
    animation: dense-sum-move var(--sum-move-ms) ease-in-out forwards;
    color: #202020;
    font-size: 14px;
    pointer-events: none;
    position: absolute;
    text-align: right;
    z-index: 2;
  }

  .relu-input,
  .relu-output,
  .relu-pass-status {
    display: inline-block;
    opacity: 1;
    transition: opacity 240ms ease, transform 240ms ease;
  }

  .relu-pass-status {
    font-size: 12px;
    font-weight: 600;
    margin-left: 8px;
  }

  .relu-pass-active {
    color: #2b93bd;
  }

  .relu-pass-blocked {
    color: #505050;
  }

  :global(body.ai-test-ui-mode) .title-text {
    color: #0efcff;
    text-shadow: 0 0 8px rgba(14, 252, 255, 0.5);
  }

  :global(body.ai-test-ui-mode) .terms {
    border-bottom-color: rgba(232, 253, 255, 0.72);
    color: #e8fdff;
  }

  :global(body.ai-test-ui-mode) .relu,
  :global(body.ai-test-ui-mode) .operator,
  :global(body.ai-test-ui-mode) .equals,
  :global(body.ai-test-ui-mode) .result,
  :global(body.ai-test-ui-mode) .sum-runner,
  :global(body.ai-test-ui-mode) .operand-label,
  :global(body.ai-test-ui-mode) .vector,
  :global(body.ai-test-ui-mode) .comma {
    color: #e8fdff;
    text-shadow: 0 0 7px rgba(14, 252, 255, 0.28);
  }

  :global(body.ai-test-ui-mode) .vector::before,
  :global(body.ai-test-ui-mode) .vector::after {
    border-color: rgba(232, 253, 255, 0.76);
  }

  :global(body.ai-test-ui-mode) .term {
    text-shadow: 0 0 8px color-mix(in srgb, var(--pair-color), transparent 45%);
  }

  :global(body.ai-test-ui-mode) .term-rest,
  :global(body.ai-test-ui-mode) .rest-contribution {
    color: #8edce7;
  }

  :global(body.ai-test-ui-mode) .bias-positive {
    color: #0efcff;
  }

  :global(body.ai-test-ui-mode) .bias-negative {
    color: #ffb86b;
  }

  :global(body.ai-test-ui-mode) .relu-pass-active {
    color: #69f0ae;
  }

  :global(body.ai-test-ui-mode) .relu-pass-blocked {
    color: #ffb86b;
  }

  :global(body.ai-test-ui-mode) .control-button {
    color: #ffffff;
    opacity: 0.86;
  }

  .relu-hidden {
    opacity: 0;
    transform: translateY(-2px);
  }

  @keyframes dense-sum-move {
    0% {
      opacity: 1;
      transform: translate(0, 0) scale(1);
    }
    70% {
      opacity: 1;
    }
    100% {
      opacity: 0.2;
      transform: translate(var(--sum-dx), var(--sum-dy)) scale(0.96);
    }
  }

</style>

<div class="container">
  <div class="box">
    <div class="buttons">
      <div class="delete-button control-button" on:click={handleClickX} title="Close">
        <i class="fas control-icon fa-times-circle"></i>
      </div>
    </div>

    <div class="title-text">
      Dense Unit <i>{layerName}[{nodeIndex}]</i>
    </div>

    <div class="formula" bind:this={formulaElement}>
      <div class="left">
        <div class="terms">
          <div class="matrix-product">
            <div class="weight-operand">
              <div class="operand-label weight-label">W</div>
              <div class="vector weight-vector" aria-label="weight vector">
                {#each displayTerms as term, i}
                  <span
                    class="term-entry"
                    class:term-hidden={i >= revealedTermCount}
                    class:term-active={i === activeTermIndex}
                  >
                    <span
                      class:term-rest={term.isRest}
                      class="term"
                      style="--pair-color: {termColor(i, term)}"
                      title={termTitle(term)}
                    >
                      {#if term.isRest}
                        {restLabel(term)}
                        <span class="rest-contribution">({format(term.contribution)})</span>
                      {:else}
                        {weightLabel(term)}<sub>{term.sourceIndex}</sub>
                      {/if}
                    </span>
                    {#if i < displayTerms.length - 1}
                      <span class="comma">,</span>
                    {/if}
                  </span>
                {/each}
              </div>
            </div>

            <span class="operator">&times;</span>

            <div class="input-operand">
              <div class="operand-label input-label">X</div>
              <div class="vector input-vector" aria-label="input vector">
                {#each displayTerms as term, i}
                  <span
                    class:term-rest={term.isRest}
                    class:term-hidden={i >= revealedTermCount}
                    class:term-active={i === activeTermIndex}
                    class="term"
                    style="--pair-color: {termColor(i, term)}"
                    title={termTitle(term)}
                  >
                    {#if term.isRest}
                      {restLabel(term)}
                    {:else}
                      {inputLabel(term)}<sub>{term.sourceIndex}</sub>
                    {/if}
                  </span>
                {/each}
              </div>
            </div>

            <span class="operator">+</span>
            <span
              class:term-hidden={!showBias}
              class:bias-active={showBias && !showSum}
              class:bias-positive={bias >= 0}
              class:bias-negative={bias < 0}
              class="bias"
              title="bias: {format(bias)}"
            >
              b({format(bias)})
            </span>

            <span class="result-group" class:term-hidden={!showSum}>
              <span class="equals">=</span>
              <span class="result" bind:this={resultValueElement}>{format(preActivation)}</span>
            </span>
          </div>
        </div>
        <div class="relu">
          ReLU(<span
            class="relu-input"
            class:relu-hidden={!showReluInput}
            bind:this={reluInputElement}
          >{format(preActivation)}</span>) =
          <span
            class="relu-output"
            class:relu-hidden={!showReluOutput}
          >{format(output)}</span>
          <span
            class="relu-pass-status"
            class:relu-hidden={!showReluOutput}
            class:relu-pass-active={isActive}
            class:relu-pass-blocked={!isActive}
          >
            {isActive ? 'Passes' : 'Not passes'}
          </span>
        </div>
      </div>

      {#if showMovingSum}
        <span class="sum-runner" style={movingSumStyle}>{format(preActivation)}</span>
      {/if}
    </div>
  </div>
</div>
