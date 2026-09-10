<script>
  import { createEventDispatcher, onMount, tick } from "svelte";

  export let value = "";
  export let options = [];
  export let disabled = false;
  export let placeholder = "Select";
  export let ariaLabel = "Select an option";
  export let iconClass = "";
  export let id = undefined;
  export let className = "";

  const dispatch = createEventDispatcher();
  let root;
  let trigger;
  let menu;
  let open = false;

  $: selectedOption = options.find((option) => option.value === value);
  $: displayLabel = selectedOption?.label || placeholder;

  const close = (restoreFocus = false) => {
    open = false;
    if (restoreFocus) {
      tick().then(() => trigger?.focus());
    }
  };

  const toggle = () => {
    if (!disabled) {
      open = !open;
    }
  };

  const choose = (option) => {
    if (option.disabled) return;
    value = option.value;
    dispatch("change", value);
    close(true);
  };

  const handleTriggerKeydown = async (event) => {
    if (event.key === "Escape" && open) {
      event.preventDefault();
      close(true);
      return;
    }
    if (!["ArrowDown", "ArrowUp", "Enter", " "].includes(event.key)) return;
    event.preventDefault();
    open = true;
    await tick();
    const activeOption = menu?.querySelector('[aria-selected="true"]');
    const firstOption = menu?.querySelector('button:not(:disabled)');
    (activeOption || firstOption)?.focus();
  };

  const handleMenuKeydown = (event) => {
    const enabledOptions = Array.from(
      menu?.querySelectorAll('button:not(:disabled)') || [],
    );
    const currentIndex = enabledOptions.indexOf(document.activeElement);

    if (event.key === "Escape") {
      event.preventDefault();
      close(true);
    } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      const direction = event.key === "ArrowDown" ? 1 : -1;
      const nextIndex =
        (currentIndex + direction + enabledOptions.length) % enabledOptions.length;
      enabledOptions[nextIndex]?.focus();
    } else if (event.key === "Home" || event.key === "End") {
      event.preventDefault();
      enabledOptions[event.key === "Home" ? 0 : enabledOptions.length - 1]?.focus();
    }
  };

  onMount(() => {
    const handleOutsidePointer = (event) => {
      if (open && !root?.contains(event.target)) close();
    };
    document.addEventListener("pointerdown", handleOutsidePointer);
    return () => document.removeEventListener("pointerdown", handleOutsidePointer);
  });
</script>

<div class={`wondering-select ${className}`} bind:this={root}>
  <button
    bind:this={trigger}
    type="button"
    {id}
    class="wondering-select-trigger"
    class:is-open={open}
    aria-label={`${ariaLabel}: ${displayLabel}`}
    aria-haspopup="listbox"
    aria-expanded={open}
    {disabled}
    on:click={toggle}
    on:keydown={handleTriggerKeydown}
  >
    {#if iconClass}
      <span class="wondering-select-icon" aria-hidden="true">
        <i class={iconClass}></i>
      </span>
    {/if}
    <span class:placeholder={!selectedOption} class="wondering-select-label">
      {displayLabel}
    </span>
    <span class="wondering-select-chevron" aria-hidden="true">
      <i class={open ? "fas fa-chevron-up" : "fas fa-chevron-down"}></i>
    </span>
  </button>

  {#if open}
    <div
      bind:this={menu}
      class="wondering-select-menu"
      role="listbox"
      aria-label={ariaLabel}
      tabindex="-1"
      on:keydown={handleMenuKeydown}
    >
      {#each options as option}
        <button
          type="button"
          role="option"
          aria-selected={option.value === value}
          class:selected={option.value === value}
          disabled={option.disabled}
          on:click={() => choose(option)}
        >
          {option.label}
        </button>
      {/each}
    </div>
  {/if}
</div>

<style>
  .wondering-select {
    position: relative;
    min-width: 0;
    font-family: inherit;
  }

  .wondering-select-trigger {
    width: 100%;
    height: 36px;
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 8px 11px;
    color: var(--control-text);
    background: var(--control-surface);
    border: 1px solid var(--control-border);
    border-radius: 8px;
    box-shadow: 0 3px 0 var(--control-shadow);
    font: inherit;
    font-size: 14px;
    font-weight: 500;
    line-height: 1;
    cursor: pointer;
    transition: background-color 140ms ease, box-shadow 140ms ease,
      transform 140ms ease;
  }

  .wondering-select-trigger:hover:not(:disabled),
  .wondering-select-trigger.is-open {
    background: var(--control-surface-hover);
  }

  .wondering-select-trigger:focus-visible,
  .wondering-select-trigger.is-open {
    outline: 3px solid rgba(62, 126, 159, 0.2);
    outline-offset: 1px;
    border-color: var(--primary-border);
  }

  .wondering-select-trigger:active:not(:disabled) {
    transform: translateY(2px) scale(0.96);
    box-shadow: 0 1px 0 var(--control-shadow);
  }

  .wondering-select-trigger:disabled {
    cursor: not-allowed;
    opacity: 0.5;
  }

  .wondering-select-icon,
  .wondering-select-chevron {
    display: grid;
    place-items: center;
    color: #69615b;
    font-size: 12px;
    flex: 0 0 auto;
  }

  .wondering-select-label {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .wondering-select-label.placeholder {
    color: #6f6861;
  }

  .wondering-select-chevron {
    margin-left: auto;
    font-size: 10px;
  }

  .wondering-select-menu {
    position: absolute;
    z-index: 100;
    top: calc(100% + 7px);
    left: 0;
    width: max-content;
    min-width: 100%;
    max-height: min(420px, 60vh);
    overflow-y: auto;
    padding: 4px;
    background: #fffaf0;
    border: 1px solid #d3ccc1;
    border-radius: 10px;
    box-shadow: 0 10px 28px rgba(48, 41, 37, 0.12),
      0 2px 6px rgba(48, 41, 37, 0.08);
  }

  .wondering-select-menu button {
    width: 100%;
    min-height: 34px;
    display: flex;
    align-items: center;
    padding: 7px 11px;
    color: var(--control-text);
    background: transparent;
    border: 0;
    border-radius: 6px;
    font: inherit;
    font-size: 14px;
    font-weight: 400;
    line-height: 1.3;
    text-align: left;
    white-space: nowrap;
    cursor: pointer;
    transition: background-color 120ms ease;
  }

  .wondering-select-menu button:hover:not(:disabled),
  .wondering-select-menu button:focus-visible {
    background: #f0ece4;
    outline: none;
  }

  .wondering-select-menu button.selected {
    background: #e5e1da;
    font-weight: 500;
  }

  .wondering-select-menu button:disabled {
    color: #989087;
    cursor: default;
  }

  @media (prefers-reduced-motion: reduce) {
    .wondering-select-trigger,
    .wondering-select-menu button {
      transition: none;
    }
  }
</style>
