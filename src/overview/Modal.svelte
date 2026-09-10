<script>
  import { createEventDispatcher } from "svelte";
  import { modalStore } from "../stores.js";
  import {
    getBackendImageUrl,
    prepareCustomImageWithBackend,
  } from "../utils/cnn-backend.js";

  let inputValue = "";
  let showLoading = false;
  let files;
  let errorMessage = "";
  const dispatch = createEventDispatcher();

  let modalInfo = { show: false };
  modalStore.set(modalInfo);
  modalStore.subscribe((value) => {
    modalInfo = value;
  });

  const closeModal = () => {
    modalInfo.show = false;
    modalStore.set(modalInfo);
    errorMessage = "";
    dispatch("xClicked", { preImage: modalInfo.preImage });
  };

  const useCustomImage = async (source) => {
    if (!source || showLoading) return;

    showLoading = true;
    errorMessage = "";

    try {
      let result = await prepareCustomImageWithBackend(source);
      modalInfo.show = false;
      modalStore.set(modalInfo);
      dispatch("urlTyped", {
        url: getBackendImageUrl(result.imagePath),
        imagePath: result.imagePath,
      });
      inputValue = "";
      files = undefined;
    } catch (error) {
      errorMessage = error?.message || "The image could not be added.";
    } finally {
      showLoading = false;
    }
  };

  const addLink = () => useCustomImage(inputValue.trim());

  const imageUpload = () => {
    let file = files?.[0];
    if (!file) return;

    let reader = new FileReader();
    reader.onload = (event) => useCustomImage(event.target.result);
    reader.onerror = () => {
      errorMessage = "The uploaded image could not be read.";
    };
    reader.readAsDataURL(file);
  };

  const handleKeydown = (event) => {
    if (event.key === "Escape" && modalInfo.show && !showLoading) closeModal();
  };
</script>

<svelte:window on:keydown={handleKeydown} />

{#if modalInfo.show}
  <div class="image-modal" role="presentation">
    <button class="image-modal-backdrop" aria-label="Close image dialog" on:click={closeModal}></button>

    <section class="image-modal-card" role="dialog" aria-modal="true" aria-labelledby="image-modal-title">
      <header class="image-modal-head">
        <div>
          <h2 id="image-modal-title">Add your own image</h2>
          <p>Choose a local image or paste a direct image link.</p>
        </div>
        <button class="close-button" aria-label="Close" disabled={showLoading} on:click={closeModal}>
          <i class="fas fa-times" aria-hidden="true"></i>
        </button>
      </header>

      <div class="image-modal-body">
        <label class="field-label" for="custom-image-url">Image link</label>
        <div class="link-row">
          <div class="url-field">
            <i class="fas fa-link" aria-hidden="true"></i>
            <input
              id="custom-image-url"
              type="url"
              bind:value={inputValue}
              placeholder="https://example.com/image.jpg"
              disabled={showLoading}
              on:keydown={(event) => event.key === "Enter" && addLink()}
            />
          </div>
          <button class="wonder-button primary" disabled={!inputValue.trim() || showLoading} on:click={addLink}>
            {showLoading ? "Adding..." : "Add image"}
          </button>
        </div>

        <div class="divider"><span>OR</span></div>

        <label class="wonder-button upload-button">
          <input
            type="file"
            accept="image/png,image/jpeg"
            bind:files
            disabled={showLoading}
            on:change={imageUpload}
          />
          <i class="fas fa-upload" aria-hidden="true"></i>
          <span>{showLoading ? "Preparing image..." : "Upload from device"}</span>
        </label>

        {#if errorMessage}
          <p class="error-message" role="alert">{errorMessage}</p>
        {/if}
      </div>

      <footer class="image-modal-foot">
        <button class="wonder-button cancel-button" disabled={showLoading} on:click={closeModal}>Cancel</button>
      </footer>
    </section>
  </div>
{/if}

<style>
  .image-modal {
    position: fixed;
    inset: 0;
    z-index: 1000;
    display: grid;
    place-items: center;
    padding: 20px;
    font-family: "Public Sans", sans-serif;
  }

  .image-modal-backdrop {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    padding: 0;
    border: 0;
    background: rgba(31, 29, 27, 0.56);
    cursor: default;
  }

  .image-modal-card {
    position: relative;
    width: min(520px, calc(100vw - 40px));
    overflow: hidden;
    border: 1px solid var(--control-border);
    border-radius: 24px;
    background: var(--page-canvas);
    color: var(--control-text);
    box-shadow: 0 12px 0 rgba(71, 65, 60, 0.28);
  }

  .image-modal-head {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 20px;
    padding: 24px 24px 18px;
  }

  .image-modal-head h2 {
    margin: 0 0 6px;
    font-size: 20px;
    font-weight: 700;
    line-height: 1.25;
  }

  .image-modal-head p {
    margin: 0;
    color: #625b55;
    font-size: 14px;
    line-height: 1.45;
  }

  .close-button {
    display: grid;
    flex: 0 0 36px;
    width: 36px;
    height: 36px;
    place-items: center;
    padding: 0;
    border: 0;
    background: transparent;
    color: var(--control-text);
    font-size: 16px;
    cursor: pointer;
    transition-property: color, opacity, transform;
    transition-duration: 150ms;
    transition-timing-function: cubic-bezier(0.2, 0, 0, 1);
  }

  .close-button:hover:not(:disabled) {
    color: var(--primary-border);
  }

  .close-button:active:not(:disabled) {
    transform: scale(0.96);
  }

  .close-button:focus-visible {
    outline: 2px solid var(--primary-border);
    outline-offset: 2px;
  }

  .image-modal-body { padding: 0 24px 20px; }

  .field-label {
    display: block;
    margin-bottom: 8px;
    text-align: left;
    font-size: 13px;
    font-weight: 600;
  }

  .link-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 10px;
  }

  .url-field {
    display: flex;
    align-items: center;
    gap: 10px;
    min-width: 0;
    height: 48px;
    padding: 0 14px;
    border: 1px solid var(--control-border);
    border-radius: 10px;
    background: #ffffff;
    color: #77706a;
  }

  .url-field:focus-within {
    border-color: var(--primary-border);
    box-shadow: 0 0 0 3px rgba(62, 126, 159, 0.18);
  }

  .url-field input {
    width: 100%;
    min-width: 0;
    border: 0;
    outline: 0;
    background: transparent;
    color: var(--control-text);
    font: inherit;
    font-size: 14px;
  }

  .wonder-button {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 9px;
    min-height: 48px;
    padding: 0 18px;
    border: 1px solid var(--control-border);
    border-radius: 10px;
    background: var(--control-surface);
    color: var(--control-text);
    font: inherit;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    box-shadow: 0 4px 0 var(--control-shadow);
    transition-property: transform, background-color, box-shadow, opacity;
    transition-duration: 150ms;
    transition-timing-function: cubic-bezier(0.2, 0, 0, 1);
  }

  .wonder-button:hover:not(:disabled) {
    background: var(--control-surface-hover);
    transform: translateY(1px);
    box-shadow: 0 3px 0 var(--control-shadow);
  }

  .wonder-button:active:not(:disabled) { transform: scale(0.96); }

  .wonder-button:disabled,
  .close-button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .wonder-button.primary {
    border-color: var(--primary-border);
    background: var(--primary-surface);
    color: var(--primary-text);
    box-shadow: 0 4px 0 var(--primary-shadow);
  }

  .divider {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 22px 0;
    color: #77706a;
    font-size: 11px;
    font-weight: 700;
  }

  .divider::before,
  .divider::after {
    content: "";
    height: 1px;
    flex: 1;
    background: #d8d2ca;
  }

  .upload-button {
    position: relative;
    width: 100%;
    box-sizing: border-box;
  }

  .upload-button input {
    position: absolute;
    inset: 0;
    width: 100%;
    opacity: 0;
    cursor: pointer;
  }

  .error-message {
    margin: 16px 0 0;
    color: #b42318;
    font-size: 13px;
  }

  .image-modal-foot {
    display: flex;
    justify-content: flex-end;
    padding: 16px 24px 22px;
    border-top: 1px solid #d8d2ca;
  }

  .cancel-button { min-height: 42px; }

  @media (max-width: 560px) {
    .link-row { grid-template-columns: 1fr; }
    .wonder-button.primary { width: 100%; }
  }
</style>
