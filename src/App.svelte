<script>
  import Explainer from "./Explainer.svelte";
</script>

<div id="app-page">
  <recommender-overlay
    my-brand="CNN Visualizer"
    brands-to-ignore="CNN 101"
    recommendation-count="10"
    similar-candidate-count="15"
    position-left="30"
    display-delay="30000"
    homepage-url="https://poloclub.github.io/"
  />
  <Explainer />
</div>

<style>
  #app-page {
    position: fixed;
    inset: 0;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    box-sizing: border-box;
    background: var(--page-chrome);
  }
  @media (max-width: 1100px), (max-height: 600px) {
    #app-page { overflow-y: auto; overflow-x: hidden; }
  }
</style>
