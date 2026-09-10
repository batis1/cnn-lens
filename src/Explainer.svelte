<script>
  import Overview from './overview/Overview.svelte';
  import { cnnStore } from './stores.js';

  // Enum to control the displaying view
  const View = {
    OVERVIEW: 'overview',
    LAYERVIEW: 'layerview',
    DETAILVIEW: 'detailview'
  };

  let mainView = View.OVERVIEW;

  /* Example to read loaded cnn in other components:
  $: if ( $cnnStore.length != 0) {
    console.log($cnnStore);
  }
  */

</script>

<style>
#explainer {
  width: 100%;
  min-height: 0;
  flex: 1 1 auto;
  padding: 0;
  margin: auto;
  overflow: hidden;
  /* outline: 1px solid var(--g-dark-gray); */
}
@media (max-width: 1100px), (max-height: 600px) {
  #explainer { overflow: visible; flex: 1 0 auto; margin: 0; }
}
</style>

<div id='explainer'>
    <Overview />
</div>
