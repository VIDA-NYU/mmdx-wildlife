<script lang="ts">
  import { onMount } from "svelte";
  import type { Hits } from "./Api";
  import { random } from "./Api";
  import ImageCard from "./ImageCard.svelte";
  import LabelAll from "./LabelAll.svelte";
  import { selectedDataStore } from "./stores";

  let result: Promise<Hits> | null = null;
  let limit: string = "16";
  let allSelectedData: {[key: string]: boolean; };

  selectedDataStore.subscribe((storeSelectedData) => {
    allSelectedData = storeSelectedData;
  });

  function onQuerySubmit() {
    result = random(+limit);
    result.then( (hits: Hits) => {
      if (result) {
        const imagePaths = hits.hits.map((item) => ({[item.image_path]: true}));
        selectedDataStore.update((storeSelectedData) => {
          return { ...Object.assign({}, ...imagePaths) };
        });
      }
    });
  }

  function onLabelAllEvent() {
    console.log("onChangeLabels");
    result = result;
  }
</script>

<div class="container">
  <div class="py-4">
    <div class="row gy-2 gx-2 align-items-center">
      <div class="col-auto me-2">
        <button class="btn btn-primary" on:click={onQuerySubmit}>
          <i class="fa fa-random" aria-hidden="true" />
          Load new samples
        </button>
      </div>
      <div class="col-auto">
        <label class="col-form-label" for="limitSelect">
          Number of results:
        </label>
      </div>
      <div class="col-auto">
        <select
          class="form-select form-select-sm"
          id="limitSelect"
          bind:value={limit}
          on:change={onQuerySubmit}
        >
          <option value="4">4</option>
          <option value="8">8</option>
          <option value="16">16</option>
          <option value="32">32</option>
          <option value="64">64</option>
        </select>
      </div>
      {#await result}
        <div class="col-auto">
          <span>
            <i class="fa fa-spinner fa-spin" aria-hidden="true" />
            Loading...
          </span>
        </div>
      {:then result}
        {#if result}
          <div class="col-auto">
            <span class="form-text">
              Showing {result.hits.length} random samples.
            </span>
          </div>
        {/if}
      {/await}
    </div>

    {#await result}
      <!-- Awaiting state is already handled above -->
    {:then result}
      {#if result}
        <div class="mt-3 d-flex flex-wrap">
          {#each result.hits as hit, idx}
            <div class="w-25">
              <ImageCard {hit} />
            </div>
          {/each}
          <div class="w-25">
            <LabelAll allHits={result.hits} on:changeLabels={onLabelAllEvent} />
          </div>
        </div>
      {/if}
    {:catch error}
      <div role="alert" class="mt-2 mb-2 alert alert-danger s-dffc7DbZVYr0">
        <strong class="s-dffc7DbZVYr0">Error:</strong>
        {error.message}
      </div>
    {/await}
  </div>
</div>
