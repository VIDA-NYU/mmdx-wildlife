<script lang="ts">
  import Modal from "./Modal.svelte";
  import type { Hit } from "./Api";
  import { navigate } from "svelte-routing";
  import {
    descriptionsStore,
    animalStore,
    negativeKeywordStore,
    selectedDataStore,
  } from "./stores";
  import AutoComplete from "simple-svelte-autocomplete";
  import * as api from "./Api";

  export let hit: Hit;
  $: parsedHitMetadata = hit.metadata;
  $: hitLabels = hit.labels_types_dict;

  let showModal = false;

  let allLabels: string[];
  let allAnimals: string[];
  let allNegKeywords: string[];
  let allSelectedData: { [key: string]: boolean };
  let selectedDescription: string;
  let selectedNegKeyword: string;
  let selectedAnimal: string;
  let isSelected: boolean = true;

  // tableau10 colors
  const colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
  ];

  descriptionsStore.subscribe((storeLabels) => {
    allLabels = storeLabels;
  });

  animalStore.subscribe((storeAnimals) => {
    allAnimals = storeAnimals;
  });

  negativeKeywordStore.subscribe((storeNegKeyword) => {
    allNegKeywords = storeNegKeyword;
  });

  selectedDataStore.subscribe((storeSelectedData) => {
    allSelectedData = storeSelectedData;
  });

  function togggleSelected() {
    isSelected = !isSelected;
    selectedDataStore.update((selectedData) => {
      selectedData[hit.image_path] = isSelected;
      return selectedData;
    });
  }

  function addLabelExclusive(newLabel: string, type: api.LabelType) {
    if (!newLabel || newLabel === "") {
      return;
    }
    let hitLabels = hit.labels_types_dict;
    let hitKey = Object.keys(hitLabels).find((key) => hitLabels[key] === type);
    if (typeof hitKey === 'string') {
      if (hitKey === newLabel) {
        return;
      }
      if (
        hitKey !== newLabel
      ){
        api.removeLabel(hit.image_path, hitKey, `${type}`);
        delete hitLabels[hitKey];
      }
    }
    hitLabels[newLabel] = `${type}`;
    hit.labels_types_dict = hitLabels;
    try {
      api.addLabel(hit.image_path, newLabel, `${type}`);
    } catch (e) {
      console.log(e);
    }
  }

  function addLabelInclusive(newLabel: string, type: api.LabelType) {
    if (!newLabel || newLabel === "") {
      return;
    }
    let hitLabels = hit.labels_types_dict;
    let hitKey = Object.keys(hitLabels).find((key) => hitLabels[key] === type);
    console.log(hitKey)
    if ((Array.isArray(hitKey)) && hitKey.length > 0) {
      if (hitKey.includes(newLabel)) {
        return;
      }
    }
    hitLabels[newLabel] = `${type}`;
    hit.labels_types_dict = hitLabels;
    try {
      api.addLabel(hit.image_path, newLabel, `${type}`);
    } catch (e) {
      console.log(e);
    }
  }

  function handleCreateDescription(newLabel: string) {
    descriptionsStore.update((storeLabels) => {
      return [...new Set([...storeLabels, newLabel])];
    });
    return newLabel; // return the new label to the autocomplete
  }

  function handleCreateKeyword(newKeyword: string) {
    negativeKeywordStore.update((storeNegKeyword) => {
      return [...new Set([...storeNegKeyword, newKeyword])];
    });
    return newKeyword; // return the new label to the autocomplete
  }

  function removeLabels(label: string) {
    let hitLabels = hit.labels_types_dict;
    let type =  hitLabels[label] as api.LabelType;
    console.log(hitLabels, type);
    if (hitLabels && hitLabels[label]) {
      api.removeLabel(hit.image_path, label, `${type}`);
      delete hitLabels[label];
      hit.labels_types_dict = hitLabels;
    }
    console.log("end: ", hitLabels);
  }
</script>

<div class="card me-3 mb-3">
  <img
    src={"/images/" + hit.image_path}
    style="max-height: 350px;"
    class="card-img-top"
  />
  <div class="box-container">
    <div
      class="d-flex flex-row-reverse btn-group btn-group-toggle btn-group-sm"
      data-toggle="buttons"
    >
      <label class="btn btn-sm btn-outline-primary" style="font-size: 0.7em">
        <input
          class="form-check-input"
          id="selectedCheck"
          type="checkbox"
          autocomplete="off"
          bind:checked={isSelected}
          on:click={() => togggleSelected()}
        />
        {"Select Ad"}
      </label>
    </div>
    <div class="card-body">
      <p class="card-text mb-2">
        {hit.title ? hit.title : hit.image_path}
      </p>
      <div class="btn-toolbar mt-1">
        <div class="btn-group me-2" role="group" aria-label="">
          <button
            class="btn btn-sm btn-success"
            on:click={() => addLabelExclusive("animal origin", "relevant")}
          >
            <i class="fa fa-thumbs-up" aria-hidden="true" />
          </button>
          <button
            class="btn btn-sm btn-warning"
            on:click={() => addLabelExclusive("not animal origin", "relevant")}
          >
            <i class="fa fa-thumbs-down" aria-hidden="true" />
          </button>
        </div>
        <div class="btn-group me-2 mt-1" role="group" aria-label="">
          <AutoComplete
            debug={false}
            inputClassName="form-control"
            items={allAnimals}
            bind:selectedItem={selectedAnimal}
            create={false}
            onChange={(label) => addLabelExclusive(label, "animal")}
            placeholder="Animal"
          />
        </div>
        <div class="box-container">
          <div class="btn-group me-2" role="group" aria-label="">
            <AutoComplete
              debug={false}
              inputClassName="form-control"
              items={allLabels}
              bind:selectedItem={selectedDescription}
              create={true}
              onCreate={handleCreateDescription}
              onChange={(label) => addLabelInclusive(label, "description")}
              placeholder="Description"
            />
          </div>
        </div>
        <div class="box-container">
          <div class="btn-group me-2" role="group" aria-label="">
            <AutoComplete
              debug={false}
              inputClassName="form-control"
              items={allNegKeywords}
              bind:selectedItem={selectedNegKeyword}
              create={true}
              onChange={(label) => addLabelInclusive(label, "keywords")}
              onCreate={handleCreateKeyword}
              placeholder="Negative Keyword"
            />
          </div>
        </div>
        {#if hitLabels && Object.keys(hitLabels).length > 0}
          <div class="btn-toolbar mb-1">
            {#each Object.entries(hitLabels) as [label, value], idx}
              <span
                class="badge rounded-pill bg-secondary me-1 mt-1 mb-1 position-relative"
              >
                <!-- style="background-color: {colors[idx]} !important;" -->
                {label}
                <span
                  role="button"
                  on:click={() => removeLabels(label)}
                  class="position-absolute top-0 start-100 translate-middle"
                >
                  <i class="fa fa-times-circle" aria-hidden="true" />
                  <span class="visually-hidden">Remove label</span>
                </span>
              </span>
            {/each}
          </div>
        {/if}
        <div class="d-flex justify-content-between">
          <div class="btn-group">
            <button
              class="btn btn-sm btn-info me-1"
              on:click={() => (showModal = true)}
            >
              <i class="fa fa-info-circle me-1" aria-hidden="true" />
              Metadata
            </button>
            <button
              class="btn btn-sm btn-info"
              on:click={() => navigate("/search/image?q=" + hit.image_path)}
            >
              <i class="fa fa-search me-1" aria-hidden="true" />
              Find Similar
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<Modal bind:showModal>
  <h2 slot="header">Metadata</h2>

  <ul slot="body" class="definition-list___">
    <li>
      <strong>title:</strong>
      {hit.title ? hit.title : hit.image_path}
    </li>
    {#each Object.keys(parsedHitMetadata) as key}
      <li>
        {#if key == "url"}
          <strong>{key}:</strong>
          <a
            href={parsedHitMetadata[key]}
            target="_blank"
            referrerpolicy="no-referrer">{parsedHitMetadata[key]}</a
          >
        {:else}
          <strong>{key}:</strong> {parsedHitMetadata[key]}
        {/if}
      </li>
    {/each}
  </ul>
</Modal>

<style>
  :global(.autocomplete-list) {
    --bs-dropdown-zindex: 1000;
    --bs-dropdown-min-width: 10rem;
    --bs-dropdown-padding-x: 0;
    --bs-dropdown-padding-y: 0.1rem;
    --bs-dropdown-spacer: 0.125rem;
    --bs-dropdown-font-size: 1rem;
    --bs-dropdown-color: var(--bs-body-color);
    --bs-dropdown-bg: var(--bs-body-bg);
    --bs-dropdown-border-color: var(--bs-border-color-translucent);
    --bs-dropdown-border-radius: var(--bs-border-radius);
    --bs-dropdown-border-width: var(--bs-border-width);
    --bs-dropdown-inner-border-radius: calc(
      var(--bs-border-radius) - var(--bs-border-width)
    );
    --bs-dropdown-divider-bg: var(--bs-border-color-translucent);
    --bs-dropdown-divider-margin-y: 0.1rem;
    --bs-dropdown-box-shadow: 0 0.5rem 1rem rgba(0, 0, 0, 0.15);
    --bs-dropdown-link-color: var(--bs-body-color);
    --bs-dropdown-link-hover-color: var(--bs-body-color);
    --bs-dropdown-link-hover-bg: var(--bs-tertiary-bg);
    --bs-dropdown-link-active-color: #fff;
    --bs-dropdown-link-active-bg: #7c6bb2;
    --bs-dropdown-link-disabled-color: var(--bs-tertiary-color);
    --bs-dropdown-item-padding-x: 1rem;
    --bs-dropdown-item-padding-y: 0.25rem;
    --bs-dropdown-header-color: #6c757d;
    --bs-dropdown-header-padding-x: 1rem;
    --bs-dropdown-header-padding-y: 0.1rem;

    color: var(--bs-dropdown-color) !important;
    text-align: left !important;
    list-style: none !important;
    background-color: var(--bs-dropdown-bg) !important;
    background-clip: padding-box !important;
    border: var(--bs-dropdown-border-width) solid
      var(--bs-dropdown-border-color) !important;
    border-radius: var(--bs-dropdown-border-radius) !important;
    width: 171px !important;
  }
  :global(.autocomplete-list-item) {
    color: var(--bs-dropdown-color) !important;
  }
  :global(
      .autocomplete-list .selected,
      .dropdown-item:hover,
      .dropdown-item:focus
    ) {
    color: var(--bs-dropdown-link-hover-color) !important;
    background-color: var(--bs-dropdown-link-hover-bg) !important;
  }
  :global(.input-container > .form-control) {
    padding: 0.25rem 0.5rem !important;
    font-size: 0.875rem !important;
    border-radius: var(--bs-border-radius-sm);
  }
  :global(.autocomplete) {
    height: 2em !important;
    min-width: 100px !important;
  }
  :global(.autocomplete-list-item-create) {
    width: 169px !important;
  }
  :global(.input-container) {
    /* width: 207px !important; */
    width: 171px !important;
  }
  h1 {
    font-size: 1rem;
    text-align: center;
  }
</style>
