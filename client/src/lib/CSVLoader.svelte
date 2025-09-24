<script lang=ts>
  import * as api from "./Api";
  import Modal from "./Modal.svelte";
  import { projectName } from "./stores";
  export let dataToCSV = [];
  import { navigate } from "svelte-routing";


  export let allowedFileExtensions = ["csv"];

  let isRunning = false;
  let response;

  async function getStatus() {
    response = await api.getStatus();
    if (response) {
      isRunning = response.lts_status;
    }
  }

  let projectMessage="";

  let maxFileSize = 31457280;
  let starting = false;

  // Variables for form and responses
  let uploader;
  let testUploader;
  let writingText = ""; // Variable to bind the writing text
  let uploading = false; // Add a variable to track the uploading state
  let uploadingTest = false; // Add a variable to track the test uploading state
  let saving = false;
  let responseMessage = "";
  let responseMessagePrompt = "";
  let responseMessageSave = "";
  let responseMessageTest = "";
  let validationUploader;
  let uploadingValidation = false;
  let responseMessageValidation = "";


  // LTS Generator parameters (added for the modal)
  let task_prompt = `You are labeling tool to create labels for a classification task .I will provide text data from an advertisement of a product.We are interested in any animal intended to be used for their lether/skin. Also any product made out these materials are important.
  The product should be classified in two labels:
  Label 1: relevant product - if the product is a animal or a product made of any animal leather or skin.
  Label 2: not a relevant product - if the product is 100% synthetic with no animal involved (vegan) such as fake lather or fake skin.Return only one of the two labels, no explanation is necessary.
  Examples:
    1. Advertisement: Huge 62\" Inside Spread Alaskan Yukon Bull Moose Shoulder Mount  | eBay
    Label: not a relevant product
    The product in example 1 is a bull mount. The animal entended used is not about the leather or skin.
    2. Advertisement: Python skin For Sale in Launceston, Cornwall .
    Label: relevant product
    The product on example 2 are selling Python skin wich is a product we are intered in.
    3. Advertisement: Gator leather boots, wallets and purse for sale.
    Label: relevant product
    In exemple 3 we have a 3 different products all made of leather of alligator.
    4. Advertisement: Mario Buccellati, a Rare and Exceptional Italian Silver Gator For Sale at 1stDibs
    Label: not a relevant product
    This example 4 is also not an animal product. The gator in the ad is made out of silver
    5. Advertisement: Leather Recycled African Safari Bookmarks | eBay
    Label: not a relevant product"
`
  let sampling = "thompson";
  let sample_size = 200;
  let model_finetune = "bert-base-uncased";
  let model_init = "bert-base-uncased";
  let labeling = "llama";
  let metric = "f1";
  let validation_size = 400;
  let cluster_size = 10;
  let budget = "trainingSize";
  let baseline = 0;
  let bugetValue = 2000;
  let cluster = "lda"
  let projectId = "";
  let humanLabels = 40;
  let model_name = "meta-llama/Llama-3.3-70B-Instruct";
  let samplingVersion = "random"; // Add this line



  projectName.subscribe((name) => {
    projectId = name;
  });



  let showModal = false; // Reactive variable to control modal visibility

  interface ArgsDict {
    [key: string]: any;  // Allow any key with a string type and any value
  }
  let argsDict: ArgsDict = {}; // Initialize as an empty object


  function updateArgs() {
    argsDict = {
      task_prompt,
      sampling,
      sample_size,
      model_finetune,
      model_init,
      labeling,
      metric,
      validation_size,
      cluster_size,
      budget,
      baseline,
      bugetValue,
      cluster,
      humanLabels,
      model_name, // Add model_name here
      samplingVersion, // Ensure samplingVersion is included
    };
    console.log("Updated argsDict:", argsDict); // Debugging to verify updates
  }

  // Handle file upload
  async function uploadFile(event) {
    event.preventDefault();
    const file = uploader.files[0];
    uploading = true; // Set uploading state to true
    await onUpload(file);
    uploading = false; // Reset uploading state
  }

  async function uploadTestFile(event) {
    event.preventDefault();
    const file = testUploader.files[0];
    uploadingTest = true; // Set test uploading state to true
    await onUploadTest(file);
    uploadingTest = false; // Reset test uploading state
  }

  async function uploadValidationFile(event) {
    event.preventDefault();
    const file = validationUploader.files[0];
    uploadingValidation = true;
    await onUploadValidation(file);
    uploadingValidation = false;
  }

  // Handle the file upload process
  async function onUpload(file) {
    try {
      responseMessage = await api.loadCSV(file, projectId, "train");
    } catch (error) {
      responseMessage = `Error loading CSV data: ${error.message}`;
    }
  }

  async function onUploadTest(file) {
    try {
      responseMessageTest = await api.loadCSV(file, projectId, "test");
    } catch (error) {
      responseMessageTest = `Error loading Test CSV data: ${error.message}`;
    }
  }

  async function onUploadValidation(file) {
    try {
      responseMessageValidation = await api.loadCSV(file, projectId, "validation");
    } catch (error) {
      responseMessageValidation = `Error loading Validation CSV data: ${error.message}`;
    }
  }

  // Start the LTS data generation
  async function createLtsConfig() {
    updateArgs();
    saving = true;
    argsDict["model_init"] = argsDict["model_finetune"];
    try {
      const response = await api.createLtsConfig(projectId, argsDict); // Pass the updated argsDict
      responseMessageSave = "Project Saved!";
      console.log(response);
    } catch (error) {
      responseMessageSave = `Error starting LTS data generation: ${error.message}`;
    }
    saving = false;
  }

  // Reset project ID function
  function resetProjectId() {
    projectId = ""; // Clear the projectId variable
    projectName.update(() => {
      return "";
    });
  }

  async function confirmName(new_name) {
    projectName.update(() => {
      return new_name;
    });

    try {
      projectMessage = await api.setLabelsDb(new_name);
      await getStatus(); // Check the status after setting the project name
    } catch (error) {
      projectMessage = `Error setting labels db: ${error.message}`;
    }

    return new_name;
  }


  async function startTraining(){
     try {
      starting = true;
      const trainingResponse = await api.startTraining({ projectId: projectId, labeling: ""});
    } catch (error) {
      const responseMessage = `Error starting model training: ${error.message}`;
    }
    navigate("/result?q=")
  }


</script>
<div class="container-fluid px-5">
  <!-- Create Project ID Section -->
  <div class="py-4">
    {#if  isRunning}
    <div class="alert alert-primary position-absolute bottom-50 end-50" role="alert">
      <h1>
        <span class="fa fa-exclamation-triangle" />
        Project running
      </h1>
    </div>
    {/if}
    <!-- <h1 class=
     "">Enter Project Name</h1> -->
    <div class="form-group">
      <label for="projectIdInput"style="font-size: 1.2rem;"><strong>Project Name</strong></label>
      <input
        id="projectIdInput"
        bind:value={projectId}
        type="text"
        class="form-control"
        style="max-width:400px"
        placeholder="Enter your project name"
      />
    </div>
    <div class="pt-2">
      <button
        class="btn btn-info"
        on:click={() => confirmName(projectId)}
        disabled={isRunning}
      >
        <span class="fa fa-id-card mr-2" />
        Confirm Project Name
      </button>
      <!-- Reset Project ID Button -->
      <button class="btn btn-warning ml-2"
      on:click={resetProjectId}
      disabled={isRunning}
      >
        <span class="fa fa-refresh mr-2" />
        Reset Project
      </button>
    </div>
    <!-- </div> -->
    {#if projectMessage}
      <div>
        <p>{projectMessage}</p>
      </div>
    {/if}

    {#if projectId}
      <div class="mt-2">
        <p><strong>Project Name: {projectId}</strong></p>
      </div>
    {/if}
  </div>

  <!-- Load Data Section -->
  <div class="py-4">
    <!-- <h1>Load Data</h1> -->
    <label for="projectData"style="font-size: 1.2rem;"><strong>Select Dataset</strong></label>
    <input
      id="projectData"
      bind:this={uploader}
      type="file"
      class="form-control"
      style="max-width:400px"
    />
    <div class="pt-2">
      <button class="btn btn-primary"
      on:click={uploadFile}
      disabled={isRunning}
      >
        <span class="fa fa-download mr-2" />
        Load File
      </button>
    </div>

    <div class="mt-2">
      {#if uploading}
        <span>
          <i class="fa fa-spinner fa-spin" aria-hidden="true" />
          Loading...
        </span>
      {:else if responseMessage}
        <div>
          <p>{responseMessage}</p>
        </div>
      {/if}
    </div>
  </div>

  <!-- Test Dataset -->
  <div class="py-4">
    <label for="testData" style="font-size: 1.2rem;"><strong>Select Test Dataset</strong></label>
    <input
      id="testData"
      bind:this={testUploader}
      type="file"
      class="form-control"
      style="max-width:400px"
    />
    <div class="pt-2">
      <button class="btn btn-primary"
        on:click={uploadTestFile}
        disabled={isRunning}
      >
        <span class="fa fa-download mr-2" />
        Load Test File
      </button>
    </div>
  </div>

  <!-- Validation Dataset -->
  <div class="py-4">
    <label for="validationData" style="font-size: 1.2rem;"><strong>Select Validation Dataset</strong></label>
    <input
      id="validationData"
      bind:this={validationUploader}
      type="file"
      class="form-control"
      style="max-width:400px"
    />
    <div class="pt-2">
      <button class="btn btn-primary"
        on:click={uploadValidationFile}
        disabled={isRunning}
      >
        <span class="fa fa-download mr-2" />
        Load Validation File
      </button>
    </div>
    <div class="mt-2">
      {#if uploadingValidation}
        <span>
          <i class="fa fa-spinner fa-spin" aria-hidden="true" />
          Loading Validation Set...
        </span>
      {:else if responseMessageValidation}
        <div>
          <p>{responseMessageValidation}</p>
        </div>
      {/if}
    </div>
  </div>

  <!-- Start LTS Data Generator Button -->
  <div class="py-4">
    <label for="seetings"style="font-size: 1.2rem;"><strong>HILTS Settings</strong></label>
    <div class="pt-2">
      <button class="btn btn-secondary" on:click={() => (showModal = true)}  disabled={isRunning}>
        <!-- <span class="fa fa-cogs mr-2" /> -->
        Update Settings
      </button>
      <div class="pt-2">
        <button class="btn btn-success"
          on:click={startTraining}
          disabled={isRunning}>
          <!-- startLTSGenerator -->
          <span class="fa fa-cogs mr-2" />
          Start Project
        </button>
        {#if starting}
            <span>
              <i class="fa fa-spinner fa-spin" aria-hidden="true"></i>
              Waiting project to start...
            </span>
        {/if}
      </div>

      <div class="mt-2">
        {#if saving}
          <span>
            <i class="fa fa-spinner fa-spin" aria-hidden="true" />
            Starting LTS Data Generator...
          </span>
        {:else if responseMessageSave}
          <div>
            <p>{responseMessageSave}</p>
          </div>
        {/if}
      </div>
    </div>
  </div>
</div>


<!-- Modal for LTS Data Generator -->
<Modal bind:showModal={showModal} closeBtnName="Save" onCloseAction={createLtsConfig}>
  <h2 slot="header">
    <div class="card">
      <div class="card-header bg-primary text-white">HILTS Settings
    </div></h2>
  <div slot="body">
    <!-- Modal Content (LTS Parameters) -->
    <div class="row">
      <div class="col-12">
        <label for="prompt">Describe Task</label>
        <textarea
          bind:value={task_prompt}
          rows="5"
          class="form-control"
          placeholder="Write your prompt here..."
        ></textarea>
      </div>
      <div class="col-6">
        <label for="sampling">Sampling</label>
        <select id="sampling" class="form-control" bind:value={sampling}>
          <option value="random">Random</option>
          <option value="thompson">Thompson</option>
        </select>
      </div>
      <div class="col-6">
        <label for="sample_size">Sample Size</label>
        <input
          id="sample_size"
          type="number"
          class="form-control"
          bind:value={sample_size}
        />
      </div>

      <div class="col-6">
        <label for="model_finetune">Base Model</label>
        <!-- <select
          id="model_finetune"
          class="form-control"
          bind:value={model_finetune}
        > -->
        <textarea
          bind:value={model_finetune}
          rows="1"
          class="form-control"
          placeholder="HuggingFaceModel"
        ></textarea>
          <!-- <option value="bert-base-uncased">BERT</option>
          <option value="google-bert/bert-base-multilingual-cased"
            >multilingual bert</option
          > -->
        <!-- </select> -->
      </div>
      <!-- <div class="col-6">
        <label for="stop">Stop</label>
        <input
          id="stop"
          type="number"
          class="form-control"
          bind:value={stop}
        />
      </div> -->
      <div class="col-6">
        <label for="labeling">LLM Labeling</label>
        <select id="labeling" class="form-control" bind:value={labeling}>
          <option value="gpt">GPT</option>
          <option value="llama">LLAMA</option>
        </select>
      </div>
      <div class="col-6">
        <label for="metric">Evaluation Metric</label>
        <select id="metric" class="form-control" bind:value={metric}>
          <option value="accuracy">Accuracy</option>
          <option value="f1">F1 Score</option>
          <option value="recall">Recall</option>
          <option value="recall">Precision</option>
        </select>
      </div>
      <div class="col-6">
        <label for="baseline">Baseline Metric Value</label>
        <input
          id="baseline"
          type="number"
          class="form-control"
          bind:value={baseline}
        />
      </div>
      <div class="col-6">
        <label for="validation_size">Validation Data Size</label>
        <input
          id="validation_size"
          type="number"
          class="form-control"
          bind:value={validation_size}
        />
      </div>
      <div class="col-6">
        <label for="budget">Budget</label>
        <div class="d-flex">
          <select id="budget" class="form-control" bind:value={budget}>
            <option value="trainingSize">Minimum Training Size</option>
            <option value="metric">Metric</option>
          </select>
            <input
              id="custom-budget"
              type="number"
              class="form-control ml-2"
              bind:value={bugetValue}
              placeholder="Enter custom value"
            />
        </div>
      </div>
      <div class="col-6">
        <label for="cluster">Cluster</label>
        <select id="cluster" class="form-control" bind:value={cluster}>
        <option value="hdbscan">GMM</option>
        <option value="dbscan">AGGLO</option>
        <option value="kmeans">K-means</option>
        <option value="lda">LDA</option>
        </select>
      <div class="col-6">
        <label for="cluster_size">Number of Clusters</label>
        <input
          id="cluster_size"
          type="number"
          class="form-control"
          bind:value={cluster_size}
        />
      </div>
      </div>
      <div class="col-6">
        <label for="sampling_version">Sampling Version</label>
        <select id="sampling_version" class="form-control" bind:value={samplingVersion}>
          <option value="random">random</option>
          <option value="nn-voting">embedding</option>
          <option value="uncertainty">uncertainty</option>
          <option value="AUTO">AUTO</option>
        </select>
      </div>
      <div class="col-6">
        <label for="model_name">Model Name</label>
        <textarea
          id="model_name"
          bind:value={model_name}
          rows="1"
          class="form-control"
          placeholder="Enter model name (e.g., meta-llama/Llama-3.3-70B-Instruct)"
        ></textarea>
      </div>
      <div class="col-6">
        <label for="human_labels">Minimum number of samples to correct</label>
        <input
          id="human_labels"
          type="number"
          class="form-control"
          bind:value={humanLabels}
        />
      </div>

    </div>
  </div>
</Modal>



<style>
  h1{
    color: #636363
  }
  .container {
    padding: 20px;
    max-width: 800px;
    margin: 0 auto;
  }
</style>
