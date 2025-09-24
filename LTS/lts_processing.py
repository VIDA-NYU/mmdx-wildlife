import pandas as pd
import numpy as np
import os
import json
from .utils import load_and_save_csv
from .config import update_config
from .state_manager import write_state

def LTS(sampler, data, sample_size, filter_label, trainer, labeler, filename, balance, metric, baseline, labeling, loop, project_path, process_path, samplingVersion):
    training_data, chosen_bandit = sampler.get_sample_data(data, sample_size, filter_label, trainer, labeling, filename, project_path)
    if training_data.empty:
        return None

    ## Generate labels
    if labeling != "file":
        write_state(project_path, "LLM labeling")
        training_data = labeler.generate_inference_data(training_data, 'clean_title')
        answer = labeler.predict_animal_product(training_data)
        training_data["answer"] = answer
        training_data["answer"] = training_data["answer"].str.strip()
        training_data["label"] = np.where(training_data["answer"].str.contains("not a relevant product"), 0, 1)
        training_data["label"] = training_data["label"].astype(int)
        training_data["label_llm"] = training_data["label"].astype(int)
        file_name = f"{project_path}/{filename}_data_labeled"
        # update to the complete labeled file
        load_and_save_csv(file_name, training_data)
        training_data = add_previous_data(training_data, project_path)  # ADD POSITIVE DATA IF AVAILABLE
        # save current sample for possible label update
        if samplingVersion != "AUTO":
            training_data.to_csv(f"{project_path}/current_sample_training.csv", index=False)
            return {}

    # if "ground_truth" in training_data.columns:
    #     training_data["label"] = training_data["ground_truth"].astype(int)
        # save current sample for possible label update
    training_data.to_csv(f"{project_path}/current_sample_training.csv", index=False)
    if sampler.__class__.__name__ == "ThompsonSampler":
        sampler.update(chosen_bandit, training_data)

    # Balance Data (undersampling) # postpone use of undesire labels negative
    training_data, still_unbalenced = maybe_balance_data(balance, training_data)

    ## FINE TUNE MODEL
    # get base model
    write_state(project_path, "Training")

    model_name, baseline = get_model_and_baseline(trainer, metric, baseline)

    results, results_test, _ = trainer.train_data(training_data, still_unbalenced, process_path)

    improved = (results[f"eval_{metric}"] - baseline) > 0

    # not retrain but fine-tune best model
    if improved:
        print(f"Model improved")
        baseline = results[f"eval_{metric}"]
        model_name = f"{project_path}/models/fine_tunned_{loop}_bandit_{chosen_bandit}"
        trainer.update_model(model_name, baseline, save_model=True)
        # update_config(project_path, {"model_finetune": model_name, "bugetValue": loop, "baseline": baseline})
        # save separated training data file
        name = f'{project_path}/{filename}_training_data'
        load_and_save_csv(name, training_data)
        # use model to label next sample
        if filter_label:
            trainer.set_clf(True)
    else:
        #back to initial model
        # trainer.update_model(model_name, baseline, save_model=False)
        # save positive sample
        load_and_save_csv(f"{project_path}/positive_data", training_data[training_data["label"]==1])

    update_config(project_path, {"model_finetune": model_name, "bugetValue": loop, "baseline": baseline})

    save_bendit_results(filename, chosen_bandit, results, project_path)
    return results_test

def add_previous_data(df, project_path):
    if os.path.exists(f'{project_path}/positive_data.csv'):
        pos = pd.read_csv(f'{project_path}/positive_data.csv')
        df = pd.concat([df, pos]).sample(frac=1)
        print(f"adding positive data: {df['label'].value_counts()}")
        # we try this data only one more time if does not improve it's not a good sample
        os.remove(f'{project_path}/positive_data.csv')
    return df

def maybe_balance_data(balance, df):
    if balance:
        df_diff = df[df["label"] != df["label_llm"]]
        if len(df[df["label"]==1]) > 0:
            unbalanced = len(df[df["label"]==0]) / len(df[df["label"]==1]) >= 2
            if unbalanced:
                label_counts = df["label"].value_counts()
                # Determine the number of samples to keep for each label
                min_count = min(label_counts)
                balanced_df = pd.concat([
                    df[df["label"] == 0].sample(min_count*2),
                    df[df["label"] == 1].sample(min_count)
                ])

                balanced_df = pd.concat([balanced_df, df_diff]).drop_duplicates()
                # Shuffle the rows
                df = balanced_df.sample(frac=1).reset_index(drop=True)
                print(f"Balanced data: {df.label.value_counts()}")
        else:
            unbalanced = True
    else:
        unbalanced = len(df[df["label"]==0]) / len(df[df["label"]==1]) >= 2
    return df, unbalanced

def get_model_and_baseline(trainer, metric, baseline):
    # init_model = trainer.get_init_model()
    model_name = trainer.get_base_model()

    model_results = trainer.get_last_model_acc()
    if model_results:
        baseline = model_results[model_name]
        print(f"previous model {metric} metric baseline of: {baseline}")
    else:
        print(f"Starting with metric {metric} baseline {baseline}")
    print(f"Starting training")

    return model_name, baseline

def save_bendit_results(filename, chosen_bandit, results, project_path):
    if os.path.exists(f'{project_path}/{filename}_model_results.json'):
        with open(f'{project_path}/{filename}_model_results.json', 'r') as file:
            existing_results = json.load(file)
    else:
        existing_results = {}

    if existing_results.get(str(chosen_bandit)):
        existing_results[str(chosen_bandit)].append(results)
    else:
        existing_results[str(chosen_bandit)] = [results]
    # Write the updated list to the file
    with open(f'{project_path}/{filename}_model_results.json', 'w') as file:
        json.dump(existing_results, file, indent=4)
