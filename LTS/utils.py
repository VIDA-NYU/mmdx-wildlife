import pandas as pd
import numpy as np
import os
import json
from random_sampling import RandomSampler
from fine_tune import BertFineTuner
from thompson_sampling import ThompsonSampler

def load_data_with_lda(filename, preprocessor, cluster_size, id):
    """
    Load data from a CSV file and apply LDA if necessary.
    Returns the processed DataFrame.
    """
    try:
        data = pd.read_csv(id + "/" + filename + "_lda.csv")
        print("Using data saved on disk")
    except FileNotFoundError:
        print("Creating LDA")
        data = pd.read_csv(id + "/" + filename + ".csv")
        data = preprocessor.preprocess_df(data)
        from lda import LDATopicModel  # Import here to avoid circular imports
        lda_topic_model = LDATopicModel(num_topics=int(cluster_size))
        topics = lda_topic_model.fit_transform(data['clean_title'].to_list())
        data["label_cluster"] = topics
        data.to_csv(id + "/" + filename + "_lda.csv", index=False)
        print("LDA created")
    return data

def save_validation_data(validation, filename):
    """
    Save validation data to a CSV file.
    """
    validation.to_csv(filename, index=False)

def load_json(filename):
    """
    Load data from a JSON file.
    Returns the loaded data.
    """
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            return json.load(file)
    return {}

def save_json(data, filename):
    """
    Save data to a JSON file.
    """
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

def balance_data(df):
    """
    Balance the DataFrame based on the label column.
    Returns the balanced DataFrame.
    """
    label_counts = df["label"].value_counts()
    if len(label_counts) == 2:
        min_count = min(label_counts)
        balanced_df = pd.concat([
            df[df["label"] == 0].sample(min_count * 2),
            df[df["label"] == 1].sample(min_count)
        ])
        return balanced_df.sample(frac=1).reset_index(drop=True)
    return df

def print_summary(sampler):
    """
    Print a summary of the sampler's performance.
    """
    print("Bandit with highest expected improvement:", np.argmax(sampler.wins / (sampler.wins + sampler.losses)))
    print(sampler.wins)
    print(sampler.losses)

def prepare_validation(validation_path, validation_size, data, labeler, preprocessor, id):
    if os.path.exists(f'{id}/validation.csv'):
        validation = pd.read_csv(f'{id}/validation.csv')
    elif validation_path:
        validation = pd.read_csv(validation_path)
        validation = preprocessor.preprocess_df(validation)
        save_validation_data(validation, f"{id}/validation.csv")
    else:
        sampler = RandomSampler(data['label_cluster'].nunique(), id)
        sample_validation, _ = sampler.create_validation_data(data, validation_size)
        validation = labeler.generate_inference_data(sample_validation, 'clean_title')
        if "label" not in validation.columns:
            validation["answer"] = validation.apply(lambda x: labeler.predict_animal_product(x), axis=1)
            validation["answer"] = validation["answer"].str.strip()
            validation["label"] = np.where(validation["answer"] == 'relevant animal', 1, 0)
        save_validation_data(validation, f"{id}/validation.csv")
    return validation

def initialize_trainer(model, model_finetune, validation, id):
    if model == "text":
        return BertFineTuner(model_finetune, None, validation, id=id)
    else:
        raise NotImplementedError

def initialize_sampler(sampling, cluster_size, id):
    if sampling == "thompson":
        return ThompsonSampler(cluster_size, id)
    elif sampling == "random":
        return RandomSampler(cluster_size, id)
    else:
        raise ValueError("Choose one of thompson or random")
