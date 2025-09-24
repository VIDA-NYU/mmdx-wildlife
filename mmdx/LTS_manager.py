import multiprocessing
import os
import pandas as pd
import numpy as np
from LTS.main import initialize_LTS
from LTS.state_manager import write_state
from LTS.lts_processing import LTS
import json
import shutil
from .search import VectorDB
import time
from LTS.config import update_config
from mmdx.settings import (
    DATA_PATH,
    DB_PATH,
)
from mmdx.model import ClipModel



multiprocessing.set_start_method('spawn', force=True)


class LTSManager:
    def __init__(self, projectID: str, labeldb: VectorDB, demo: bool):
        self.project_id = projectID
        self.status = False
        self.process = None
        # self.labeldb = labeldb
        self.project_path = f"data/{self.project_id}"
        self.metrics_path = f"{self.project_path}/metrics.json"
        self.final_model_res = f"{self.project_path}/final_model_metrics.json"
        self.stop_path = f"{self.project_path}/stop.txt"
        self.demo = demo

    def train_model(self, label_hilts):
        # Reinitialize the VectorDB object in the child process
        model = ClipModel()
        labeldb = VectorDB.from_data_path(
            data_path=DATA_PATH,
            db_path=DB_PATH,
            S3_Client=None,  # Pass S3_Client if needed
            model=model,
            delete_existing=False,
        )
        labeldb.set_label_db(self.project_id)

        # Continue with the rest of your training logic
        process_id = self.process.pid
        self.process_path = f"{self.project_path}/{process_id}"
        print(f"process_path inside train: {self.process_path}")

        print(f"processid: {process_id}")

        args, sampler, data, trainer, labeler = initialize_LTS(self.project_path, self.demo)
        samplingVersion = args.get("samplingVersion")
        print("samplingVersion", samplingVersion)
        budget = args.get("budget")
        budget_value = int(args.get("bugetValue"))
        # stop_on = args.get("stop")
        result_json = []
        if budget == "trainingSize":
            loops = int(budget_value/args.get("sample_size"))
            print("trainingSize loops", loops)
            for idx in range(loops):
                print("Loop idx", idx)
                loop = idx+1
                if os.path.exists(self.metrics_path):
                    result_json = self.get_metrics(self.project_path)
                    loop = max(result_json["step"]) + 1
                with open(self.process_path + "/loop.txt", "w") as f:
                    f.write(str(loop))
                with open(self.project_path + "/loop.txt", "w") as f:
                    f.write(str(loop))
                log_path = f"{self.process_path}/log/"
                self.remove_dirc(log_path)
                if idx == 0 and label_hilts =="file":
                    label = label_hilts
                    # stop_on += 1
                else:
                    label = args.get("labeling")
                if not self.demo:
                    res =  LTS(sampler, data, args.get("sample_size"), True, trainer, labeler, "filename", True, args.get("metric"), args.get("baseline"), label, loop, self.project_path, self.process_path, samplingVersion)
                else:
                    res = self.get_demo_res(loop, args, label)
                if len(res) == 0: ## end of LLM Labeling
                    csvpath =f"data/{self.project_id}/current_sample_training.csv"

                    if os.path.exists(csvpath):
                        df = pd.read_csv(csvpath)
                        image_paths = df["image_path"].to_list()
                        label_llm = ["animal origin" if label == 1 else "not animal origin" for label in df["label"].to_list()]
                        for path, labelllm in zip(image_paths, label_llm):
                            labeldb.add_label(image_path=path,label= labelllm, table="relevant")
                        with open(os.path.join(self.project_path, "state.txt"), "w") as f:
                            f.write("User Labeling")
                    return

                if len(res) > 0: ## end of training loop
                    # Check if the keys in `res` contain "eval_"
                    is_eval_format = any(key.startswith("eval_") for key in res.keys())

                    # Extract metrics based on the format
                    precision_key = "eval_precision" if is_eval_format else "precision"
                    recall_key = "eval_recall" if is_eval_format else "recall"
                    f1_key = "eval_f1" if is_eval_format else "f1"
                    accuracy_key = "eval_accuracy" if is_eval_format else "accuracy"

                    result = {
                        "step": [loop],
                        "precision": [res[precision_key]],
                        "recall": [res[recall_key]],
                        "f1_score": [res[f1_key]],
                        "accuracy": [res[accuracy_key]],
                    }

                    if not result_json:
                        # If no previous metrics exist, create a new metrics file
                        with open(self.metrics_path, "w") as json_file:
                            json.dump(result, json_file, indent=4)
                    else:
                        # Append metrics to the existing result_json
                        result_json["precision"].append(res[precision_key])
                        result_json["recall"].append(res[recall_key])
                        result_json["f1_score"].append(res[f1_key])
                        result_json["accuracy"].append(res[accuracy_key])
                        result_json["step"].append(loop)

                        # Save the updated metrics to the file
                        with open(self.metrics_path, "w") as json_file:
                            json.dump(result_json, json_file, indent=4)
                else:
                    if os.path.exists(self.stop_path):
                        print(f"Removing Stop file {self.stop_path}")
                        os.remove(self.stop_path)
                    final_csv = f"{self.project_path}/filename_data_labeled.csv"
                    if os.path.exists(final_csv):
                        print("Starting final training with complete labeled data...")
                        final_df = pd.read_csv(final_csv)
                        # Use the same trainer and test_data as before
                        results_eval, results_test, trainer = trainer.train_data(final_df, still_unbalanced=False, state_path=self.project_path)
                        trainer.save_model(self.project_path, "final_model")
                        print("Final training results:", results_eval, results_test)
                        with open(self.final_model_res, "w") as json_file:
                            json.dump(results_test, json_file, indent=4)
                    print("LTS Finished!")
                    return # End of LTS process
            final_csv = f"{self.project_path}/filename_data_labeled.csv"
            if os.path.exists(final_csv):
                print("Starting final training with complete labeled data...")
                final_df = pd.read_csv(final_csv)
                # Use the same trainer and test_data as before
                results_eval, results_test, trainer = trainer.train_data(final_df, still_unbalanced=False, state_path=self.project_path)
                trainer.save_model(self.project_path, "final_model")
                print("Final training results:", results_eval, results_test)
                with open(self.final_model_res, "w") as json_file:
                    json.dump(results_test, json_file, indent=4)
            print("LTS Finished!")
    # TODO
        # elif budget=="metric":

    def get_demo_res(self, loop, args, label):
        if label != "file":
            # if loop == 1:
                # # remove any existing state.txt file
                # if os.path.exists(os.path.join(self.project_path, "current_sample_training.csv")):
                #     os.remove(os.path.join(self.project_path, "current_sample_training.csv"))
                # if os.path.exists(os.path.join(self.project_path, "epoch_training.json")):
                #     os.remove(os.path.join(self.project_path, "epoch_training.json"))
                # if os.path.exists(os.path.join(self.project_path, "metrics.json")):
                #     os.remove(os.path.join(self.project_path, "metrics.json"))
                # if os.path.exists(os.path.join(self.project_path, "labels.db")):
                #     os.remove(os.path.join(self.project_path, "labels.db"))
                # if os.path.exists(os.path.join(self.project_path, "hilts")):
                #     os.remove(os.path.join(self.project_path, "hilts"))
                # if os.path.exists(os.path.join(self.project_path, "hilts_data.csv")):
                #     os.remove(os.path.join(self.project_path, "hilts_data.csv"))
            with open(os.path.join(self.project_path, "state.txt"), "w") as f:
                f.write("LLM Labeling")
            csvpath =f"data/{self.project_id}/filename_data_labeled.csv"
            size = loop*args.get("sample_size")
            start = size - args.get("sample_size")
            df = pd.read_csv(csvpath)
            df = df[start:size]
            label_llm = [1 if label == "relevant product" else 0 for label in df["answer"].to_list()]
            df["label"] = label_llm
            df.to_csv(f"{self.project_path}/current_sample_training.csv", index=False)
            return {}

        else:
            # wait for 2 seconds
            # read from epoch_training.json
            epoch_file = os.path.join(self.project_path, "epoch_training_demo.json")
            write_state(self.project_path, "Training")
            with open(epoch_file, 'r') as file:
                if os.path.getsize(epoch_file) > 0:
                    try:
                        epochs = json.load(file)
                    except json.JSONDecodeError:
                        print(f"Malformed JSON in {epoch_file}. Attempting to clean.")
                        epochs = LTSManager.clean_json(epoch_file) or {}
                else:
                    print(f"Warning: {epoch_file} is empty.")
                    epochs = {}
            # based on the loop number get the last epoch result and save in the epoch path so the get_epoch will work
            total_loops = epochs.get(str(loop), [])
            for train in total_loops:
                # Simulate saving epoch data for the current loop
                self.save_epoch_for_loop(train)
                time.sleep(5)
                # read from metrics.json
            metrics_file = os.path.join(self.project_path, "metrics_demo.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, "r") as json_file:
                    result_json_demo = json.load(json_file)
            update_config(self.project_path, {"model_finetune": "bert-base-uncased", "bugetValue": loop, "baseline": 0})


            # get the result from the loop number

            result = {
                "step": [loop],
                "precision": [result_json_demo["precision"][loop-1]],
                "recall": [result_json_demo["recall"][loop-1]],
                "f1": [result_json_demo["f1_score"][loop-1]],
                "accuracy": [result_json_demo["accuracy"][loop-1]],
            }
            return result




    def start_training(self, label_hits):
        """
        Starts the model training in a separate process.
        """
        # Pass only pickleable arguments to the process
        self.process = multiprocessing.Process(
            target=self.train_model,
            args=(label_hits, ),
        )
        self.process.start()
        self.status = self.process.is_alive()
        self.process_id = self.process.pid
        print(f"Training process started with PID: {self.process_id}")

        # Save the process ID to a file
        self.process_path = f"{self.project_path}/{self.process_id}"
        os.makedirs(self.process_path, exist_ok=True)
        return self.process

    def get_status(self):
        status = {
            "loop": -1,
            "lts_status": "",
            "lts_state": "",
            "stats": {
                "llm_labels": [],
                "epochs": {},
                "training_metrics": {},
                "inference_progress": None
            }
        }
        loop = self.get_loop_idx(self.process_path)
        status["lts_status"] = self.process.is_alive()
        status["loop"] = loop

        metrics = self.get_metrics(self.project_path)
        epochs = self.get_epoch_logs(self.project_path, self.process_path, loop)
        labels = self.get_llm_labels(self.project_path)

        status["lts_state"] = self.get_LTS_state(self.project_path)

        # Get inference progress if available
        inference_progress_path = f"{self.project_path}/inference_progress.json"
        if os.path.exists(inference_progress_path):
            with open(inference_progress_path, "r") as f:
                status["stats"]["inference_progress"] = json.load(f)

        if labels:
            status["stats"]["llm_labels"].append(labels)
        if epochs:
            status["stats"]["epochs"] = epochs
        if metrics:
            status["stats"]["training_metrics"]= metrics

        if not status["lts_status"]:
            self.remove_dirc(self.process_path) ## remove process folder after process is done

        return status

    def stop_LTS(self):
        p_process = f"data/{self.project_id}/{self.process_id}/stop.txt"
        with open(p_process, "w") as f:
            f.write("stop")


    @staticmethod
    def get_LTS_state(project_path):
        state_file = os.path.join(project_path, "state.txt")
        if os.path.exists(state_file):
            try:
                with open(state_file, "r") as file:
                    return file.read().strip()
            except Exception as e:
                print(f"Error reading state file: {e}")
                return None
        return None

    @staticmethod
    def get_loop_idx(process_path):
        loop_file = os.path.join(process_path, "loop.txt")
        if os.path.exists(loop_file):
            with open(loop_file, "r") as file:
                return file.read()
        return -1

    @staticmethod
    def get_metrics(project_path):
        metrics_file = os.path.join(project_path, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as json_file:
                return json.load(json_file)
        return None

    @staticmethod
    def get_llm_labels(project_path):
        try:
            labels_file = os.path.join(project_path, "current_sample_training.csv")
            if os.path.exists(labels_file):
                df = pd.read_csv(labels_file)
                count_1 = len(df[df["label"] == 1]) if not df.empty else 0
                count_0 = len(df[df["label"] == 0]) if not df.empty else 0
                return {"1": count_1, "0": count_0}
        except Exception as e:
            print(f"Error reading labels: {e}")
        return None


    @staticmethod
    def get_epoch_logs(project_path, process_path, loop):
        """
        Retrieve and update epoch logs for the current loop.

        Args:
            project_path (str): Path to the project directory.
            process_path (str): Path to the process directory.
            loop (int): Current loop number.

        Returns:
            dict: Updated epoch logs or None if an error occurs.
        """
        log_path = os.path.join(process_path, "log")
        epoch_file = os.path.join(log_path, "epoch.json")
        training_file = os.path.join(project_path, "epoch_training.json")

        try:
            # Check if the epoch file exists
            if os.path.exists(epoch_file):
                # Ensure the file is not empty
                if os.path.getsize(epoch_file) > 0:
                    with open(epoch_file, 'r') as file:
                        epoch_logs = json.load(file)
                else:
                    print(f"Warning: {epoch_file} is empty.")
                    return None

                # Handle the training file for multiple loops
                # if loop == 1 and os.path.exists(training_file):
                #     os.remove(training_file)

                if os.path.exists(training_file):
                    with open(training_file, 'r') as file:
                        if os.path.getsize(training_file) > 0:
                            try:
                                epochs = json.load(file)
                            except json.JSONDecodeError:
                                print(f"Malformed JSON in {training_file}. Attempting to clean.")
                                epochs = LTSManager.clean_json(training_file) or {}
                        else:
                            print(f"Warning: {training_file} is empty.")
                            epochs = {}
                    epochs[f"{loop}"] = epoch_logs
                else:
                    epochs = {f"{loop}": epoch_logs}

                # Validate JSON before saving
                if LTSManager.validate_json(epochs):
                    with open(training_file, 'w') as file:
                        json.dump(epochs, file, indent=4)
                else:
                    print(f"Invalid JSON data. Skipping save for {training_file}.")
                    return None

                return epochs

            # Handle the case where the training file exists but the epoch file does not
            elif os.path.exists(training_file):
                with open(training_file, 'r') as file:
                    if os.path.getsize(training_file) > 0:
                        try:
                            return json.load(file)
                        except json.JSONDecodeError:
                            print(f"Malformed JSON in {training_file}. Attempting to clean.")
                            return LTSManager.clean_json(training_file)
                    else:
                        print(f"Warning: {training_file} is empty.")
                        return None

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in {epoch_file} or {training_file}: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error in get_epoch_logs: {e}")
            return None

        return None

    @staticmethod
    def remove_dirc(path):
        # Check if directory exists
        if os.path.exists(path):
            # Iterate through the directory and remove files/subdirectories
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove subdirectory
                else:
                    os.remove(file_path)  # Remove file

            # Now remove the empty directory
            os.rmdir(path)

    @staticmethod
    def validate_json(data):
        """
        Validate if the given data can be serialized to JSON.

        Args:
            data (dict): The data to validate.

        Returns:
            bool: True if valid, False otherwise.
        """
        try:
            json.dumps(data)
            return True
        except (TypeError, ValueError) as e:
            print(f"Invalid JSON data: {e}")
            return False

    @staticmethod
    def clean_json(file_path):
        """
        Attempt to clean a malformed JSON file by removing duplicate keys or extra data.

        Args:
            file_path (str): Path to the JSON file.

        Returns:
            dict: The cleaned JSON data, or None if the file cannot be cleaned.
        """
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Attempt to parse each line as JSON
            cleaned_data = []
            for line in lines:
                try:
                    cleaned_data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

            # Combine all valid JSON objects into a single dictionary
            combined_data = {}
            for entry in cleaned_data:
                combined_data.update(entry)

            return combined_data
        except Exception as e:
            print(f"Failed to clean JSON file {file_path}: {e}")
            return None

    def save_epoch_for_loop(self, train_res):
        """
        Simulate saving the epoch data for a specific loop in the process path.

        Args:
            train_res (dict): The epoch data to save.
        """
        # Path to the epoch.json file
        log_epoch_file = os.path.join(self.process_path, "log", "epoch.json")

        # Ensure the log directory exists
        os.makedirs(os.path.dirname(log_epoch_file), exist_ok=True)

        # Read the existing JSON file if it exists
        if os.path.exists(log_epoch_file):
            try:
                with open(log_epoch_file, "r") as log_file:
                    existing_data = json.load(log_file)
            except json.JSONDecodeError:
                print(f"Malformed JSON in {log_epoch_file}. Starting with an empty list.")
                existing_data = []
        else:
            existing_data = []

        # Append the new epoch data
        if isinstance(existing_data, list):
            existing_data.append(train_res)
        else:
            print(f"Unexpected format in {log_epoch_file}. Overwriting with a new list.")
            existing_data = [train_res]

        # Save the updated JSON data back to the file
        with open(log_epoch_file, "w") as log_file:
            json.dump(existing_data, log_file, indent=4)
        print(f"Appended epoch data to {log_epoch_file}")







