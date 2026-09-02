import json

import mlflow
import tempfile
import os
import wandb
import hydra

steps = [
    "data_ingestion",
    "preprocessing",
    "tests",
    "data_split",
    "train_random_forest"]


# This automatically reads in the configuration
@hydra.main(version_base=None, config_name='config', config_path='.') 
def go(config):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else steps


    if "data_ingestion" in active_steps:
        # Download file and load in W&B
        _ = mlflow.run(
            "components/data_ingestion",
            "main",
            env_manager="conda",
            parameters={
                "sample": config["etl"]["sample"],
                "artifact_name": "dataset.csv",
                "artifact_type": "raw_dataset",
                "artifact_description": "Raw file as downloaded"
            },
        )

    if "preprocessing" in active_steps:
        _ = mlflow.run(
            "components/preprocessing",
            "main",
            env_manager="conda",
            parameters={
                "input_artifact": "dataset.csv:latest",
                "output_artifact": "clean_dataset.csv",
                "output_type": "clean_dataset",
                "output_description": "Data with outliers and null values removed",
            },
        )

    if "tests" in active_steps:
        _ = mlflow.run(
            "components/tests",
            "main",
            env_manager="conda",
            parameters={
                "csv": "clean_dataset.csv:latest",
                "ref": "clean_dataset.csv:reference",
                "kl_threshold": config["data_check"]["kl_threshold"],
            },
        )

    if "data_split" in active_steps:
        _ = mlflow.run(
            "components/data_split",
            "main",
            env_manager="conda",
            parameters={
                "input": "clean_dataset.csv:latest",
                "test_size": config["modeling"]["test_size"],
                "random_seed": config["modeling"]["random_seed"],
                "stratify_by": config["modeling"]["stratify_by"]
            },
        )

    if "train_random_forest" in active_steps:

        # NOTE: we need to serialize the random forest configuration into JSON
        rf_config = os.path.abspath("rf_config.json")
        with open(rf_config, "w+") as fp:
            json.dump(dict(config["modeling"]["random_forest"].items()), fp)

        _ = mlflow.run(
            "src/train_random_forest",
            "main",
            env_manager="conda",
            parameters={
                "trainval_artifact": "trainval_data.csv:latest",
                "val_size": config["modeling"]["val_size"],
                "random_seed": config["modeling"]["random_seed"],
                "stratify_by": config["modeling"]["stratify_by"],
                "rf_config": rf_config,
                "max_tfidf_features": config["modeling"]["max_tfidf_features"],
                "output_artifact": 'random_forest_export'
            },
        )


    if "test_regression_model" in active_steps:

        _ = mlflow.run(
            "components/test_regression_model",
            "main",
            env_manager="conda",
            parameters={
                "mlflow_model": "random_forest_export:prod",
                "test_dataset": "test_data.csv:latest"
            },
        )


if __name__ == "__main__":
    go()
