import json
import mlflow
import os
import wandb
import hydra

steps = [
    "data_ingestion",
    "preprocessing",
    "eda",
    "test_data",
    "data_split",
    "train",
    "test_model"]


# This automatically reads in the configuration
@hydra.main(version_base=None, config_name='config', config_path='.') 
def go(config):
    """
    Execute the configured ML pipeline steps using Hydra, MLflow, and W&B.

    Args:
        config: Hydra configuration object containing project settings,
            pipeline steps, ETL parameters, data checks, and modeling parameters.
    """

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

    if "eda" in active_steps:
        _ = mlflow.run(
            "components/eda",
            "main",
            env_manager="conda",
            parameters={
                "input_artifact": "clean_dataset.csv:latest",
                "output_artifact": "eda_report.html",
                "output_type": "eda_report",
                "output_description": "Exploratory data analysis report"
            },
        )

    if "test_data" in active_steps:
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

    if "train" in active_steps:

        _ = mlflow.run(
            "components/train",
            "main",
            env_manager="conda",
            parameters={
                "trainval_artifact": "trainval_data.csv:latest",
                "val_size": config["modeling"]["val_size"],
                "random_seed": config["modeling"]["random_seed"],
                "stratify_by": config["modeling"]["stratify_by"],
                "output_artifact": 'random_forest_export'
            },
        )


    if "test_model" in active_steps:

        _ = mlflow.run(
            "components/test_model",
            "main",
            env_manager="conda",
            parameters={
                "mlflow_model": "random_forest_export:latest",
                "test_dataset": "test_data.csv:latest"
            },
        )


if __name__ == "__main__":
    go()