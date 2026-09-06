"""
This script trains a Random Forest
"""
import argparse
import logging
import os
import shutil
import matplotlib.pyplot as plt
import mlflow
import yaml
import pandas as pd
import numpy as np
import wandb

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline, make_pipeline


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    """
    Train, evaluate, and export a Random Forest classification pipeline.

    Args:
        args: Command-line arguments containing the training artifact,
            validation size, random seed, stratification setting, and
            output artifact name.
    """

    run = wandb.init(job_type="train")
    run.config.update(args)

    # Get the Random Forest configuration and update W&B
    with open('../../config.yaml') as f:
        rf_config = yaml.safe_load(f)['modeling']['random_forest']
    run.config.update(rf_config)

    # Fix the random seed for the Random Forest, so we get reproducible results
    rf_config['random_state'] = args.random_seed

    # Get the train and validation artifact
    trainval_local_path = run.use_artifact(args.trainval_artifact).file()

    X = pd.read_csv(trainval_local_path)
    y = X.pop("y")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_size, stratify=y, random_state=args.random_seed)

    logger.info("Preparing sklearn pipeline")

    sk_pipe, processed_features = get_inference_pipeline(rf_config)

    # Then fit it to the X_train, y_train data
    logger.info("Fitting")

    # Fit the pipeline sk_pipe by calling the .fit method on X_train and
    # y_train
    sk_pipe.fit(X_train, y_train)

    # Compute r2 and MAE
    logger.info("Metrics Computation")

    y_pred = sk_pipe.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, pos_label='yes')
    recall = recall_score(y_val, y_pred, pos_label='yes')
    f1 = f1_score(y_val, y_pred, pos_label='yes')

    logger.info(f"Accuracy: {accuracy}")
    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")
    logger.info(f"F1 Score: {f1}")

    logger.info("Exporting model")

    # Save model package in the MLFlow sklearn format
    if os.path.exists("../../model"):
        shutil.rmtree("../../model")

    # Save the sk_pipe pipeline as a mlflow.sklearn model
    mlflow.sklearn.save_model(sk_pipe, "../../model")

    # Upload the model we just exported to W&B
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type="model_export",
        description="Random Forest model exported in MLFlow format",
        metadata=rf_config
    )

    artifact.add_dir("../../model")
    run.log_artifact(artifact)

    # Plot feature importance
    fig_conf_mtrx = plot_confusion_matrix(y_val, y_pred)

    # Saving the confusion matrix figure locally
    with open("../../visualizations/confusion_matrix.png", "wb") as f:
        fig_conf_mtrx.savefig(f, format="png")

    # Save metrics results to W&B summary
    run.summary['accuracy'] = accuracy
    run.summary['precision'] = precision
    run.summary['recall'] = recall
    run.summary['f1'] = f1

    # Upload to W&B the confusion matrix figure
    run.log({"confusion_matrix": wandb.Image(fig_conf_mtrx)})


def plot_confusion_matrix(y_val, y_pred):
    """
    Create a confusion matrix figure from validation labels and predictions.

    Args:
        y_val: True target values from the validation dataset.
        y_pred: Predicted target values from the trained model.

    Returns:
        matplotlib.figure.Figure: Figure containing the confusion matrix.
    """

    cm = confusion_matrix(y_val, y_pred, labels=["no", "yes"])
    fig, ax = plt.subplots(figsize=(6, 6))

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "No",
            "Yes"]).plot(
        ax=ax)
    return fig


def get_inference_pipeline(rf_config):
    """
    Create the preprocessing and Random Forest inference pipeline.

    Args:
        rf_config: Dictionary containing the Random Forest model configuration.

    Returns:
        tuple: The sklearn pipeline and the list of processed feature names.
    """

    # Let's handle the categorical features first
    # Ordinal categorical are categorical values for which the order is
    # meaningful
    ordinal_cat_cols = ["education"]
    non_ordinal_cat_cols = [
        "job",
        "marital",
        "default",
        "housing",
        "loan",
        "contact",
        "poutcome",
        "month",
        "day_of_week"]

    ordinal_cat_preproc = OrdinalEncoder()
    non_ordinal_cat_preproc = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy="most_frequent")),
            ('encoder', OneHotEncoder())
        ]
    )

    # Let's impute the numerical columns to make sure we can handle missing
    # values
    median_imputed_cols = [
        "age",
        "duration",
        "campaign",
        "pdays",
        "emp.var.rate",
        "cons.price.idx",
        "cons.conf.idx",
        "euribor3m",
        "nr.employed"]

    zero_imputed_cols = ["previous"]

    median_imputer = SimpleImputer(strategy="median")
    zero_imputer = SimpleImputer(strategy="constant", fill_value=0)

    # putting everything together
    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal_cat", ordinal_cat_preproc, ordinal_cat_cols),
            ("non_ordinal_cat", non_ordinal_cat_preproc, non_ordinal_cat_cols),
            ("impute_zero", zero_imputer, zero_imputed_cols),
            ("impute_median", median_imputer, median_imputed_cols)],
    )

    processed_features = ordinal_cat_cols + non_ordinal_cat_cols + \
        zero_imputed_cols + median_imputed_cols

    # Create random forest
    random_forest = RandomForestClassifier(**rf_config)

    sk_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("random_forest", random_forest)
        ]
    )

    return sk_pipe, processed_features


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Basic cleaning of dataset")

    parser.add_argument(
        "--trainval_artifact",
        type=str,
        help="Artifact containing the training dataset. It will be split into train and validation"
    )

    parser.add_argument(
        "--val_size",
        type=float,
        help="Size of the validation split. Fraction of the dataset, or number of items",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        help="Seed for random number generator",
        default=42,
        required=False,
    )

    parser.add_argument(
        "--stratify_by",
        type=str,
        help="Column to use for stratification",
        default="none",
        required=False,
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output serialized model",
        required=True,
    )

    args = parser.parse_args()

    go(args)
