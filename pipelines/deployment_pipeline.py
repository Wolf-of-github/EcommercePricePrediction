import numpy as np
import pandas as pd

from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output

from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_train import train_model

# Docker settings for MLflow
docker_settings = DockerSettings(required_integrations=[MLFLOW])

# Deployment trigger step that takes in accuracy and min_accuracy directly
@step
def deployment_trigger(accuracy: float, min_accuracy: float) -> bool:
    return accuracy > min_accuracy

# Continuous deployment pipeline
@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    data_path: str,
    min_accuracy: float = 0,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT
):
    # Data ingestion
    df = ingest_df(data_path='./data/olist_customers_dataset.csv')

    # Data cleaning
    X_train, X_test, y_train, y_test = clean_df(df)

    # Model training
    model = train_model(X_train, X_test, y_train, y_test)

    # Model evaluation
    r2_score, rmse = evaluate_model(model, X_test, y_test)

    # Deployment decision (pass min_accuracy directly)
    deployment_decision = deployment_trigger(
        accuracy=r2_score,
        min_accuracy=min_accuracy  # Pass min_accuracy directly
    )

    # Model deployment step
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout
    )