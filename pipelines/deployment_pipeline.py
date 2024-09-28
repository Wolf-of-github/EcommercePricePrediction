import json
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

from pipelines.utils import get_data_for_test
from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_train import train_model

# Docker settings for MLflow
docker_settings = DockerSettings(required_integrations=[MLFLOW])

@step(enable_cache=False)
def dynamic_importer() -> str:
    data = get_data_for_test()
    return data

# Deployment trigger step that takes in accuracy and min_accuracy directly
@step
def deployment_trigger(accuracy: float, min_accuracy: float) -> bool:
    return accuracy > min_accuracy

@step(enable_cache = False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = 'model'
) -> MLFlowDeploymentService:
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name= pipeline_step_name,
        model_name= model_name,
        running=running
    )

    if not existing_services:
        raise RuntimeError(
        f"No MLflow deployment service found for pipeline {pipeline_name}," 
        f"step {pipeline_step_name} and model {model_name}."
        f"pipeline for the '{model_name}' model is currently"
        f"running"
        )
    return existing_services[0]


@step
def predictor(
    service: MLFlowDeploymentService,
    data: str,  
)->np.ndarray:


    # Start the service with a timeout of 10 seconds
    service.start(timeout=10)

    # Load data from JSON format
    data = json.loads(data)

    # Remove unnecessary keys
    data.pop("columns")
    data.pop("index")

    # Define the columns to use for the DataFrame
    columns_for_df = [
        "payment_sequential",
        "payment_installments",
        "payment_value",
        "price",
        "freight_value",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]

    # Check if the service is already started (NOP - No Operation if started)
    if service.is_running:  # Assuming there's an attribute or method `is_running` to check
        print("Service is already running")

    # Convert the JSON data to a DataFrame
    df = pd.DataFrame(data["data"], columns=columns_for_df)

    # Convert DataFrame to JSON list
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))

    # Convert the JSON list to a numpy array
    data = np.array(json_list)

    # Make predictions using the service
    prediction = service.predict(data)

    # Return the predictions
    return prediction

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

@pipeline(enable_cache=False, settings = {"docker": docker_settings})
def interface_pipeline(pipeline_name: str, pipeline_step_name: str):
    data = dynamic_importer()
    service = prediction_service_loader(
        pipeline_name = pipeline_name,
        pipeline_step_name = pipeline_step_name,
        running = False
    )

    prediction = predictor(service = service, data = data)
    return prediction