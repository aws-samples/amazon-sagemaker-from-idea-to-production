"""Utility to log container/runtime info to MLflow."""
import os
import mlflow


def log_runtime_info():
    """Log container and runtime information as MLflow tags."""
    # CONTAINER_IMAGE is set via config.yaml EnvironmentVariables
    container_image = os.environ.get("CONTAINER_IMAGE", "")
    
    if not container_image:
        # Fallback: check SAGEMAKER_TRAINING_MODULE
        training_module = os.environ.get("SAGEMAKER_TRAINING_MODULE", "")
        if "xgboost" in training_module:
            container_image = "sagemaker-xgboost"
        elif "sklearn" in training_module:
            container_image = "sagemaker-sklearn"
        else:
            container_image = "local" if not os.environ.get("SM_INPUT") else "n/a"

    job_name = os.environ.get("TRAINING_JOB_NAME",
               os.environ.get("PROCESSING_JOB_NAME", "local"))

    mlflow.set_tag("sagemaker.container_image", container_image)
    mlflow.set_tag("sagemaker.job_name", job_name)
    mlflow.set_tag("sagemaker.instance_type", os.environ.get("SM_CURRENT_INSTANCE_TYPE", "local"))
    mlflow.set_tag("sagemaker.runtime", "sagemaker" if job_name != "local" else "local")
