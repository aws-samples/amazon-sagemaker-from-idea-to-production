"""Register step — MLflow Model Registry sync with inference specification logging.

This follows the "inference specification logging" pattern: instead of repacking
the model with ModelBuilder and mutating the auto-created Model Package after
the fact, we make the MLflow artifact store the single source of truth:

1. Resolve the MLflow *logged model* produced by the training run.
2. Upload our own ``inference.py`` (pipeline_steps/inference.py, shipped with
   the step code via IncludeLocalWorkDir) into the logged model's artifact
   store under ``code/`` using ``MlflowClient.log_model_artifacts()``.
3. Log an *inference specification* on the logged model with the
   ``sagemaker-mlflow`` plugin (>= 0.5.0): container image, an ``S3Prefix``
   ModelDataSource pointing at the MLflow artifact location, and the
   ``SAGEMAKER_PROGRAM`` / ``SAGEMAKER_SUBMIT_DIRECTORY`` environment variables.
4. Call ``mlflow.register_model()``. Because the MLflow app runs with
   ``AutoModelRegistrationEnabled`` and the spec was logged FIRST, the synced
   SageMaker Model Package is created *already deployable* — serving directly
   from the MLflow artifact store, no repacking, no artifact copies.

The endpoint downloads everything under the model's artifact location to
/opt/ml/model (MLmodel, model file, conda.yaml, code/inference.py), so
``inference.py`` can load the model with ``mlflow.xgboost.load_model()``.
"""
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from time import gmtime, strftime

import boto3
import mlflow

try:
    from pipeline_steps.runtime_utils import log_runtime_info
except ImportError:
    def log_runtime_info(): pass


def _resolve_logged_model(mlflow_client, mlflow_run_id):
    """Return the MLflow LoggedModel entity produced by the training run."""
    run = mlflow_client.get_run(mlflow_run_id)
    model_outputs = run.outputs.model_outputs if run.outputs else []
    if not model_outputs:
        raise RuntimeError(
            f"MLflow run {mlflow_run_id} has no logged model outputs — "
            "was the model logged (e.g. via mlflow.xgboost.log_model)?"
        )
    return mlflow_client.get_logged_model(model_outputs[0].model_id)


def _log_inference_code(mlflow_client, model_id):
    """Upload our own inference.py into the logged model's artifact store.

    The script lives next to this module (pipeline_steps/inference.py) and is
    shipped into the step container by IncludeLocalWorkDir. Logging it through
    MLflow (rather than a raw S3 upload) keeps the artifact store as the single
    source of truth, and log_model_artifacts preserves the directory layout, so
    code/inference.py lands at <artifact_location>/code/inference.py — which the
    S3Prefix ModelDataSource downloads to /opt/ml/model/code/inference.py.
    """
    script_path = Path(__file__).parent / "inference.py"
    if not script_path.exists():
        raise FileNotFoundError(
            f"{script_path} not found — provide pipeline_steps/inference.py "
            "with at least a model_fn(model_dir)."
        )
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "code"))
        shutil.copy(script_path, os.path.join(tmp, "code", "inference.py"))
        mlflow_client.log_model_artifacts(model_id, tmp)
    print(f"## Logged code/inference.py from {script_path}")


def _log_inference_specification(model_id, artifact_location, container_image):
    """Attach the inference specification to the logged model.

    Must run BEFORE mlflow.register_model(): the auto-sync copies the spec onto
    the SageMaker Model Package at registration time.
    """
    import sagemaker_mlflow  # requires sagemaker-mlflow >= 0.5.0

    inference_spec = {
        "Containers": [{
            "Image": container_image,
            # Serve directly from the MLflow artifact store — no repacking.
            "ModelDataSource": {
                "S3DataSource": {
                    "S3Uri": artifact_location.rstrip("/") + "/",
                    "S3DataType": "S3Prefix",
                    "CompressionType": "None",
                }
            },
            # Point the container at our inference script. The BYOC serving
            # stack loads code/inference.py and dispatches to its
            # model_fn / input_fn / predict_fn / output_fn.
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
            },
        }],
        "SupportedContentTypes": ["text/csv", "application/json"],
        "SupportedResponseMIMETypes": ["text/csv", "application/json"],
        "SupportedRealtimeInferenceInstanceTypes": ["ml.m5.large", "ml.m5.xlarge"],
        "SupportedTransformInstanceTypes": ["ml.m5.large", "ml.m5.xlarge"],
    }
    sagemaker_mlflow.log_inference_specification(
        model_id, inference_specification=inference_spec
    )
    print("## Inference specification logged on the MLflow model")


def register(
    model_package_group_name,
    model_approval_status,
    evaluation_result,
    mlflow_run_id=None,
    training_job_name=None,
    pipeline_run_id=None,
):
    """Register the trained model: log inference spec on the MLflow model, then
    register it — auto-sync creates a deployable SageMaker Model Package."""
    try:
        suffix = strftime('%d-%H-%M-%S', gmtime())
        sm_client = boto3.client("sagemaker")
        container_image = os.environ.get("CONTAINER_IMAGE", "")
        mlflow_client = mlflow.MlflowClient()

        pipeline_run = mlflow.start_run(run_id=pipeline_run_id) if pipeline_run_id else None
        run = mlflow.start_run(run_name=f"register-{suffix}", nested=True)
        log_runtime_info()

        # Log evaluation result
        with open("evaluation.json", "w") as f:
            f.write(json.dumps(evaluation_result))
        mlflow.log_artifact(local_path="evaluation.json")

        if not mlflow_run_id:
            raise ValueError(
                "mlflow_run_id is required — the training step returns it "
                "(see pipeline_steps/train_fn.py)."
            )

        # Step 1: Resolve the logged model created by the training run
        logged_model = _resolve_logged_model(mlflow_client, mlflow_run_id)
        print(f"## Logged model: {logged_model.model_id} at {logged_model.artifact_location}")

        # Step 2: Upload our inference.py into the model's artifact store
        _log_inference_code(mlflow_client, logged_model.model_id)

        # Step 3: Log the inference specification BEFORE registering, so the
        # auto-synced Model Package is created already deployable.
        # (Skipped for notebook-local test runs, where no serving image is set.)
        if container_image and container_image != "local":
            _log_inference_specification(
                logged_model.model_id, logged_model.artifact_location, container_image
            )
        else:
            print("## CONTAINER_IMAGE is local/unset — skipping inference specification")

        # Step 4: Register in the MLflow Model Registry. AutoModelRegistrationEnabled
        # syncs it to the SageMaker Model Registry as a new Model Package version
        # under the same (single, shared) model package group.
        model_version = mlflow.register_model(
            model_uri=f"models:/{logged_model.model_id}", name=model_package_group_name
        )
        print(f"## Registered in MLflow: {model_version.name} v{model_version.version}")

        # Step 5: The sync records the Model Package ARN as a tag on the MLflow
        # model version — much simpler than searching the registry.
        sm_model_package_arn = None
        for _ in range(20):
            mv = mlflow_client.get_model_version(model_version.name, model_version.version)
            sm_model_package_arn = mv.tags.get("sagemaker.model_package_arn")
            if sm_model_package_arn:
                break
            time.sleep(3)
            print(".", end="", flush=True)
        if not sm_model_package_arn:
            raise TimeoutError("SageMaker Model Package was not auto-created within timeout")
        sm_model_package_group = sm_model_package_arn.split("/")[1]
        print(f"\n## Synced SageMaker Model Package: {sm_model_package_arn}")

        # Step 6: Set approval status and traceability metadata. The inference
        # specification is already on the package — no spec mutation needed.
        sm_client.update_model_package(
            ModelPackageArn=sm_model_package_arn,
            ModelApprovalStatus=model_approval_status,
            CustomerMetadataProperties={
                "TrainingJobName": training_job_name or "local",
                "ContainerImage": container_image or "n/a",
                "MlflowRunId": mlflow_run_id or "",
                "MlflowModelId": logged_model.model_id,
                "MlflowArtifactLocation": logged_model.artifact_location,
            },
        )
        print(f"## Model Package status set to: {model_approval_status}")

        mlflow.log_params({
            "registered_model_name": model_version.name,
            "registered_model_version": model_version.version,
            "sm_model_package_arn": sm_model_package_arn,
            "sm_model_package_group": sm_model_package_group,
            "model_artifact_location": logged_model.artifact_location,
        })

        return {
            "model_package_group_name": sm_model_package_group,
            "model_package_arn": sm_model_package_arn,
            "registered_model_name": model_version.name,
            "registered_model_version": str(model_version.version),
            "pipeline_run_id": pipeline_run.info.run_id if pipeline_run else '',
        }

    except Exception as e:
        print(f"Exception in register: {e}")
        raise e
    finally:
        while mlflow.active_run():
            mlflow.end_run()
