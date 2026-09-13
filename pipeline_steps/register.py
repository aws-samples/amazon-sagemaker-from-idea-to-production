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
5. Move the version to its lifecycle stage (default ``staging`` / ``pending``) by
   setting the MLflow alias ``sagemakerlifecycle-<stage>-<status>``, which the sync
   turns into the Model Package's ``ModelLifeCycle``. The step never asks for
   ``production``: that is the governance officer's decision, and the
   data-scientist guardrail (``sagemaker:ModelLifeCycle/stage = production``)
   denies it.
6. Record the traceability a reviewer needs. SageMaker supports tag operations on
   Model Package *Groups* only, so group-wide facts (project, use case, team) are
   resource tags on the group, while per-version facts (who registered it, from
   where, the test AUC) go into the version's ``CustomerMetadataProperties``.

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

# governance/ ships with the step code (IncludeLocalWorkDir).
from governance.approval_gate import (
    set_lifecycle,
    set_version_metadata,
    tag_model_package_group,
    wait_for_lifecycle,
)


def _resolve_logged_model(mlflow_client, mlflow_run_id):
    """Return the MLflow LoggedModel entity produced by the training run."""
    run = mlflow_client.get_run(mlflow_run_id)
    model_outputs = run.outputs.model_outputs if run.outputs else []
    if not model_outputs:
        raise RuntimeError(
            f"MLflow run {mlflow_run_id} has no logged model outputs — "
            "was the model logged (e.g. via mlflow.xgboost.autolog)?"
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


def _ensure_registered_model(mlflow_client, name, group_tags):
    """Create the MLflow registered model up front so the synced SageMaker Model
    Package Group is created WITH the group-level tags. mlflow.register_model()
    would create it too, but without tags. A no-op when the model exists."""
    from mlflow.exceptions import MlflowException

    tags = {k: str(v) for k, v in (group_tags or {}).items() if v not in (None, "")}
    try:
        mlflow_client.create_registered_model(
            name,
            tags=tags,
            description="Bank-marketing propensity model (XGBoost). Registered by the from-idea-to-prod workshop.",
        )
        print(f"## Created registered model '{name}' with group tags {sorted(tags)}")
    except MlflowException as e:
        if e.error_code != "RESOURCE_ALREADY_EXISTS":
            raise
        print(f"## Registered model '{name}' already exists — reusing it")


def _set_lifecycle_stage(mlflow_client, sm_client, model_version, model_package_arn, stage, status):
    """Move the new version to ``stage``/``status`` from MLflow, by alias.

    Lifecycle stage is driven from MLflow: setting the alias
    ``sagemakerlifecycle-<stage>-<status>`` makes the sync update the Model Package's
    ``ModelLifeCycle``. A version holds one lifecycle alias at a time, so any existing
    one is removed first.

    The step never asks for ``production``: promotion is the governance officer's
    decision, and the data-scientist guardrail
    (``sagemaker:ModelLifeCycle/stage = production``) denies it."""
    alias = set_lifecycle(mlflow_client, model_version.name, model_version.version, stage, status)
    print(f"## MLflow alias set: {alias}")
    return wait_for_lifecycle(sm_client, model_package_arn, stage, status)


def register(
    model_package_group_name,
    model_approval_status,
    evaluation_result,
    mlflow_run_id=None,
    training_job_name=None,
    pipeline_run_id=None,
    lifecycle_stage="staging",
    lifecycle_status="pending",
    tags=None,
    group_tags=None,
):
    """Register the trained model: log inference spec on the MLflow model, then
    register it — auto-sync creates a deployable SageMaker Model Package — and
    finally move it to ``lifecycle_stage``/``lifecycle_status`` and record its
    provenance.

    ``tags`` are recorded on the Model Package version as
    ``CustomerMetadataProperties`` (merged with the provenance this step derives
    itself); ``group_tags`` become resource tags of the Model Package Group, because
    SageMaker supports tag operations on groups only, not on individual versions.

    The step stops at ``staging``: promotion to ``production`` is the governance
    officer's decision, and the data-scientist guardrail denies it."""
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

        # Fallback: resolve mlflow_run_id from the training job's model.tar.gz if it
        # wasn't passed directly. This applies when training runs as a script via
        # TrainingStep and writes mlflow_run_id.txt into model.tar.gz. In the @step
        # pipeline and the CI/CD pipeline (notebook 04), mlflow_run_id is passed directly.
        if not mlflow_run_id and training_job_name and training_job_name != "local":
            try:
                import s3fs, tarfile
                desc = sm_client.describe_training_job(TrainingJobName=training_job_name)
                s3 = s3fs.S3FileSystem()
                local_tar = tempfile.mktemp(suffix=".tar.gz")
                with s3.open(desc["ModelArtifacts"]["S3ModelArtifacts"], "rb") as remote, open(local_tar, "wb") as local:
                    local.write(remote.read())
                with tarfile.open(local_tar, "r:gz") as tar:
                    f = tar.extractfile("mlflow_run_id.txt")
                    if f:
                        mlflow_run_id = f.read().decode().strip()
                        print(f"## Retrieved MLflow run_id from model.tar.gz: {mlflow_run_id}")
            except Exception as e:
                print(f"## Could not extract mlflow_run_id: {e}")

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
        # under the same (single, shared) model package group. Creating the
        # registered model explicitly first lets the group carry its tags.
        _ensure_registered_model(mlflow_client, model_package_group_name, group_tags)
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
        # EvaluationMetrics is picked up by the EventBridge model-card automation
        # (notebook 02) and rendered into the card's evaluation_details.
        evaluation_metrics = {
            name: float(metric["value"])
            for name, metric in evaluation_result.get("classification_metrics", {}).items()
            if isinstance(metric, dict) and "value" in metric
        }
        sm_client.update_model_package(
            ModelPackageArn=sm_model_package_arn,
            ModelApprovalStatus=model_approval_status,
            CustomerMetadataProperties={
                "TrainingJobName": training_job_name or "local",
                "ContainerImage": container_image or "n/a",
                "MlflowRunId": mlflow_run_id or "",
                "MlflowModelId": logged_model.model_id,
                "MlflowArtifactLocation": logged_model.artifact_location,
                "EvaluationMetrics": json.dumps(evaluation_metrics)[:1024],
            },
        )
        print(f"## Model Package status set to: {model_approval_status}")

        # Step 7: Lifecycle stage, driven from MLflow by alias. The step stops at
        # staging; promotion to production is the governance officer's call.
        lifecycle = _set_lifecycle_stage(
            mlflow_client, sm_client, model_version, sm_model_package_arn,
            lifecycle_stage.lower(), lifecycle_status.lower(),
        )

        # Step 8: Provenance. SageMaker supports tag operations on Model Package
        # *Groups* only, so per-version facts go into CustomerMetadataProperties
        # (visible on the version in Studio) and group-wide facts become resource
        # tags on the group — which is also what IAM aws:ResourceTag conditions see.
        is_job = bool(training_job_name and training_job_name != "local")
        version_metadata = {
            "RegistrationSource": "sagemaker-pipeline" if is_job else "notebook-local",
            "RegisteredBy": os.environ.get("USER", "") or "unknown",
            "MlflowExperiment": os.environ.get("MLFLOW_EXPERIMENT_NAME", ""),
            "LifecycleStage": f"{lifecycle.get('Stage')}/{lifecycle.get('StageStatus')}",
            "TestAuc": f"{evaluation_metrics['auc_score']:.4f}" if "auc_score" in evaluation_metrics else "",
        }
        version_metadata.update(tags or {})
        set_version_metadata(sm_client, sm_model_package_arn, version_metadata)
        print(f"## Recorded on the version: {sorted(k for k, v in version_metadata.items() if v)}")

        if group_tags:
            tag_model_package_group(sm_client, sm_model_package_arn, group_tags)
            print(f"## Tagged the Model Package Group with {sorted(group_tags)}")

        mlflow.log_params({
            "registered_model_name": model_version.name,
            "registered_model_version": model_version.version,
            "sm_model_package_arn": sm_model_package_arn,
            "sm_model_package_group": sm_model_package_group,
            "model_artifact_location": logged_model.artifact_location,
            "lifecycle_stage": f"{lifecycle.get('Stage')}/{lifecycle.get('StageStatus')}",
        })

        return {
            "model_package_group_name": sm_model_package_group,
            "model_package_arn": sm_model_package_arn,
            "registered_model_name": model_version.name,
            "registered_model_version": str(model_version.version),
            "lifecycle_stage": lifecycle.get("Stage", ""),
            "lifecycle_status": lifecycle.get("StageStatus", ""),
            "pipeline_run_id": pipeline_run.info.run_id if pipeline_run else '',
        }

    except Exception as e:
        print(f"Exception in register: {e}")
        raise e
    finally:
        while mlflow.active_run():
            mlflow.end_run()
