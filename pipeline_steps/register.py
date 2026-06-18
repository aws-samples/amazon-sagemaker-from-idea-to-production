import os
import json
import time
import boto3
import mlflow
from time import gmtime, strftime
try:
    from pipeline_steps.runtime_utils import log_runtime_info
except ImportError:
    def log_runtime_info(): pass


def _build_model_artifacts(container_image, model_uri, tracking_uri):
    """Use ModelBuilder to create deployable S3 artifacts with InferenceSpec."""
    from sagemaker.serve import ModelBuilder
    from sagemaker.serve.spec.inference_spec import InferenceSpec
    from sagemaker.serve.builder.schema_builder import SchemaBuilder
    from sagemaker.serve.utils.types import ModelServer

    class XGBoostMLflowSpec(InferenceSpec):
        def __init__(self, mlflow_model_uri, tracking_uri):
            self._mlflow_model_uri = mlflow_model_uri
            self._tracking_uri = tracking_uri

        def load(self, model_dir=None):
            import mlflow as _mlflow
            _mlflow.set_tracking_uri(self._tracking_uri)
            return _mlflow.xgboost.load_model(self._mlflow_model_uri)

        def invoke(self, input_object, model):
            import xgboost as xgb
            import numpy as np
            if isinstance(input_object, str):
                rows = [list(map(float, r.split(","))) for r in input_object.strip().split("\n") if r]
                dmatrix = xgb.DMatrix(np.array(rows))
            elif isinstance(input_object, list):
                dmatrix = xgb.DMatrix(np.array(input_object))
            else:
                dmatrix = xgb.DMatrix(np.array(input_object))
            return "\n".join(map(str, model.predict(dmatrix)))

    schema_builder = SchemaBuilder(
        sample_input="0.0," * 58 + "0.0",
        sample_output="0.5",
    )

    model_builder = ModelBuilder(
        image_uri=container_image,
        instance_type="ml.m5.large",
        inference_spec=XGBoostMLflowSpec(model_uri, tracking_uri),
        schema_builder=schema_builder,
        model_server=ModelServer.TORCHSERVE,
        dependencies={"auto": False},
    )
    model_builder.build()

    # Get the S3 model data URL and secret key
    model_data_url = getattr(model_builder, 's3_upload_path', None) or \
                     getattr(model_builder, 's3_model_data_url', None)
    secret_key = getattr(model_builder, 'secret_key', '')
    return model_data_url, secret_key


def register(
    model_package_group_name,
    model_approval_status,
    evaluation_result,
    mlflow_run_id=None,
    training_job_name=None,
    pipeline_run_id=None,
):
    """Register a model via MLflow, build deployable artifacts with ModelBuilder,
    and update the auto-created SageMaker Model Package."""
    try:
        suffix = strftime('%d-%H-%M-%S', gmtime())
        sm_client = boto3.client("sagemaker")
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
        container_image = os.environ.get("CONTAINER_IMAGE", "")

        pipeline_run = mlflow.start_run(run_id=pipeline_run_id) if pipeline_run_id else None
        run = mlflow.start_run(run_name=f"register-{suffix}", nested=True)
        log_runtime_info()

        # Log evaluation result
        with open("evaluation.json", "w") as f:
            f.write(json.dumps(evaluation_result))
        mlflow.log_artifact(local_path="evaluation.json")

        # Resolve mlflow_run_id from training job's model.tar.gz if needed.
        # This is used in the hybrid pipeline (Part 3 of notebook 03) where TrainingStep
        # runs training/train.py as a script. The script writes mlflow_run_id.txt into
        # model.tar.gz, but the @step register function can't access train_fn's return
        # value — so we extract it from the artifact. In the @step-only pipeline (Part 2)
        # and the CI/CD pipeline (notebook 04), mlflow_run_id is passed directly.
        if not mlflow_run_id and training_job_name and training_job_name != "local":
            try:
                import s3fs, tarfile, tempfile
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

        # Step 1: Register in MLflow Model Registry
        model_uri = f"runs:/{mlflow_run_id}/model"
        model_version = mlflow.register_model(model_uri=model_uri, name=model_package_group_name)
        print(f"## Registered in MLflow: {model_version.name} v{model_version.version}")

        # Step 2: Build deployable model artifacts with ModelBuilder
        model_data_url = None
        secret_key = ''
        if container_image and container_image != "local":
            try:
                model_data_url, secret_key = _build_model_artifacts(container_image, model_uri, tracking_uri)
                if model_data_url:
                    print(f"## ModelBuilder created artifacts at: {model_data_url}")
                else:
                    print("## ModelBuilder built but model_data_url not found")
            except Exception as e:
                print(f"## ModelBuilder failed: {e}")
                secret_key = ''

        # Step 3: Wait for auto-sync to create SageMaker Model Package.
        # The auto-created package is matched to the MLflow model version via the
        # package's MetadataProperties.GeneratedBy, which records the MLflow
        # registered-model name and version. We can't match on SourceUri because
        # SageMaker stores it as a resolved s3:// artifact path while
        # model_version.source is a "models:/..." URI (they never compare equal).
        print("## Waiting for SageMaker auto-registration...")
        sm_model_package_arn = None
        mlflow_model_name = model_version.name
        mlflow_model_version = str(model_version.version)

        def _matches_mlflow_version(pkg_desc):
            generated_by = pkg_desc.get("MetadataProperties", {}).get("GeneratedBy", "")
            try:
                meta = json.loads(generated_by)
            except (json.JSONDecodeError, TypeError):
                return False
            return (
                meta.get("MLflowRegisteredModel") == mlflow_model_name
                and str(meta.get("ModelVersion")) == mlflow_model_version
            )

        for _ in range(10):
            groups = sm_client.list_model_package_groups(
                NameContains=model_package_group_name
            )["ModelPackageGroupSummaryList"]
            for g in groups:
                pkgs = sm_client.list_model_packages(
                    ModelPackageGroupName=g["ModelPackageGroupName"],
                    SortBy="CreationTime", SortOrder="Descending", MaxResults=5,
                )
                for p in pkgs["ModelPackageSummaryList"]:
                    pkg_desc = sm_client.describe_model_package(ModelPackageName=p["ModelPackageArn"])
                    if _matches_mlflow_version(pkg_desc):
                        sm_model_package_arn = p["ModelPackageArn"]
                        sm_model_package_group = g["ModelPackageGroupName"]
                        break
                if sm_model_package_arn:
                    break
            if sm_model_package_arn:
                break
            time.sleep(3)
            print(".", end="", flush=True)

        if not sm_model_package_arn:
            raise TimeoutError("SageMaker Model Package was not auto-created within timeout")
        print(f"\n## Found SageMaker Model Package: {sm_model_package_arn}")

        # Step 4: Update the model package to make it deployable
        update_kwargs = {
            "ModelPackageArn": sm_model_package_arn,
            "ModelApprovalStatus": model_approval_status,
            "CustomerMetadataProperties": {
                "TrainingJobName": training_job_name or "local",
                "ContainerImage": container_image or "n/a",
                "MlflowRunId": mlflow_run_id or "",
                "MlflowModelUri": model_uri,
            },
        }

        if container_image and container_image != "local":
            inference_spec = {
                "Containers": [{
                    "Image": container_image,
                    "Environment": {
                        "SAGEMAKER_PROGRAM": "inference.py",
                        "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
                        "SAGEMAKER_SERVE_SECRET_KEY": secret_key if secret_key else "",
                    },
                }],
                "SupportedContentTypes": ["text/csv"],
                "SupportedResponseMIMETypes": ["text/csv"],
                "SupportedRealtimeInferenceInstanceTypes": ["ml.m5.large", "ml.m5.xlarge"],
                "SupportedTransformInstanceTypes": ["ml.m5.large", "ml.m5.xlarge"],
            }
            if model_data_url:
                inference_spec["Containers"][0]["ModelDataUrl"] = model_data_url
            update_kwargs["InferenceSpecification"] = inference_spec

        sm_client.update_model_package(**update_kwargs)
        print(f"## Updated SageMaker Model Package:")
        print(f"   Container: {container_image}")
        print(f"   ModelDataUrl: {model_data_url or 'n/a'}")
        print(f"   TrainingJob: {training_job_name or 'local'}")

        mlflow.log_params({
            "registered_model_name": model_version.name,
            "registered_model_version": model_version.version,
            "sm_model_package_arn": sm_model_package_arn,
            "sm_model_package_group": sm_model_package_group,
            "model_data_url": model_data_url or "n/a",
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
