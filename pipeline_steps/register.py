import json
import os
import tempfile
import tarfile
import boto3
import mlflow
import s3fs
from time import gmtime, strftime

from sagemaker.core.helper.session_helper import Session
from sagemaker.core.model_metrics import MetricsSource, ModelMetrics
from sagemaker.core.common_utils import name_from_base

from sagemaker.serve import ModelBuilder
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.serve.spec.inference_spec import InferenceSpec
from sagemaker.serve.builder.schema_builder import SchemaBuilder
from sagemaker.serve.utils.types import ModelServer


class XGBoostMLflowSpec(InferenceSpec):
    """Inference spec that loads an XGBoost model from MLflow."""
    def __init__(self, mlflow_model_uri, tracking_uri):
        self._mlflow_model_uri = mlflow_model_uri
        self._tracking_uri = tracking_uri

    def load(self, model_dir=None):
        import mlflow
        mlflow.set_tracking_uri(self._tracking_uri)
        return mlflow.xgboost.load_model(self._mlflow_model_uri)

    def invoke(self, input_object, model):
        import xgboost as xgb
        import numpy as np
        if isinstance(input_object, str):
            rows = [list(map(float, r.split(",")) ) for r in input_object.strip().split("\n") if r]
            dmatrix = xgb.DMatrix(np.array(rows))
        elif isinstance(input_object, list):
            dmatrix = xgb.DMatrix(np.array(input_object))
        else:
            dmatrix = xgb.DMatrix(np.array(input_object))
        return "\n".join(map(str, model.predict(dmatrix)))


def register(
    model_package_group_name,
    model_approval_status,
    evaluation_result,
    output_s3_prefix,
    tracking_server_arn,
    image_uri,
    # For @step path: pass model S3 path directly
    model_s3_path=None,
    # For hybrid path: pass training job name (model artifacts extracted automatically)
    training_job_name=None,
    model_statistics_s3_path=None,
    model_constraints_s3_path=None,
    model_data_statistics_s3_path=None,
    model_data_constraints_s3_path=None,
    experiment_name=None,
    pipeline_run_id=None,
    run_id=None,
):
    try:
        suffix = strftime('%d-%H-%M-%S', gmtime())
        mlflow.set_tracking_uri(tracking_server_arn)
        experiment = mlflow.set_experiment(
            experiment_name=experiment_name if experiment_name else f"register-{suffix}"
        )
        pipeline_run = mlflow.start_run(run_id=pipeline_run_id) if pipeline_run_id else None
        run = mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run(run_name=f"register-{suffix}", nested=True)

        # Log evaluation result
        with open("evaluation.json", "w") as f:
            f.write(json.dumps(evaluation_result))
        mlflow.log_artifact(local_path="evaluation.json")

        sm_client = boto3.client("sagemaker")

        # Resolve model data URL
        if training_job_name:
            desc = sm_client.describe_training_job(TrainingJobName=training_job_name)
            model_data_url = desc["ModelArtifacts"]["S3ModelArtifacts"]
        elif model_s3_path:
            # Package raw model file into model.tar.gz for SageMaker
            s3 = s3fs.S3FileSystem()
            local_model = "xgboost-model"
            with s3.open(model_s3_path, "rb") as remote, open(local_model, "wb") as local:
                local.write(remote.read())

            tar_path = tempfile.mktemp(suffix=".tar.gz")
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(local_model, arcname="xgboost-model")

            model_data_url = f"{output_s3_prefix}/model/model.tar.gz"
            with s3.open(model_data_url, "wb") as remote, open(tar_path, "rb") as local:
                remote.write(local.read())
        else:
            raise ValueError("Provide either training_job_name or model_s3_path")

        # Find the MLflow model URI from the latest run's logged model
        mlflow_client = mlflow.tracking.MlflowClient()
        current_run_id = mlflow.active_run().info.run_id
        # Search for the training run that logged the model
        runs = mlflow_client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="",
            order_by=["start_time DESC"],
            max_results=10,
        )
        mlflow_model_uri = None
        for r in runs:
            artifacts = mlflow_client.list_artifacts(r.info.run_id)
            for a in artifacts:
                if a.path == "model" and a.is_dir:
                    mlflow_model_uri = f"runs:/{r.info.run_id}/model"
                    break
            if mlflow_model_uri:
                break

        # Build model metrics
        model_metrics = ModelMetrics(
            model_statistics=MetricsSource(s3_uri=model_statistics_s3_path, content_type="application/json") if model_statistics_s3_path else None,
            model_constraints=MetricsSource(s3_uri=model_constraints_s3_path, content_type="application/json") if model_constraints_s3_path else None,
            model_data_statistics=MetricsSource(s3_uri=model_data_statistics_s3_path, content_type="application/json") if model_data_statistics_s3_path else None,
            model_data_constraints=MetricsSource(s3_uri=model_data_constraints_s3_path, content_type="application/json") if model_data_constraints_s3_path else None,
        )

        # Use ModelBuilder to build and register the model
        # Same InferenceSpec pattern as notebook 02
        schema_builder = SchemaBuilder(
            sample_input="0.0," * 58 + "0.0",  # 59 features
            sample_output="0.5",
        )

        model_builder = ModelBuilder(
            image_uri=image_uri,
            s3_model_data_url=model_data_url,
            instance_type="ml.m5.large",
            inference_spec=XGBoostMLflowSpec(
                mlflow_model_uri or "placeholder",
                tracking_server_arn,
            ),
            schema_builder=schema_builder,
            model_server=ModelServer.TORCHSERVE,
            dependencies={"auto": False},
        )
        model_builder.build()

        # ModelBuilder.register() has a bug: passes Framework enum to boto3 instead of string.
        # Workaround: register directly via core API using the container def from build().
        from sagemaker.core.model_registry import create_model_package_from_containers
        sagemaker_session = Session()

        model_package_response = create_model_package_from_containers(
            sagemaker_session=sagemaker_session,
            containers=[{"Image": image_uri, "ModelDataUrl": model_data_url}],
            content_types=["text/csv"],
            response_types=["text/csv"],
            inference_instances=["ml.m5.xlarge", "ml.m5.large"],
            transform_instances=["ml.m5.xlarge", "ml.m5.large"],
            model_package_group_name=model_package_group_name,
            approval_status=model_approval_status,
            model_metrics=model_metrics._to_request_dict(),
        )
        model_package_arn = model_package_response.get("ModelPackageArn")

        mlflow.log_params({
            "model_package_arn": model_package_arn,
            "model_data_url": model_data_url,
            "mlflow_model_uri": mlflow_model_uri or "not found",
        })

        print(f"## Registered model package: {model_package_arn}")

        return {
            "model_package_arn": model_package_arn,
            "model_package_group_name": model_package_group_name,
            "pipeline_run_id": pipeline_run.info.run_id if pipeline_run else '',
        }

    except Exception as e:
        print(f"Exception in register: {e}")
        raise e
    finally:
        mlflow.end_run()
