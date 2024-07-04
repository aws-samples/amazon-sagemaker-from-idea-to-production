import json
import sagemaker
import boto3
import mlflow
from time import gmtime, strftime
from sagemaker.estimator import Estimator
from sagemaker.model_metrics import (
    MetricsSource, 
    ModelMetrics, 
    FileSource
)

def register(
    training_job_name,
    model_package_group_name,
    model_approval_status,
    evaluation_result,
    output_s3_prefix,
    tracking_server_arn,
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
        experiment = mlflow.set_experiment(experiment_name=experiment_name if experiment_name else f"{register.__name__ }-{suffix}")
        pipeline_run = mlflow.start_run(run_id=pipeline_run_id) if pipeline_run_id else None            
        run = mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run(run_name=f"register-{suffix}", nested=True)

        evaluation_result_path = f"evaluation.json"
        with open(evaluation_result_path, "w") as f:
            f.write(json.dumps(evaluation_result))
            
        mlflow.log_artifact(local_path=evaluation_result_path)
            
        estimator = Estimator.attach(training_job_name)
        
        model_metrics = ModelMetrics(
            model_statistics=MetricsSource(
                s3_uri=model_statistics_s3_path,
                content_type="application/json",
            ) if model_statistics_s3_path else None,
            model_constraints=MetricsSource(
                s3_uri=model_constraints_s3_path,
                content_type="application/json",
            ) if model_constraints_s3_path else None,
            model_data_statistics=MetricsSource(
                s3_uri=model_data_statistics_s3_path,
                content_type="application/json",
            ) if model_data_statistics_s3_path else None,
            model_data_constraints=MetricsSource(
                s3_uri=model_data_constraints_s3_path,
                content_type="application/json",
            ) if model_data_constraints_s3_path else None,
        )
    
        model_package = estimator.register(
            content_types=["text/csv"],
            response_types=["text/csv"],
            inference_instances=["ml.m5.xlarge", "ml.m5.large"],
            transform_instances=["ml.m5.xlarge", "ml.m5.large"],
            model_package_group_name=model_package_group_name,
            approval_status=model_approval_status,
            model_metrics=model_metrics,
            model_name="from-idea-to-prod-pipeline-model",
            domain="MACHINE_LEARNING",
            task="CLASSIFICATION", 
        )

        mlflow.log_params({
            "model_package_arn":model_package.model_package_arn,
            "model_statistics_uri":model_statistics_s3_path if model_statistics_s3_path else '',
            "model_constraints_uri":model_constraints_s3_path if model_constraints_s3_path else '',
            "data_statistics_uri":model_data_statistics_s3_path if model_data_statistics_s3_path else '',
            "data_constraints_uri":model_data_constraints_s3_path if model_data_constraints_s3_path else '',
        })

        return {
            "model_package_arn":model_package.model_package_arn,
            "model_package_group_name":model_package_group_name,
            "pipeline_run_id":pipeline_run.info.run_id if pipeline_run else ''
        }

    except Exception as e:
        print(f"Exception in processing script: {e}")
        raise e
    finally:
        mlflow.end_run()

