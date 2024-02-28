import json
import sagemaker
import boto3
from sagemaker.s3_utils import parse_s3_url
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
):
    evaluation_result_path = f"evaluation.json"
    with open(evaluation_result_path, "w") as f:
        f.write(json.dumps(evaluation_result))
        
    evaluation_file_s3_path = f"{output_s3_prefix}/{training_job_name}/output/{evaluation_result_path}"
    boto3.client("s3").upload_file(evaluation_result_path, *parse_s3_url(evaluation_file_s3_path))
        
    estimator = Estimator.attach(training_job_name)

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=evaluation_file_s3_path,
            content_type="application/json",
        )
    )

    model_package = estimator.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.xlarge", "ml.m5.large"],
        transform_instances=["ml.m5.xlarge", "ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
        model_name="from-idea-to-prod-xgboost-model"
    )
    
    return model_package.model_package_arn