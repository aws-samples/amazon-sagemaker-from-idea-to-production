import argparse
import csv
import json
import logging
import math
import os
import time

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
sm_client = boto3.client("sagemaker")
runtime_client = boto3.client("sagemaker-runtime")


def load_payload(payload_file, expected_feature_count):
    """Load and validate one headerless numeric CSV feature row."""
    with open(payload_file, encoding="utf-8") as handle:
        payload = next((line.strip() for line in handle if line.strip()), None)
    if not payload:
        raise ValueError(f"No non-empty payload row found in {payload_file}")

    fields = next(csv.reader([payload]))
    try:
        values = [float(value) for value in fields]
    except ValueError as error:
        raise ValueError(f"Payload contains a non-numeric feature: {error}") from error

    if len(values) != expected_feature_count:
        raise ValueError(
            f"Payload has {len(values)} features; expected {expected_feature_count}"
        )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Payload contains NaN or an infinite feature value")
    return payload, len(values)


def invoke_endpoint(endpoint_name, payload):
    """Invoke one CSV row and require exactly one probability response."""
    inference_id = f"pipeline-staging-{time.time_ns()}"
    started = time.perf_counter()
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="text/csv",
        Accept="text/csv",
        Body=payload,
        InferenceId=inference_id,
    )
    latency_ms = (time.perf_counter() - started) * 1000

    http_status = response["ResponseMetadata"]["HTTPStatusCode"]
    content_type = response.get("ContentType", "").split(";", 1)[0].strip().lower()
    body = response["Body"].read().decode("utf-8").strip()
    prediction_lines = [line.strip() for line in body.splitlines() if line.strip()]

    if http_status != 200:
        raise AssertionError(f"Expected HTTP 200, got {http_status}")
    if content_type != "text/csv":
        raise AssertionError(f"Expected text/csv response, got {content_type!r}")
    if len(prediction_lines) != 1:
        raise AssertionError(
            f"Expected one prediction for one input row, got {len(prediction_lines)}: {body!r}"
        )
    try:
        prediction = float(prediction_lines[0])
    except ValueError as error:
        raise AssertionError(f"Prediction is not numeric: {prediction_lines[0]!r}") from error
    if not math.isfinite(prediction) or not 0.0 <= prediction <= 1.0:
        raise AssertionError(f"Expected a finite probability in [0, 1], got {prediction}")

    return {
        "endpoint_name": endpoint_name,
        "success": True,
        "http_status": http_status,
        "content_type": content_type,
        "prediction": prediction,
        "latency_ms": round(latency_ms, 1),
        "inference_id": inference_id,
    }


def test_endpoint(endpoint_name, payload):
    """Require an InService, data-capturing endpoint and run real inference."""
    try:
        endpoint = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = endpoint["EndpointStatus"]
        if status != "InService":
            raise RuntimeError(
                f"SageMaker endpoint {endpoint_name} is {status}, not InService"
            )

        endpoint_config_name = endpoint["EndpointConfigName"]
        endpoint_config = sm_client.describe_endpoint_config(
            EndpointConfigName=endpoint_config_name
        )
        capture = endpoint_config.get("DataCaptureConfig", {})
        if not capture.get("EnableCapture"):
            raise AssertionError(
                f"Data capture is not enabled for endpoint config {endpoint_config_name}"
            )
        logger.info("Data capture enabled for %s", endpoint_config_name)
        return invoke_endpoint(endpoint_name, payload)
    except ClientError as error:
        message = error.response["Error"]["Message"]
        logger.error(message)
        raise RuntimeError(message) from error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-level", type=str, default=os.environ.get("LOGLEVEL", "INFO").upper()
    )
    parser.add_argument("--import-build-config", type=str, required=True)
    parser.add_argument("--export-test-results", type=str, required=True)
    parser.add_argument("--payload-file", type=str, required=True)
    parser.add_argument("--expected-feature-count", type=int, required=True)
    args, _ = parser.parse_known_args()

    logging.basicConfig(
        format="%(levelname)s: [%(filename)s:%(lineno)s] %(message)s",
        level=args.log_level,
    )

    with open(args.import_build_config, encoding="utf-8") as handle:
        config = json.load(handle)

    endpoint_name = "{}-{}".format(
        config["Parameters"]["SageMakerProjectName"],
        config["Parameters"]["StageName"],
    )
    payload, feature_count = load_payload(
        args.payload_file, args.expected_feature_count
    )
    results = test_endpoint(endpoint_name, payload)
    results["feature_count"] = feature_count
    results["payload_file"] = args.payload_file

    logger.info("Smoke test passed: %s", json.dumps(results, sort_keys=True))
    with open(args.export_test_results, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=4)


if __name__ == "__main__":
    main()
