"""SageMaker inference handler for the MLflow-logged XGBoost model.

This script is logged into the MLflow model's artifact store (under ``code/``)
by ``register.py`` and served by the custom BYOC XGBoost container, which is
script-mode compatible (``sagemaker-inference``): it runs the program named by
``SAGEMAKER_PROGRAM`` and dispatches requests to the functions below.

Because the model is registered with an ``S3Prefix`` ``ModelDataSource`` pointing
at the MLflow artifact location, the **entire** MLflow model directory
(``MLmodel``, ``model.ubj``, ``conda.yaml`` ...) plus this ``code/inference.py``
are downloaded to ``/opt/ml/model`` on the endpoint. That lets
``mlflow.xgboost.load_model`` load the model directly and resolve the correct
serialization format (the training step logs it with the default ``ubj`` format).
"""
import json

import numpy as np


def model_fn(model_dir):
    """Load the MLflow-logged XGBoost booster from the downloaded model dir."""
    import mlflow.xgboost

    return mlflow.xgboost.load_model(model_dir)


def input_fn(request_body, request_content_type):
    """Parse a CSV or JSON request body into a 2D numpy feature array.

    - ``text/csv``: one record per line, comma-separated features (no header,
      no label column) — matching the preprocessing output used at training.
    - ``application/json``: a list of records, or a dict with an ``instances``
      / ``inputs`` key holding a list of records.
    """
    if isinstance(request_body, (bytes, bytearray)):
        request_body = request_body.decode("utf-8")

    content_type = (request_content_type or "").split(";")[0].strip()

    if content_type == "text/csv":
        rows = [
            [float(value) for value in line.split(",")]
            for line in request_body.strip().splitlines()
            if line.strip()
        ]
        return np.array(rows, dtype=np.float32)

    if content_type == "application/json":
        payload = json.loads(request_body)
        if isinstance(payload, dict):
            payload = payload.get("instances", payload.get("inputs", payload))
        array = np.array(payload, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return array

    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """Run inference. Returns raw probabilities from the binary:logistic model."""
    import xgboost as xgb

    return model.predict(xgb.DMatrix(input_data))


def output_fn(prediction, accept):
    """Serialize predictions as CSV (default) or JSON."""
    predictions = np.asarray(prediction).ravel()

    accept_type = (accept or "").split(";")[0].strip()
    if accept_type == "application/json":
        return json.dumps({"predictions": predictions.tolist()}), "application/json"

    body = "\n".join(str(float(value)) for value in predictions)
    return body, "text/csv"
