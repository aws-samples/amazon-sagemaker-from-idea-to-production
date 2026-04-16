import os
import numpy as np
import pandas as pd
import xgboost as xgb
import mlflow
from time import gmtime, strftime
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import tarfile
import pickle as pkl
import boto3
try:
    from pipeline_steps.runtime_utils import log_runtime_info
except ImportError:
    def log_runtime_info(): pass


def load_model(model_data_s3_uri):
    bucket, key = model_data_s3_uri.replace("s3://", "").split("/", 1)
    local_file = os.path.basename(key)
    boto3.client("s3").download_file(bucket, key, local_file)

    # Handle both tar.gz (from ModelTrainer) and raw model file (from train_fn)
    if local_file.endswith(".tar.gz"):
        with tarfile.open(local_file, "r:gz") as t:
            t.extractall(path=".")
        model_path = "xgboost-model"
    else:
        model_path = local_file

    # Try native XGBoost format first, fall back to pickle
    model = xgb.Booster()
    try:
        model.load_model(model_path)
    except xgb.core.XGBoostError:
        model = pkl.load(open(model_path, "rb"))

    return model


def plot_roc_curve(fpr, tpr):
    fn = "roc-curve.png"
    fig = plt.figure(figsize=(6, 4))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig(fn)
    return fn

  
def evaluate(
    test_x_data_s3_path,
    test_y_data_s3_path,
    model_s3_path,
    output_s3_prefix,
    pipeline_run_id=None,
):
    """Evaluate model. MLflow tracking URI and experiment are read from environment."""
    try:
        suffix = strftime('%d-%H-%M-%S', gmtime())

        pipeline_run = mlflow.start_run(run_id=pipeline_run_id) if pipeline_run_id else None
        run = mlflow.start_run(run_name=f"evaluate-{suffix}", nested=True)
        log_runtime_info()
        
        # Read test data
        X_test = xgb.DMatrix(pd.read_csv(test_x_data_s3_path, header=None).values)
        y_test = pd.read_csv(test_y_data_s3_path, header=None).to_numpy()
    
        # Run predictions
        probability = load_model(model_s3_path).predict(X_test)
    
        # Evaluate predictions
        fpr, tpr, thresholds = roc_curve(y_test, probability)
        auc_score = auc(fpr, tpr)
        eval_result = {"evaluation_result": {
            "classification_metrics": {
                "auc_score": {
                    "value": auc_score,
                },
            },
        }}
        
        mlflow.log_metric("auc_score", auc_score)
        mlflow.log_artifact(plot_roc_curve(fpr, tpr))
        
        prediction_baseline_s3_path = f"{output_s3_prefix}/prediction_baseline/prediction_baseline.csv"
    
        # Save prediction baseline file
        pd.DataFrame({"prediction": np.array(np.round(probability), dtype=int),
                      "probability": probability,
                      "label": y_test.squeeze()}
                    ).to_csv(prediction_baseline_s3_path, index=False, header=True)
        
        return {
            **eval_result,
            "prediction_baseline_data": prediction_baseline_s3_path,
            "pipeline_run_id": pipeline_run.info.run_id if pipeline_run else '',
        }
            
    except Exception as e:
        print(f"Exception in evaluation: {e}")
        raise e
    finally:
        while mlflow.active_run():
            mlflow.end_run()
