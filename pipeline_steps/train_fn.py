import os
import pandas as pd
import xgboost as xgb
import mlflow
from sklearn.metrics import roc_auc_score
from time import gmtime, strftime
try:
    from pipeline_steps.runtime_utils import log_runtime_info
except ImportError:
    def log_runtime_info(): pass


def train(
    train_data_s3_path,
    validation_data_s3_path,
    output_s3_prefix,
    num_round=50,
    max_depth=3,
    eta=0.5,
    alpha=2.5,
    objective="binary:logistic",
    pipeline_run_id=None,
):
    """Train XGBoost model. MLflow tracking URI and experiment are read from environment."""
    try:
        suffix = strftime('%d-%H-%M-%S', gmtime())

        pipeline_run = mlflow.start_run(run_id=pipeline_run_id) if pipeline_run_id else None
        run = mlflow.start_run(run_name=f"training-{suffix}", nested=True)
        log_runtime_info()

        # Load data — first column is the label
        train_df = pd.read_csv(train_data_s3_path, header=None)
        val_df = pd.read_csv(validation_data_s3_path, header=None)

        dtrain = xgb.DMatrix(train_df.iloc[:, 1:].values, label=train_df.iloc[:, 0].values)
        dval = xgb.DMatrix(val_df.iloc[:, 1:].values, label=val_df.iloc[:, 0].values)

        params = {
            "max_depth": max_depth,
            "eta": eta,
            "alpha": alpha,
            "objective": objective,
            "eval_metric": "auc",
        }

        mlflow.xgboost.autolog(log_model_signatures=True, log_datasets=True)

        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_round,
            evals=[(dtrain, "train"), (dval, "validation")],
            early_stopping_rounds=5,
        )

        # Compute and log AUC
        val_auc = roc_auc_score(dval.get_label(), booster.predict(dval))
        train_auc = roc_auc_score(dtrain.get_label(), booster.predict(dtrain))
        mlflow.log_params(params)
        mlflow.log_metrics({"validation_auc": val_auc, "train_auc": train_auc})

        print(f"## Training complete — train AUC: {train_auc:.4f}, validation AUC: {val_auc:.4f}")

        # Save model to S3
        model_output_s3_path = f"{output_s3_prefix}/model/xgboost-model"
        booster.save_model("xgboost-model")
        import boto3
        bucket, key = model_output_s3_path.replace("s3://", "").split("/", 1)
        boto3.client("s3").upload_file("xgboost-model", bucket, key)

        mlflow.log_param("model_s3_path", model_output_s3_path)

        return {
            "model_s3_path": model_output_s3_path,
            "train_auc": train_auc,
            "validation_auc": val_auc,
            "mlflow_run_id": run.info.run_id,
            "training_job_name": os.environ.get("TRAINING_JOB_NAME", "local"),
            "pipeline_run_id": pipeline_run.info.run_id if pipeline_run else '',
        }

    except Exception as e:
        print(f"Exception in training: {e}")
        raise e
    finally:
        while mlflow.active_run():
            mlflow.end_run()
