"""Train step — XGBoost training with MLflow tracking.

The same ``train()`` function runs in four ways:

- called directly in a notebook (local validation)
- wrapped with ``@remote`` (notebook 02) or ``@step`` (notebook 03)
- as the entry script of a ``ModelTrainer`` job in script mode (notebook 02):
  ``main()`` reads the hyperparameters and input channels that SageMaker passes
  to the script, and calls ``train()``
"""
import argparse
import glob
import os
import shutil
from time import gmtime, strftime

import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

try:
    from pipeline_steps.runtime_utils import log_runtime_info
except ImportError:
    try:
        # ModelTrainer runs this file as a script from inside the pipeline_steps folder
        from runtime_utils import log_runtime_info
    except ImportError:
        def log_runtime_info(): pass

# MLflow run URLs printed to stdout point to the tracking API, not the MLflow UI, so suppress them
os.environ["MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"] = "true"


def train(
    train_data_s3_path,
    validation_data_s3_path,
    output_s3_prefix,
    num_round=50,
    max_depth=3,
    eta=0.5,
    alpha=2.5,
    gamma=0,
    min_child_weight=1,
    subsample=1.0,
    colsample_bytree=1.0,
    objective="binary:logistic",
    eval_metric="auc",
    early_stopping_rounds=5,
    verbosity=1,
    pipeline_run_id=None,
    model_dir=None,
):
    """Train an XGBoost model, log it to MLflow, and save it to S3.

    The data paths can be S3 URIs or local files. If ``model_dir`` is set, the
    model is also saved there, for example to SageMaker's ``/opt/ml/model``.
    MLflow tracking URI and experiment are read from the environment.
    """
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
            "gamma": gamma,
            "min_child_weight": min_child_weight,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "objective": objective,
            "eval_metric": eval_metric,
            "verbosity": verbosity,
        }

        mlflow.xgboost.autolog(log_model_signatures=True, log_datasets=True, log_models=False)

        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_round,
            evals=[(dtrain, "train"), (dval, "validation")],
            early_stopping_rounds=early_stopping_rounds,
        )

        # Compute and log AUC
        val_auc = roc_auc_score(dval.get_label(), booster.predict(dval))
        train_auc = roc_auc_score(dtrain.get_label(), booster.predict(dtrain))
        mlflow.log_params({**params, "num_round": num_round, "early_stopping_rounds": early_stopping_rounds})
        mlflow.log_metrics({"validation_auc": val_auc, "train_auc": train_auc})

        # autolog(log_models=False) leaves model logging to us. log_model returns
        # a ModelInfo whose model_id identifies the logged model on the tracking
        # server; the register step resolves it from the run's model outputs.
        model_info = mlflow.xgboost.log_model(booster, name=f"xgboost-{suffix}")
        mlflow.log_param("mlflow_model_uri", model_info.model_uri)

        print(f"## Training complete — train AUC: {train_auc:.4f}, validation AUC: {val_auc:.4f}")

        # Save model to S3, or to a local folder when validating locally
        model_output_s3_path = f"{output_s3_prefix}/model/xgboost-model"
        booster.save_model("xgboost-model")
        if model_output_s3_path.startswith("s3://"):
            import boto3
            bucket, key = model_output_s3_path.replace("s3://", "").split("/", 1)
            boto3.client("s3").upload_file("xgboost-model", bucket, key)
        else:
            os.makedirs(os.path.dirname(model_output_s3_path), exist_ok=True)
            shutil.copy("xgboost-model", model_output_s3_path)

        mlflow.log_param("model_s3_path", model_output_s3_path)

        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
            shutil.copy("xgboost-model", os.path.join(model_dir, "xgboost-model"))

        return {
            "model_s3_path": model_output_s3_path,
            "train_auc": train_auc,
            "validation_auc": val_auc,
            "mlflow_run_id": run.info.run_id,
            "mlflow_model_id": model_info.model_id,
            "training_job_name": os.environ.get("TRAINING_JOB_NAME", "local"),
            "pipeline_run_id": pipeline_run.info.run_id if pipeline_run else '',
        }

    except Exception as e:
        print(f"Exception in training: {e}")
        raise e
    finally:
        while mlflow.active_run():
            mlflow.end_run()


def _channel_file(channel_dir):
    """Return the CSV file that SageMaker downloaded into an input channel directory."""
    files = sorted(glob.glob(os.path.join(channel_dir, "*.csv")))
    if len(files) != 1:
        raise ValueError(f"Expected one CSV file in {channel_dir}, found {files}")
    return files[0]


def main():
    """Script-mode entry point for a SageMaker training job.

    SageMaker passes the hyperparameters as command-line arguments, downloads the
    input channels to local directories, and sets SM_CHANNEL_<NAME> and
    SM_MODEL_DIR to their locations.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_round", type=int, default=50)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=2.5)
    parser.add_argument("--gamma", type=float, default=0)
    parser.add_argument("--min_child_weight", type=float, default=1)
    parser.add_argument("--subsample", type=float, default=1.0)
    parser.add_argument("--colsample_bytree", type=float, default=1.0)
    parser.add_argument("--objective", type=str, default="binary:logistic")
    parser.add_argument("--eval_metric", type=str, default="auc")
    parser.add_argument("--early_stopping_rounds", type=int, default=5)
    parser.add_argument("--verbosity", type=int, default=1)
    parser.add_argument("--output_s3_prefix", type=str, required=True)
    parser.add_argument("--pipeline_run_id", type=str, default=None)
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    args, _ = parser.parse_known_args()

    hyperparams = {
        k: v for k, v in vars(args).items()
        if k not in ("train", "validation", "model_dir", "output_s3_prefix", "pipeline_run_id")
    }

    train(
        train_data_s3_path=_channel_file(args.train),
        validation_data_s3_path=_channel_file(args.validation),
        output_s3_prefix=args.output_s3_prefix,
        pipeline_run_id=args.pipeline_run_id or None,
        model_dir=args.model_dir,
        **hyperparams,
    )


if __name__ == "__main__":
    main()
