import os
from time import gmtime, strftime

import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

try:
    from pipeline_steps.runtime_utils import log_runtime_info
except ImportError:
    def log_runtime_info():
        pass


def train(
    train_data_s3_path,
    validation_data_s3_path,
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
):
    """Train an XGBoost model and log it to MLflow.

    The trained model is logged to the MLflow artifact store only; it is not
    copied to S3. Downstream steps receive ``mlflow_model_id`` and load the
    model with ``mlflow.xgboost.load_model(f"models:/{mlflow_model_id}")``.

    MLflow tracking URI and experiment are read from the environment.
    """
    try:
        suffix = strftime("%d-%H-%M-%S", gmtime())

        pipeline_run = mlflow.start_run(run_id=pipeline_run_id) if pipeline_run_id else None
        run = mlflow.start_run(run_name=f"training-{suffix}", nested=True)
        log_runtime_info()

        # The first column is the label in the header-less CSV files.
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

        val_auc = roc_auc_score(dval.get_label(), booster.predict(dval))
        train_auc = roc_auc_score(dtrain.get_label(), booster.predict(dtrain))
        mlflow.log_params(
            {
                **params,
                "num_round": num_round,
                "early_stopping_rounds": early_stopping_rounds,
            }
        )
        mlflow.log_metrics({"validation_auc": val_auc, "train_auc": train_auc})

        # The model ID is the handle used by evaluate, explain, and register.
        model_info = mlflow.xgboost.log_model(booster, name=f"xgboost-{suffix}")
        mlflow.log_param("mlflow_model_uri", model_info.model_uri)
        print(
            f"## Training complete — train AUC: {train_auc:.4f}, "
            f"validation AUC: {val_auc:.4f}"
        )

        return {
            "mlflow_model_id": model_info.model_id,
            "train_auc": train_auc,
            "validation_auc": val_auc,
            "mlflow_run_id": run.info.run_id,
            "training_job_name": os.environ.get("TRAINING_JOB_NAME", "local"),
            "pipeline_run_id": pipeline_run.info.run_id if pipeline_run else "",
        }

    except Exception as e:
        print(f"Exception in training: {e}")
        raise
    finally:
        while mlflow.active_run():
            mlflow.end_run()
