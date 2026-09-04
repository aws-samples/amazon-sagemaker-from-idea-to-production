import numpy as np
import pandas as pd
import xgboost as xgb
import mlflow
from time import gmtime, strftime
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
try:
    from pipeline_steps.runtime_utils import log_runtime_info
except ImportError:
    def log_runtime_info(): pass


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
    mlflow_model_id,
    output_s3_prefix,
    pipeline_run_id=None,
):
    """Evaluate the MLflow-logged model produced by the training step.

    ``mlflow_model_id`` is the logged model id returned by ``train``. The model
    is loaded straight from the MLflow artifact store, so there is no second
    copy of it in S3 to keep in sync.

    MLflow tracking URI and experiment are read from the environment.
    """
    try:
        suffix = strftime('%d-%H-%M-%S', gmtime())

        pipeline_run = mlflow.start_run(run_id=pipeline_run_id) if pipeline_run_id else None
        run = mlflow.start_run(run_name=f"evaluate-{suffix}", nested=True)
        log_runtime_info()
        
        # Read test data
        X_test = xgb.DMatrix(pd.read_csv(test_x_data_s3_path, header=None).values)
        y_test = pd.read_csv(test_y_data_s3_path, header=None).to_numpy()
    
        # Load the model from the MLflow artifact store and run predictions
        booster = mlflow.xgboost.load_model(f"models:/{mlflow_model_id}")
        probability = booster.predict(X_test)
    
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
        
        mlflow.log_param("evaluated_model_id", mlflow_model_id)
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
