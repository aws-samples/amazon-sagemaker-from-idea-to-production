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

# helper function to load XGBoost model into xgboost.Booster
def load_model(model_data_s3_uri):
    model_file = "./xgboost-model.tar.gz"
    bucket, key = model_data_s3_uri.replace("s3://", "").split("/", 1)
    boto3.client("s3").download_file(bucket, key, model_file)
    
    with tarfile.open(model_file, "r:gz") as t:
        t.extractall(path=".")
    
    # Load model
    model = xgb.Booster()
    model.load_model("xgboost-model")

    return model

def plot_roc_curve(fpr, tpr):
    fn = "roc-curve.png"
    fig = plt.figure(figsize=(6, 4))
    
    # Plot the diagonal 50% line
    plt.plot([0, 1], [0, 1], 'k--')
    
    # Plot the FPR and TPR achieved by our model
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
    tracking_server_arn,
    experiment_name=None,
    pipeline_run_id=None,
    run_id=None,
):
    try:
        suffix = strftime('%d-%H-%M-%S', gmtime())
        mlflow.set_tracking_uri(tracking_server_arn)
        experiment = mlflow.set_experiment(experiment_name=experiment_name if experiment_name else f"{evaluate.__name__ }-{suffix}")
        pipeline_run = mlflow.start_run(run_id=pipeline_run_id) if pipeline_run_id else None            
        run = mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run(run_name=f"evaluate-{suffix}", nested=True)
        
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
    
        # Save prediction baseline file - we need it later for the model quality monitoring
        pd.DataFrame({"prediction":np.array(np.round(probability), dtype=int),
                      "probability":probability,
                      "label":y_test.squeeze()}
                    ).to_csv(prediction_baseline_s3_path, index=False, header=True)
        
        return {
            **eval_result,
            "prediction_baseline_data":prediction_baseline_s3_path,
            "experiment_name":experiment.name,
            "pipeline_run_id":pipeline_run.info.run_id if pipeline_run else ''
        }
            
    except Exception as e:
        print(f"Exception in processing script: {e}")
        raise e
    finally:
        mlflow.end_run()