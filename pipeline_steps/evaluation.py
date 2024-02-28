import pickle as pkl
import tarfile
import joblib
import boto3
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_curve, auc
from sagemaker.s3_utils import parse_s3_url

def download_file(s3_client, s3_path, local_path):
    bucket, object_key = parse_s3_url(s3_path)
    s3_client.download_file(bucket, object_key, local_path)

def evaluate(
    test_x_data_s3_path,
    test_y_data_s3_path,
    model_s3_path,
    output_s3_prefix,
):
    s3 = boto3.client("s3")
        
    model_path = "model.tar.gz"
    test_x_path = "test_x.csv"
    test_y_path = "text_y.csv"
    prediction_baseline_path = "prediction_baseline.csv"

    download_file(s3, model_s3_path, model_path)
    download_file(s3, test_x_data_s3_path, test_x_path)
    download_file(s3, test_y_data_s3_path, test_y_path)
                
    # Read model tar file
    with tarfile.open(model_path, "r:gz") as t:
        t.extractall(path=".")
    
    # Load model
    model = xgb.Booster()
    model.load_model("xgboost-model")
    
    # Read test data
    X_test = xgb.DMatrix(pd.read_csv(test_x_path, header=None).values)
    y_test = pd.read_csv(test_y_path, header=None).to_numpy()

    # Run predictions
    probability = model.predict(X_test)

    # Evaluate predictions
    fpr, tpr, thresholds = roc_curve(y_test, probability)
    auc_score = auc(fpr, tpr)
    eval_result = {
        "classification_metrics": {
            "auc_score": {
                "value": auc_score,
            },
        },
    }
    
    # Save prediction baseline file - we need it later for the model quality monitoring
    pd.DataFrame({"prediction":np.array(np.round(probability), dtype=int),
                  "probability":probability,
                  "label":y_test.squeeze()}
                ).to_csv(prediction_baseline_path, index=False, header=True)
    
    prediction_baseline_s3_path = f"{output_s3_prefix}/prediction_baseline/{prediction_baseline_path}"
    
    s3.upload_file(prediction_baseline_path, *parse_s3_url(prediction_baseline_s3_path))
    
    return eval_result