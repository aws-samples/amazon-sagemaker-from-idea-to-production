from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pandas as pd
import numpy as np
import argparse
import os

from time import gmtime, strftime, sleep
import traceback

user_profile_name = os.getenv('USER')
print(os.environ)
def _parse_args():
    
    parser = argparse.ArgumentParser()
    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--filepath', type=str, default='/opt/ml/processing/input/')
    parser.add_argument('--filename', type=str, default='bank-additional-full.csv')
    parser.add_argument('--outputpath', type=str, default='/opt/ml/processing/output/')
    
    return parser.parse_known_args()

def process_data(df_data):
    # Indicator variable to capture when pdays takes a value of 999
    df_data["no_previous_contact"] = np.where(df_data["pdays"] == 999, 1, 0)

    # Indicator for individuals not actively employed
    df_data["not_working"] = np.where(
        np.in1d(df_data["job"], ["student", "retired", "unemployed"]), 1, 0
    )

    # remove unnecessary data
    df_model_data = df_data.drop(
        ["duration", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"],
        axis=1,
    )

    bins = [18, 30, 40, 50, 60, 70, 90]
    labels = ['18-29', '30-39', '40-49', '50-59', '60-69', '70-plus']

    df_model_data['age_range'] = pd.cut(df_model_data.age, bins, labels=labels, include_lowest=True)
    df_model_data = pd.concat([df_model_data, pd.get_dummies(df_model_data['age_range'], prefix='age', dtype=int)], axis=1)
    df_model_data.drop('age', axis=1, inplace=True)
    df_model_data.drop('age_range', axis=1, inplace=True)

    scaled_features = ['pdays', 'previous', 'campaign']
    df_model_data[scaled_features] = MinMaxScaler().fit_transform(df_model_data[scaled_features])

    df_model_data = pd.get_dummies(df_model_data, dtype=int)  # Convert categorical variables to sets of indicators

    # Replace "y_no" and "y_yes" with a single label column, and bring it to the front:
    df_model_data = pd.concat(
        [
            df_model_data["y_yes"].rename(target_col),
            df_model_data.drop(["y_no", "y_yes"], axis=1),
        ],
        axis=1,
    )
    
    return df_model_data

if __name__=="__main__":
    # Process arguments
    args, _ = _parse_args()
    
    target_col = "y"
    
    import mlflow
    
    # Set the Tracking Server URI using the ARN of the Tracking Server you created
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_ARN'])
    
    # Enable autologging in MLflow
    mlflow.autolog()

    # Use the active run_id to log 
    with mlflow.start_run(run_id=os.environ['MLFLOW_RUN_ID']) as run:
        # process data
        df_model_data = process_data(pd.read_csv(os.path.join(args.filepath, args.filename), sep=";"))
    
        # Shuffle and splitting dataset
        train_data, validation_data, test_data = np.split(
            df_model_data.sample(frac=1, random_state=1729),
            [int(0.7 * len(df_model_data)), int(0.9 * len(df_model_data))],
        )
    
        print(f"Data split > train:{train_data.shape} | validation:{validation_data.shape} | test:{test_data.shape}")
        mlflow.log_params(
            {
                "train": train_data.shape,
                "validate": validation_data.shape,
                "test": test_data.shape
            }
        )

        mlflow.set_tags(
            {
                'mlflow.user':user_profile_name,
                'mlflow.source.type':'JOB'
            }
        )
        
        # Save datasets locally
        train_data.to_csv(os.path.join(args.outputpath, 'train/train.csv'), index=False, header=False)
        validation_data.to_csv(os.path.join(args.outputpath, 'validation/validation.csv'), index=False, header=False)
        test_data[target_col].to_csv(os.path.join(args.outputpath, 'test/test_y.csv'), index=False, header=False)
        test_data.drop([target_col], axis=1).to_csv(os.path.join(args.outputpath, 'test/test_x.csv'), index=False, header=False)
        
        # Save the baseline dataset for model monitoring
        df_model_data.drop([target_col], axis=1).to_csv(os.path.join(args.outputpath, 'baseline/baseline.csv'), index=False, header=False)

        mlflow.log_artifact(local_path=os.path.join(args.outputpath, 'baseline/baseline.csv'))

        mlflow.log_artifact(local_path=os.path.join(args.outputpath, "train/train.csv"))
        mlflow.log_artifact(local_path=os.path.join(args.outputpath, "validation/validation.csv"))
        mlflow.log_artifact(local_path=os.path.join(args.outputpath, "test/test_x.csv"))
        mlflow.log_artifact(local_path=os.path.join(args.outputpath, "test/test_y.csv"))
    
    print("## Processing complete. Exiting.")
