import pandas as pd
import numpy as np
import boto3
from sagemaker.s3_utils import parse_s3_url
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def preprocess(
    data_s3_path,
    output_s3_prefix,
):
    
    s3 = boto3.client("s3")
    
    bucket, object_key = parse_s3_url(data_s3_path)
    s3.download_file(bucket, object_key, "input_data.csv")
    
    # Load data
    df_data = pd.read_csv("input_data.csv", sep=";")
    
    target_col = "y"

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
    
    # Shuffle and splitting dataset
    train_data, validation_data, test_data = np.split(
        df_model_data.sample(frac=1, random_state=1729),
        [int(0.7 * len(df_model_data)), int(0.9 * len(df_model_data))],
    )

    print(f"Data split > train:{train_data.shape} | validation:{validation_data.shape} | test:{test_data.shape}")
    
    # Save datasets locally
    train_data.to_csv("train.csv", index=False, header=False)
    validation_data.to_csv("validation.csv", index=False, header=False)
    test_data[target_col].to_csv('test_y.csv', index=False, header=False)
    test_data.drop([target_col], axis=1).to_csv('test_x.csv', index=False, header=False)
    
    # Save the baseline dataset for model monitoring
    df_model_data.drop([target_col], axis=1).to_csv("baseline.csv", index=False, header=False)
    
    # Upload datasets to S3
    train_data_output_s3_path = f"{output_s3_prefix}/train/train.csv"
    validation_data_output_s3_path = f"{output_s3_prefix}/validation/validation.csv"
    test_x_data_output_s3_path = f"{output_s3_prefix}/test/test_x.csv"
    test_y_data_output_s3_path = f"{output_s3_prefix}/test/test_y.csv"
    baseline_data_output_s3_path = f"{output_s3_prefix}/baseline/baseline.csv"
    
    s3.upload_file("train.csv", *parse_s3_url(train_data_output_s3_path))
    s3.upload_file("validation.csv", *parse_s3_url(validation_data_output_s3_path))
    s3.upload_file("test_x.csv", *parse_s3_url(test_x_data_output_s3_path))
    s3.upload_file("test_y.csv", *parse_s3_url(test_y_data_output_s3_path))
    s3.upload_file("baseline.csv", *parse_s3_url(baseline_data_output_s3_path))
    
    
    print("## Pre-processing complete. Exiting.")
    
    return {
        "train_data":train_data_output_s3_path,
        "validation_data":validation_data_output_s3_path,
        "test_x_data":test_x_data_output_s3_path,
        "test_y_data":test_y_data_output_s3_path,
        "baseline_data":baseline_data_output_s3_path
    }


