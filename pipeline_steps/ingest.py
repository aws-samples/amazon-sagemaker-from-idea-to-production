import boto3
from sagemaker.feature_store.feature_group import FeatureGroup
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timezone, date
from sagemaker.session import Session


def process_and_ingest(
    input_s3_url,
    feature_group_name,
):
    def generate_event_timestamp():
        # naive datetime representing local time
        naive_dt = datetime.now()
        # take timezone into account
        aware_dt = naive_dt.astimezone()
        # time in UTC
        utc_dt = aware_dt.astimezone(timezone.utc)
        # transform to ISO-8601 format
        event_time = utc_dt.isoformat(timespec='milliseconds')
        event_time = event_time.replace('+00:00', 'Z')
        return event_time
    
    def convert_col_name(c):
        return c.replace('.', '_').replace('-', '_').rstrip('_')
        
    # Download data from S3
    df_data = pd.read_csv(input_s3_url, sep=";")
    
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
    
    df_model_data['event_time'] = generate_event_timestamp()
    df_model_data['record_id'] = [f'R{i}' for i in range(len(df_model_data))]
    
    df_model_data = df_model_data.rename(columns=convert_col_name)
    
    df_model_data = df_model_data.convert_dtypes(infer_objects=True, convert_boolean=False)
    df_model_data['record_id'] = df_model_data['record_id'].astype('string')
    df_model_data['event_time'] = df_model_data['event_time'].astype('string')
    
    # Ingest data into the feature group
    dataset_feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=Session())
    
    print(f'Ingesting data into feature group: {dataset_feature_group.name} ...')
    dataset_feature_group.ingest(data_frame=df_model_data, max_processes=4, wait=True)
    print(f'{len(df_model_data)} customer records ingested into feature group: {dataset_feature_group.name}')
    
    return dataset_feature_group.describe()['FeatureGroupArn']