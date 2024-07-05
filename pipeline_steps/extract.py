import boto3
import pandas as pd
import numpy as np
import mlflow
from mlflow.data.pandas_dataset import PandasDataset
from time import gmtime, strftime
from sagemaker.session import Session
from sagemaker.feature_store.feature_store import FeatureStore
from sagemaker.feature_store.feature_group import FeatureGroup

def extract_features(
    feature_group_name,
    query_output_s3_path,
):
    region = boto3.Session().region_name
    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client(service_name="sagemaker", region_name=region)
    featurestore_runtime = boto_session.client(service_name="sagemaker-featurestore-runtime",region_name=region)
    
    # Create FeatureStore session object
    feature_store_session = Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_featurestore_runtime_client=featurestore_runtime,
    )

    feature_store = FeatureStore(sagemaker_session=feature_store_session)
    dataset_feature_group = FeatureGroup(feature_group_name)
    
    # Create dataset builder to retrieve the most recent version of each record
    builder = feature_store.create_dataset(
        base=dataset_feature_group,
        # included_feature_names=inlcuded_feature_names,
        output_path=query_output_s3_path,
    ).with_number_of_recent_records_by_record_identifier(1)
    
    df_dataset, query = builder.to_dataframe()
    
    return df_dataset
    
def prepare_datasets(
    feature_group_name,
    output_s3_prefix,
    query_output_s3_path,
    tracking_server_arn,
    experiment_name=None,
    pipeline_run_name=None,
    run_id=None,
):
    try:
        suffix = strftime('%d-%H-%M-%S', gmtime())
        mlflow.set_tracking_uri(tracking_server_arn)
        experiment = mlflow.set_experiment(experiment_name=experiment_name if experiment_name else f"{prepare_datasets.__name__ }-{suffix}")
        pipeline_run = mlflow.start_run(run_name=pipeline_run_name) if pipeline_run_name else None            
        run = mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run(run_name=f"feature-extraction-{suffix}", nested=True)

        target_col = "y"
        feature_store_col = ['event_time', 'record_id']
        
        df_model_data = extract_features(
            feature_group_name, 
            query_output_s3_path,
        ).drop(feature_store_col, axis=1)
        
        print(f"Extracted {len(df_model_data)} rows from the feature group {feature_group_name}")

        # log dataset
        input_dataset = mlflow.data.from_pandas(df_model_data, source=output_s3_prefix)
        mlflow.log_input(input_dataset, context="featureset")
    
         # Shuffle and splitting dataset
        train_data, validation_data, test_data = np.split(
            df_model_data.sample(frac=1, random_state=1729),
            [int(0.7 * len(df_model_data)), int(0.9 * len(df_model_data))],
        )
    
        print(f"Data split > train:{train_data.shape} | validation:{validation_data.shape} | test:{test_data.shape}")

        mlflow.log_params(
            {
                "full_dataset": df_model_data.shape,
                "train": train_data.shape,
                "validate": validation_data.shape,
                "test": test_data.shape
            }
        )

        # Set S3 upload paths
        train_data_output_s3_path = f"{output_s3_prefix}/train/train.csv"
        validation_data_output_s3_path = f"{output_s3_prefix}/validation/validation.csv"
        test_x_data_output_s3_path = f"{output_s3_prefix}/test/test_x.csv"
        test_y_data_output_s3_path = f"{output_s3_prefix}/test/test_y.csv"
        baseline_data_output_s3_path = f"{output_s3_prefix}/baseline/baseline.csv"
        
        # Upload datasets to S3
        train_data.to_csv(train_data_output_s3_path, index=False, header=False)
        validation_data.to_csv(validation_data_output_s3_path, index=False, header=False)
        test_data[target_col].to_csv(test_y_data_output_s3_path, index=False, header=False)
        test_data.drop([target_col], axis=1).to_csv(test_x_data_output_s3_path, index=False, header=False)
        
        # Save the baseline dataset for model monitoring
        df_model_data.drop([target_col], axis=1).to_csv(baseline_data_output_s3_path, index=False, header=False)
        
        s3 = boto3.client("s3")
      
        print(f"Datasets are uploaded to S3: {output_s3_prefix}. Exiting.")
        
        return {
           "train_data":train_data_output_s3_path,
            "validation_data":validation_data_output_s3_path,
            "test_x_data":test_x_data_output_s3_path,
            "test_y_data":test_y_data_output_s3_path,
            "baseline_data":baseline_data_output_s3_path,
            "experiment_name":experiment.name,
            "pipeline_run_id":pipeline_run.info.run_id if pipeline_run else ''
        }
    except Exception as e:
        print(f"Exception in processing script: {e}")
        raise e
    finally:
        mlflow.end_run()
    
