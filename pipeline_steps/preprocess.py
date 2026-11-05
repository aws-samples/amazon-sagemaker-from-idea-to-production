import os
import re
import pandas as pd
import numpy as np
import mlflow
from mlflow.data.pandas_dataset import PandasDataset
from time import gmtime, strftime
try:
    from pipeline_steps.runtime_utils import log_runtime_info
except ImportError:
    def log_runtime_info(): pass


# One-hot encoded columns use a fixed list of categories, so every dataset
# (training data, or new weeks sent to the endpoint later) produces exactly
# the same feature columns in the same order.
CATEGORICAL_FEATURES = {
    "specialty": ["Rheumatology", "Dermatology", "Gastroenterology", "Allergy/Immunology",
                  "Primary Care", "Internal Medicine", "Nurse Practitioner", "Other"],
    "region": ["Northeast", "Southeast", "Midwest", "Southwest", "West"],
    "practice_type": ["Academic", "Hospital", "Group practice", "Solo practice"],
    "segment": ["A", "B", "C", "D"],
    "formulary_access": ["Preferred", "Covered", "Restricted"],
}

# Split by time, never at random: the model is trained on earlier weeks and tested
# on later weeks, so it can't "peek" at the future.
TRAIN_LAST_WEEK = 12        # weeks 1-12 -> train
VALIDATION_LAST_WEEK = 16   # weeks 13-16 -> validation, weeks 17-20 -> test


def process_data(df_data, target_col="prescribed_next_4w"):
    """Turn raw weekly HCP snapshots into model-ready features.

    Returns a DataFrame with the label in the first column, then the features,
    and `week_index` as the last column (used for the time-based split only).
    Missing values stay as NaN: XGBoost handles them natively.
    """
    df = df_data.copy()

    # Binary label: will the HCP write at least one brand Rx in the next 4 weeks?
    df[target_col] = (df["brand_rx_next_4w"] > 0).astype(int)

    # New feature: years in practice at the time of the snapshot
    snapshot_year = pd.to_datetime(df["week_start_date"]).dt.year
    df["years_in_practice"] = snapshot_year - df["graduation_year"]

    # New feature: our brand's share of all prescriptions in the therapy area
    df["brand_share_last_13w"] = (
        df["brand_rx_last_13w"] / df["market_rx_last_13w"].replace(0, np.nan)
    ).round(3)

    # Remove columns not used for modelling:
    # - brand_rx_next_4w: the future, not known at scoring time (it would leak the answer)
    # - hcp_id, week_start_date: identifiers, not features
    # - graduation_year: replaced by years_in_practice
    df = df.drop(columns=["brand_rx_next_4w", "hcp_id", "week_start_date", "graduation_year"])

    # Convert categorical variables to sets of indicator columns
    for col, categories in CATEGORICAL_FEATURES.items():
        dummies = pd.get_dummies(
            pd.Categorical(df[col], categories=categories), dtype=int
        )
        # Readable column names, e.g. "Allergy/Immunology" -> "specialty_allergy_immunology"
        dummies.columns = [f"{col}_" + re.sub(r"[^a-z0-9]+", "_", c.lower()) for c in categories]
        dummies.index = df.index
        df = pd.concat([df.drop(columns=col), dummies], axis=1)

    # Label first, features next, week_index last
    features = [c for c in df.columns if c not in (target_col, "week_index")]
    return df[[target_col] + features + ["week_index"]]


def split_by_time(df_model_data):
    """Split into train, validation and test sets by week, then drop week_index."""
    weeks = df_model_data["week_index"]
    df = df_model_data.drop(columns="week_index")
    return (
        df[weeks <= TRAIN_LAST_WEEK],
        df[(weeks > TRAIN_LAST_WEEK) & (weeks <= VALIDATION_LAST_WEEK)],
        df[weeks > VALIDATION_LAST_WEEK],
    )


def preprocess(
    input_data_s3_path,
    output_s3_prefix,
    pipeline_run_name=None,
    pipeline_run_id=None,
):
    """Preprocess data. MLflow tracking URI and experiment are read from environment."""
    try:
        suffix = strftime('%d-%H-%M-%S', gmtime())

        # Resume or create parent run
        if pipeline_run_id:
            pipeline_run = mlflow.start_run(run_id=pipeline_run_id)
        elif pipeline_run_name:
            pipeline_run = mlflow.start_run(run_name=f"step-pipeline-{pipeline_run_name}")
        else:
            pipeline_run = None
        run = mlflow.start_run(run_name=f"processing-{suffix}", nested=True)
        log_runtime_info()

        # Load data
        df_data = pd.read_csv(input_data_s3_path)

        input_dataset = mlflow.data.from_pandas(df_data, source=input_data_s3_path)
        mlflow.log_input(input_dataset, context="raw_input")

        target_col = "prescribed_next_4w"

        df_model_data = process_data(df_data, target_col)

        model_dataset = mlflow.data.from_pandas(df_model_data)
        mlflow.log_input(model_dataset, context="model_dataset")

        # Split by time: earlier weeks for training, later weeks for testing
        train_data, validation_data, test_data = split_by_time(df_model_data)

        print(f"## Data split > train:{train_data.shape} | validation:{validation_data.shape} | test:{test_data.shape}")
    
        mlflow.log_params(
            {
                "full_dataset": df_model_data.shape,
                "split": f"by week: train 1-{TRAIN_LAST_WEEK}, validation {TRAIN_LAST_WEEK + 1}-{VALIDATION_LAST_WEEK}, test {VALIDATION_LAST_WEEK + 1}+",
                "train": train_data.shape,
                "validate": validation_data.shape,
                "test": test_data.shape
            }
        )

        # Set S3 upload path
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
        
        # We need the baseline dataset for model monitoring
        df_model_data.drop([target_col, "week_index"], axis=1).to_csv(baseline_data_output_s3_path, index=False, header=False)
           
        print("## Data processing complete. Exiting.")
        
        return {
            "train_data": train_data_output_s3_path,
            "validation_data": validation_data_output_s3_path,
            "test_x_data": test_x_data_output_s3_path,
            "test_y_data": test_y_data_output_s3_path,
            "baseline_data": baseline_data_output_s3_path,
            "pipeline_run_id": pipeline_run.info.run_id if pipeline_run else '',
        }

    except Exception as e:
        print(f"Exception in processing script: {e}")
        raise e
    finally:
        while mlflow.active_run():
            mlflow.end_run()
