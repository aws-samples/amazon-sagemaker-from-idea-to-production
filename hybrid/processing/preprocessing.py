import pandas as pd
import numpy as np
import argparse
import os
import re

from time import gmtime, strftime, sleep
import traceback

user_profile_name = os.getenv('USER')
def _parse_args():

    parser = argparse.ArgumentParser()
    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--filepath', type=str, default='/opt/ml/processing/input/')
    parser.add_argument('--filename', type=str, default='hcp_weekly_snapshots.csv')
    parser.add_argument('--outputpath', type=str, default='/opt/ml/processing/output/')

    return parser.parse_known_args()

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

if __name__=="__main__":
    # Process arguments
    args, _ = _parse_args()

    target_col = "prescribed_next_4w"

    import mlflow

    # Set the Tracking Server URI using the ARN of the Tracking Server you created
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_ARN'])

    # Enable autologging in MLflow
    mlflow.autolog()

    # Use the active run_id to log
    with mlflow.start_run(run_id=os.environ['MLFLOW_RUN_ID']) as run:
        # process data
        df_model_data = process_data(pd.read_csv(os.path.join(args.filepath, args.filename)), target_col)

        # Split by time: earlier weeks for training, later weeks for testing
        train_data, validation_data, test_data = split_by_time(df_model_data)

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

        # Save the baseline dataset for model monitoring
        df_model_data.drop([target_col, 'week_index'], axis=1).to_csv(os.path.join(args.outputpath, 'baseline/baseline.csv'), index=False, header=False)

        mlflow.log_artifact(local_path=os.path.join(args.outputpath, 'baseline/baseline.csv'))

        mlflow.log_artifact(local_path=os.path.join(args.outputpath, "train/train.csv"))
        mlflow.log_artifact(local_path=os.path.join(args.outputpath, "validation/validation.csv"))
        mlflow.log_artifact(local_path=os.path.join(args.outputpath, "test/test_x.csv"))
        mlflow.log_artifact(local_path=os.path.join(args.outputpath, "test/test_y.csv"))

    print("## Processing complete. Exiting.")
