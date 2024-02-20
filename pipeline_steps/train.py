import boto3
import sagemaker
from sagemaker.inputs import TrainingInput

def train(
    train_data_s3_path,
    validation_data_s3_path,
    train_instance_type,
    train_instance_count=1,
):
    sagemaker_session = sagemaker.session.Session()

    xgboost_image_uri = sagemaker.image_uris.retrieve("xgboost", sagemaker_session.boto_region_name, version="1.5-1")
    
    # Instantiate an XGBoost estimator object
    estimator = sagemaker.estimator.Estimator(
        image_uri=xgboost_image_uri,
        role=sagemaker.get_execution_role(), 
        instance_type=train_instance_type,
        instance_count=train_instance_count,
        # output_path=output_s3_url,
        # base_job_name=f"{pipeline_name}-train",
    )

    # Define algorithm hyperparameters
    estimator.set_hyperparameters(
        num_round=150, # the number of rounds to run the training
        max_depth=5, # maximum depth of a tree
        eta=0.5, # step size shrinkage used in updates to prevent overfitting
        alpha=2.5, # L1 regularization term on weights
        objective="binary:logistic",
        eval_metric="auc", # evaluation metrics for validation data
        subsample=0.8, # subsample ratio of the training instance
        colsample_bytree=0.8, # subsample ratio of columns when constructing each tree
        min_child_weight=3, # minimum sum of instance weight (hessian) needed in a child
        early_stopping_rounds=10, # the model trains until the validation score stops improving
        verbosity=1, # verbosity of printing messages
    )

    training_inputs = {
        "train": TrainingInput(
            train_data_s3_path,
            content_type="text/csv",
        ),
        "validation": TrainingInput(
            validation_data_s3_path,
            content_type="text/csv",
        ),
    }
        
    estimator.fit(training_inputs)
    
    return estimator.latest_training_job.name