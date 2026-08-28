"""Training script for the BYOC XGBoost container experiment.

This is a copy of ``training/train.py`` with one intentional change: the
original uses ``sagemaker_xgboost_container.data_utils.get_dmatrix`` to load
CSV input channels, but that module only exists inside AWS's managed XGBoost
container. The BYOC image built from ``byoc/Dockerfile`` is derived from
``python:3.12-slim-bookworm`` and does not have it.

To keep the BYOC image free of AWS-specific framework coupling, this script
replaces ``get_dmatrix`` with a small local helper, ``load_csv_dmatrix``,
that does the same job using only pandas and xgboost. The helper follows
the standard SageMaker XGBoost CSV convention: header-less files where the
first column is the label.
"""
import argparse
import json
import os
import pickle as pkl

import pandas as pd
from sklearn.metrics import roc_auc_score
import xgboost as xgb

from time import gmtime, strftime

suffix = strftime('%d-%H-%M-%S', gmtime())
user_profile_name = os.getenv('USER', 'sagemaker')
experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME')
region = os.getenv('REGION')


def load_csv_dmatrix(channel_dir):
    """Build an ``xgb.DMatrix`` from all CSV files in a SageMaker channel dir.

    SageMaker mounts training input channels at ``/opt/ml/input/data/<channel>/``
    as one or more files. For CSV the standard convention is header-less rows
    with the label in the first column. This helper:

    * returns ``None`` if the channel dir is missing or empty (e.g. no
      validation channel was supplied),
    * concatenates every non-hidden file in the directory into a single
      DataFrame,
    * splits column 0 as the label and columns 1+ as features,
    * returns an ``xgb.DMatrix`` ready for ``xgb.train``.

    This replaces ``sagemaker_xgboost_container.data_utils.get_dmatrix`` with
    a dependency-free equivalent for the BYOC image.
    """
    if channel_dir is None or not os.path.isdir(channel_dir):
        return None
    files = sorted(
        os.path.join(channel_dir, f)
        for f in os.listdir(channel_dir)
        if not f.startswith('.') and os.path.isfile(os.path.join(channel_dir, f))
    )
    if not files:
        return None
    frames = [pd.read_csv(f, header=None) for f in files]
    df = pd.concat(frames, ignore_index=True)
    labels = df.iloc[:, 0].values
    features = df.iloc[:, 1:].values
    return xgb.DMatrix(features, label=labels)


def _xgb_train(params, dtrain, dval, evals, num_boost_round, model_dir, is_master):
    booster = xgb.train(params=params, dtrain=dtrain, evals=evals, num_boost_round=num_boost_round)

    val_auc = roc_auc_score(dval.get_label(), booster.predict(dval))
    train_auc = roc_auc_score(dtrain.get_label(), booster.predict(dtrain))
    mlflow.log_params(params)
    mlflow.log_metrics({'validation_auc': val_auc, 'train_auc': train_auc})
    print(f'[0]#011train-auc:{train_auc}#011validation-auc:{val_auc}')

    if is_master:
        model_location = model_dir + '/xgboost-model'
        pkl.dump(booster, open(model_location, 'wb'))
        print(f'Stored trained model at {model_location}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_depth', type=int)
    parser.add_argument('--eta', type=float)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--gamma', type=int)
    parser.add_argument('--min_child_weight', type=float)
    parser.add_argument('--subsample', type=float)
    parser.add_argument('--colsample_bytree', type=float)
    parser.add_argument('--verbosity', type=int)
    parser.add_argument('--objective', type=str)
    parser.add_argument('--num_round', type=int)
    parser.add_argument('--early_stopping_rounds', type=int)
    parser.add_argument('--tree_method', type=str, default='auto')
    parser.add_argument('--predictor', type=str, default='auto')
    parser.add_argument('--output_data_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--sm_hosts', type=str, default=os.environ.get('SM_HOSTS'))
    parser.add_argument('--sm_current_host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    parser.add_argument('--sm_training_env', type=str, default=os.environ.get('SM_TRAINING_ENV'))

    args, _ = parser.parse_known_args()

    sm_hosts = json.loads(args.sm_hosts)
    sm_current_host = args.sm_current_host
    dtrain = load_csv_dmatrix(args.train)
    dval = load_csv_dmatrix(args.validation)
    watchlist = [(dtrain, 'train'), (dval, 'validation')] if dval is not None else [(dtrain, 'train')]
    sm_training_env = json.loads(args.sm_training_env)

    import mlflow
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_ARN'))
    mlflow.set_experiment(experiment_name=experiment_name if experiment_name else f'train-{suffix}')
    mlflow.xgboost.autolog(log_model_signatures=True, log_datasets=True)

    train_hp = {
        'max_depth': args.max_depth, 'eta': args.eta, 'gamma': args.gamma,
        'min_child_weight': args.min_child_weight, 'subsample': args.subsample,
        'verbosity': args.verbosity, 'objective': args.objective,
        'tree_method': args.tree_method, 'predictor': args.predictor,
    }

    xgb_train_args = dict(
        params=train_hp, dtrain=dtrain, dval=dval, evals=watchlist,
        num_boost_round=args.num_round, model_dir=args.model_dir,
    )

    # Resume parent run if MLFLOW_RUN_ID is set (for nested run grouping)
    parent_run_id = os.getenv('MLFLOW_RUN_ID')
    if parent_run_id:
        mlflow.start_run(run_id=parent_run_id)

    with mlflow.start_run(
        run_name=f'container-training-{suffix}',
        description='xgboost running in SageMaker container in script mode',
        nested=True if parent_run_id else False,
    ) as run:
        mlflow.set_tags({
            'mlflow.user': user_profile_name,
            'mlflow.source.type': 'JOB',
            'mlflow.source.name': f"https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/jobs/{sm_training_env['job_name']}",
            'sagemaker.container_image': sm_training_env.get('additional_framework_parameters', {}).get('sagemaker_training_image', os.environ.get('TRAINING_JOB_ARN', 'n/a')),
            'sagemaker.job_name': sm_training_env['job_name'],
            'sagemaker.runtime': 'sagemaker',
        })
        if dtrain:
            xgb_train_args['is_master'] = True
            _xgb_train(**xgb_train_args)
        else:
            raise ValueError('Training channel must have data to train model.')

        # Write MLflow run_id to model dir so it gets packaged in model.tar.gz
        with open(os.path.join(args.model_dir, 'mlflow_run_id.txt'), 'w') as f:
            f.write(run.info.run_id)
