import argparse
import json
import os
import pickle as pkl

import pandas as pd
from sklearn.metrics import roc_auc_score
from sagemaker_xgboost_container.data_utils import get_dmatrix
import xgboost as xgb

from time import gmtime, strftime

suffix = strftime('%d-%H-%M-%S', gmtime())
user_profile_name = os.getenv('USER', 'sagemaker')
experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME')
region = os.getenv('REGION')


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
    dtrain = get_dmatrix(args.train, 'CSV')
    dval = get_dmatrix(args.validation, 'CSV')
    watchlist = [(dtrain, 'train'), (dval, 'validation')] if dval is not None else [(dtrain, 'train')]
    sm_training_env = json.loads(args.sm_training_env)

    import mlflow
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_ARN'))
    mlflow.set_experiment(experiment_name=experiment_name if experiment_name else f'train-{suffix}')
    mlflow.xgboost.autolog(log_model_signatures=False, log_datasets=False)

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

    with mlflow.start_run(
        run_name=f'container-training-{suffix}',
        description='xgboost running in SageMaker container in script mode',
    ) as run:
        mlflow.set_tags({
            'mlflow.user': user_profile_name,
            'mlflow.source.type': 'JOB',
            'mlflow.source.name': f"https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/jobs/{sm_training_env['job_name']}"
        })
        if dtrain:
            xgb_train_args['is_master'] = True
            _xgb_train(**xgb_train_args)
        else:
            raise ValueError('Training channel must have data to train model.')
