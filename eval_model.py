"""
Script to evaluate model created by train_model.py
Author: Patrick
Date: Oct 2022
"""

import joblib
import os
import logging
import dvc.api
import pandas as pd
from ml.model import inference, compute_model_metrics

# Set up logging
# Copied and adjusted from
# https://stackoverflow.com/questions/13733552/logger-configuration-to-log-to-file-and-print-to-stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/eval_model.log"),
        logging.StreamHandler()
    ]
)
#Get parameters
params = dvc.api.params_show()

#Load the model from disk
logging.info('Loading model from %s', os.path.abspath(params['train']['model_output']))
lrc = joblib.load(params['train']['model_output'])

#Load the train data from disk
logging.info('Loading train data')
X_train = pd.read_csv(
    os.path.join(
        params['prepare']['data_output'],
        'train_features.csv'
    )
)
y_train = pd.read_csv(
    os.path.join(
        params['prepare']['data_output'],
        'train_targets.csv'
    )
)

# Load the testing data from disk
logging.info('Loading test data')
X_test = pd.read_csv(
    os.path.join(
        params['prepare']['data_output'],
        'test_features.csv'
    )
)
y_test = pd.read_csv(
    os.path.join(
        params['prepare']['data_output'],
        'test_targets.csv'
    )
)

# Get predictions for train split
logging.info('Performing inference on train set')
y_train_preds = inference(lrc, X_train)
logging.info('Calculating train metrics')
precision_train, recall_train, fbeta_train = compute_model_metrics(y_train, y_train_preds)

logging.info('Performing inference on test set')
y_test_preds = inference(lrc, X_test)
precision_test, recall_test, fbeta_test = compute_model_metrics(y_test, y_test_preds)

logging.info(
    f"""
    Evaluation results for LRC
    --------------------------

    Train
    -----
    Precision...........{precision_train}
    Recall..............{recall_train}
    F-Beta..............{fbeta_train}

    Test
    -----
    Precision...........{precision_test}
    Recall..............{recall_test}
    F-Beta..............{fbeta_test}
    """)
