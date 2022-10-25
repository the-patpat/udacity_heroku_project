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

#Load the pre-encoding dataframes 
logging.info('Loading train unencoded data')
train = pd.read_csv(
    os.path.join(
        params['prepare']['data_output'],
        'train_unencoded.csv'
    )
)

logging.info('loading test unencoded data')
test = pd.read_csv(
    os.path.join(
        params['prepare']['data_output'],
        'test_unencoded.csv'
    )
)

#Check for integrity by checking that age entries are exactly the same
assert (train.iloc[:, 0] == X_train.iloc[:, 0]).all(), "Row mismatch in training data"
assert (test.iloc[:, 0] == X_test.iloc[:, 0]).all()

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

# Measure performance on data slices
train['salary_pred'] = y_train_preds
test['salary_pred'] = y_test_preds

#Need to binarize label to enable comparison
# <=50K == 0
# >50K == 1
num_classes = len(train['salary'].unique())
for cl, val in enumerate(train['salary'].unique()):
    train.loc[train['salary'] == val, 'salary'] = num_classes- 1 - cl
assert (train['salary'] == y_train.iloc[:, 0]).all(), "Label mismatch"

for cl, val in enumerate(test['salary'].unique()):
    test.loc[test['salary'] == val, 'salary'] = cl
assert (test['salary'] == y_test.iloc[:, 0]).all(), "Label mismatch"

# Compute slice performance
for cat in params['data']['cat_features']:
    for value in train[cat].unique():
        y_group_preds_tr = train[train[cat]==value]['salary_pred'].astype(int)
        y_group_tgts_tr = train[train[cat]==value]['salary'].astype(int)
        y_group_preds_ts = test[train[cat]==value]['salary_pred'].astype(int)
        y_group_tgts_ts = test[train[cat]==value]['salary'].astype(int)
        logging.info(y_group_preds_tr)
        logging.info(y_group_tgts_tr)
        p_tr, r_tr, f_beta_tr = compute_model_metrics(y_group_tgts_tr, y_group_preds_tr)
        p_ts, r_ts, f_beta_ts = compute_model_metrics(y_group_tgts_ts, y_group_preds_ts)
        print(
            f"""
            ---- Slice Analysis ----
            For group {value} in category {cat}:
            """)
        logging.info(
            f"""
            Slice evaluation results for LRC
            --------------------------

            Train
            -----
            Precision...........{p_tr}
            Recall..............{r_tr}
            F-Beta..............{f_beta_tr}

            Test
            -----
            Precision...........{p_ts}
            Recall..............{r_ts}
            F-Beta..............{f_beta_ts}
            """)

