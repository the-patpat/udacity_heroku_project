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
from utils import MakeFileHandler

# Set up logging
# Copied and adjusted from
# https://stackoverflow.com/questions/13733552/logger-configuration-to-log-to-file-and-print-to-stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        MakeFileHandler("logs/eval_model.log"),
        logging.StreamHandler()
    ]
)
# Get parameters
params = dvc.api.params_show()

# Load the model from disk
logging.info(
    'Loading model from %s',
    os.path.abspath(
        params['train']['model_output']))
lrc = joblib.load(params['train']['model_output'])

# Load the train data from disk
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

# Load the pre-encoding dataframes
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

# Check for integrity by checking that age entries are exactly the same
assert (train.iloc[:, 0] == X_train.iloc[:, 0]
        ).all(), "Row mismatch in training data"
assert (test.iloc[:, 0] == X_test.iloc[:, 0]).all()

# Get predictions for train split
logging.info('Performing inference on train set')
y_train_preds = inference(lrc, X_train)
logging.info('Calculating train metrics')
precision_train, recall_train, fbeta_train = compute_model_metrics(
    y_train, y_train_preds)

logging.info('Performing inference on test set')
y_test_preds = inference(lrc, X_test)
precision_test, recall_test, fbeta_test = compute_model_metrics(
    y_test, y_test_preds)

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

# Load label binarizer
lb = joblib.load(
    os.path.join(
        params['prepare']['data_output'],
        'lb.joblib'
    )
)

# Binarize targets for comparison. Could also be done by loading the files
train['salary'] = lb.transform(train['salary'])
test['salary'] = lb.transform(test['salary'])

# Check that the lb is doing the job properly
assert (train['salary'] == y_train.iloc[:, 0]).all(), "Label mismatch"
assert (test['salary'] == y_test.iloc[:, 0]).all(), "Label mismatch"

# Compute slice performance
metrics_df = pd.DataFrame()
for cat in params['data']['cat_features']:
    for value in train[cat].unique():

        # Get the group slices
        # Use astype int to have the correct type
        y_group_preds_tr = train[train[cat] ==
                                 value]['salary_pred'].astype(int)
        y_group_tgts_tr = train[train[cat] == value]['salary'].astype(int)
        y_group_preds_ts = test[train[cat] == value]['salary_pred'].astype(int)
        y_group_tgts_ts = test[train[cat] == value]['salary'].astype(int)

        # Compute the metrics
        p_tr, r_tr, f_beta_tr = compute_model_metrics(
            y_group_tgts_tr, y_group_preds_tr)
        p_ts, r_ts, f_beta_ts = compute_model_metrics(
            y_group_tgts_ts, y_group_preds_ts)

        metrics_df = pd.concat(
            [metrics_df,
             pd.DataFrame(
                 [
                     [cat, value, 'train', p_tr, r_tr, f_beta_tr],
                     [cat, value, 'test', p_ts, r_ts, f_beta_ts]
                 ]
             )]
        )

        # Log the metrics
        logging.info(
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

# Put out to file
metrics_df.reset_index()
metrics_df.columns = [
    'category',
    'group',
    'split',
    'precision',
    'recall',
    'f-beta']

with open('slice_output.txt', 'w') as f:
    f.write(metrics_df.to_string(index=False))
    f.close()
