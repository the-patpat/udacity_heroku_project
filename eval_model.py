"""
Script to evaluate model created by train_model.py
Author: Patrick
Date: Oct 2022
"""

import joblib
import pandas as pd
from ml.model import inference, compute_model_metrics

# Load the model from disk
lrc = joblib.load('lrc_census.joblib')

#Load the train data from disk
X_train = pd.read_csv('data/train_features.csv')
y_train = pd.read_csv('data/train_targets.csv')
# Load the testing data from disk
X_test = pd.read_csv('data/test_features.csv')
y_test = pd.read_csv('data/test_targets.csv')

y_test_preds = inference(lrc, X_test)
precision_test, recall_test, fbeta_test = compute_model_metrics(y_test, y_test_preds)
y_train_preds = inference(lrc, X_train)
precision_train, recall_train, fbeta_train = compute_model_metrics(y_train, y_train_preds)

print(
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
