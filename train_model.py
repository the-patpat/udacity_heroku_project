"""
Training script for the machine learning model
Author: Patrick
Date: Oct 2022
"""

import joblib
import logging
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import dvc.api
from ml.data import process_data
from ml.model import train_model

# Set up logging
# Copied and adjusted from
# https://stackoverflow.com/questions/13733552/logger-configuration-to-log-to-file-and-print-to-stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/train_model.log"),
        logging.StreamHandler()
    ]
)
if __name__ == '__main__':

    #Argument parsing
    params = dvc.api.params_show()

    # Load the train data
    logging.info(
        'Loading train features from %s',
        os.path.abspath(os.path.join(
            params['prepare']['data_output'],
            'train_features.csv'
        ))
    )
    X_train = pd.read_csv(
        os.path.join(
            params['prepare']['data_output'],
            'train_features.csv'
        )
    )
    # Load the test data
    logging.info(
        'Loading train targets from %s',
        os.path.abspath(os.path.join(
            params['prepare']['data_output'],
            'train_targets.csv'
        ))
    )
    y_train = pd.read_csv(
        os.path.join(
            params['prepare']['data_output'],
            'train_targets.csv'
        )
    )
     # Train and save a model.
    logging.info('Fitting model')
    model = train_model(X_train, y_train)
    logging.info(
        'Dumping model to %s',
        os.path.abspath(
            params['train']['model_output']
        )
    )
    joblib.dump(model, params['train']['model_output'])
    logging.info('Dumped model')