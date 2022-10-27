import os
import argparse
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
import dvc.api
import logging
from ml.data import process_data
from utils import MakeFileHandler

# Set up logging
# Copied and adjusted from
# https://stackoverflow.com/questions/13733552/logger-configuration-to-log-to-file-and-print-to-stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        MakeFileHandler("logs/process_data.log"),
        logging.StreamHandler()
    ]
)

if __name__ == '__main__':

    params = dvc.api.params_show()

    # Read in the data
    census_df = pd.read_csv(params['prepare']['input_data'])

    # Optional enhancement, use K-fold cross validation instead of a
    # train-test split.
    logging.info(
        'Splitting data with test_size=%f',
        params['data']['test_size']
    )

    train, test = train_test_split(
        census_df,
        test_size=float(params['data']['test_size']),
        random_state=int(params['random_state'])
    )

    logging.info('Dumping un-encoded train and test frames')
    train.to_csv(
        os.path.join(params['prepare']['data_output'],
                     'train_unencoded.csv'),
        index=False
    )
    test.to_csv(
        os.path.join(params['prepare']['data_output'],
                     'test_unencoded.csv'),
        index=False
    )

    cat_features = params['data']['cat_features']
    logging.info("Categorical features are %s", str(cat_features))
    # Default target was salary

    logging.info('Processing training data')
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features,
        label=params['data']['target_variable'], training=True
    )

    # Proces the test data with the process_data function.
    logging.info('Processing test data')
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features,
        label=params['data']['target_variable'],
        training=False, encoder=encoder, lb=lb
    )

    # Dump the encoders and the binarizer
    # Needed for the inference pipeline when in production
    joblib.dump(
        encoder,
        os.path.join(params['prepare']['data_output'], 'ohe.joblib')
    )
    logging.info(
        'Dumped one-hot-encoder at %s',
        os.path.abspath(
            os.path.join(params['prepare']['data_output'], 'ohe.joblib')
        )
    )

    joblib.dump(
        lb,
        os.path.join(params['prepare']['data_output'], 'lb.joblib')
    )
    logging.info(
        'Dumped linear binarizer at %s',
        os.path.abspath(
            os.path.join(params['prepare']['data_output'], 'jb.joblib')
        )
    )
    # Dump the training data
    X_train_df = pd.DataFrame(X_train)
    X_train_df.to_csv(
        os.path.join(
            params['prepare']['data_output'],
            'train_features.csv'),
        index=False)
    logging.info(
        'Dumped train features at %s',
        os.path.abspath(
            os.path.join(
                params['prepare']['data_output'],
                'train_features.csv')))

    y_train_df = pd.DataFrame(y_train)
    y_train_df.to_csv(
        os.path.join(
            params['prepare']['data_output'],
            'train_targets.csv'),
        index=False)
    logging.info(
        'Dumped train targets at %s',
        os.path.abspath(
            os.path.join(params['prepare']['data_output'], 'train_targets.csv')
        )
    )

    # Dump the test data
    X_test_df = pd.DataFrame(X_test)
    X_test_df.to_csv(
        os.path.join(
            params['prepare']['data_output'],
            'test_features.csv'),
        index=False)
    logging.info(
        'Dumped test features at %s',
        os.path.abspath(
            os.path.join(params['prepare']['data_output'], 'test_features.csv')
        )
    )
    y_test_df = pd.DataFrame(y_test)
    y_test_df.to_csv(
        os.path.join(
            params['prepare']['data_output'],
            'test_targets.csv'),
        index=False)
    logging.info(
        'Dumped test targets at %s',
        os.path.abspath(
            os.path.join(params['prepare']['data_output'], 'test_targets.csv')
        )
    )
