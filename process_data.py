import os
import argparse
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
import dvc.api
from ml.data import process_data

if __name__ == '__main__':

    params = dvc.api.params_show()

    # Read in the data
    census_df = pd.read_csv(params['prepare']['input_data'])

    # Optional enhancement, use K-fold cross validation instead of a
    # train-test split.
    train, test = train_test_split(
        census_df,
        test_size=float(params['data']['test_size']),
        random_state=int(params['random_state'])
    )

    cat_features = params['data']['cat_features']

    # Default target was salary
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features,
        label=params['data']['target_variable'], training=True
    )

    # Proces the test data with the process_data function.
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features,
        label=params['data']['target_variable'],
        training=False, encoder=encoder, lb=lb
    )

    # Dump the encoders and the binarizer
    # Needed for the inference pipeline when in production
    joblib.dump(
        encoder,
        os.path.join(params['prepare']['data_output'], 'ohe.joblib'))
    joblib.dump(
        lb,
        os.path.join(params['prepare']['data_output'], 'lb.joblib')
    )

    # Dump the training data
    X_train_df = pd.DataFrame(X_train)
    X_train_df.to_csv(
        os.path.join(
            params['prepare']['data_output'],
            'train_features.csv'),
        index=False)
    y_train_df = pd.DataFrame(y_train)
    y_train_df.to_csv(
        os.path.join(
            params['prepare']['data_output'],
            'train_targets.csv'),
        index=False)

    # Dump the test data
    X_test_df = pd.DataFrame(X_test)
    X_test_df.to_csv(
        os.path.join(
            params['prepare']['data_output'],
            'test_features.csv'),
        index=False)
    y_test_df = pd.DataFrame(y_test)
    y_test_df.to_csv(
        os.path.join(
            params['prepare']['data_output'],
            'test_targets.csv'),
        index=False)
