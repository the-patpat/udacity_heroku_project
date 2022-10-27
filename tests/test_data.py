"""
Test script for the ml.data module
Author: Patrick
Date: Oct 2022
"""
import sys
sys.path.insert(0, './')
from ml.data import process_data

def test_process_data(params, split_data):
    """Tests the process_data method"""
    cat_features = params['data']['cat_features']
    target_variable = params['data']['target_variable']
    train, test = split_data
    # Call the process_data function for training data
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features,
        label=target_variable, training=True
    )

    # Number of rows should match
    assert X_train.shape[0] == y_train.shape[0], (
        "Different number of samples in feature and target split")
    # y_train should be a 1d vector
    assert len(y_train.shape) == 1, "There is more than one target variable"

    # Call the process data_function with testing enabled
    _, _, encoder_test, lb_test = process_data(
        test, categorical_features=cat_features,
        label=target_variable, training=False,
        encoder=encoder, lb=lb
    )
    # Running with training=False should return the same encoder and lb
    # We're using is to check for reference equality (same instance)
    assert encoder_test is encoder, (
        "Encoder got copied or modified by function")
    # Same with the LB
    assert lb_test is lb, (
        "LabelBinarizer got modifierd by function"
    )
