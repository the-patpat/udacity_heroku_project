import pytest
import joblib
import sys
sys.path.insert(0, './')
from ml.model import inference, compute_model_metrics
from ml.data import process_data

def test_inference(model, encoded_test_data):
    
    # Get the encoded testing data
    X_test, y_test = encoded_test_data
    
    # Perform inference
    y_pred = inference(model, X_test)

    # Check if the shapes are the same
    assert y_pred.ravel().shape == y_test.values.ravel().shape, (
        "Prediction shape and target shape mismatch"
    )
    
def test_compute_model_metrics(model, encoded_test_data):

    # Get encoded testing data
    X_test, y_test = encoded_test_data
    
    # Perform inference (usually we should have reference prediction)
    y_pred = inference(model, X_test)
    
    # Compute the metrics
    pr, rc, f_beta = compute_model_metrics(y_test, y_pred)

    # Check the datatypes
    assert isinstance(pr, float), "Returned precision is not a float"
    assert isinstance(rc, float), "Returned recall is not a float"
    assert isinstance(f_beta, float), "Returned f-beta is not a float"

    # Viable test: check that metrics are holding up to minimal standards
    assert pr >= 0.1, "Precision is lower than 0.1"
    assert rc >= 0.1, "Recall is lower than 0.1"
    assert f_beta >= 0.1, "F-Beta is lower than 0.1"
    