"""
Conftest.py for fixtures that are reused in multiple test scripts
Author: Patrick
Date: Oct 2022
"""
import sys
import joblib
import pandas as pd
import yaml

@pytest.fixture(scope='session')
def params():
    """Returns the dvc parameters"""
    par = None
    with open('params.yaml', 'r') as yaml_file:
        par = yaml.load(yaml_file, Loader=SafeLoader)
    return par

@pytest.fixture(scope='session')
def split_data():
    """Returns the training data"""
    train = pd.read_csv('data/train_unencoded.csv')
    test = pd.read_csv('data/test_unencoded.csv')
    return train, test


@pytest.fixture(scope='session')
def model():
    return joblib.load('model/lrc_census.joblib')