# Script to train machine learning model.

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model

if __name__ == '__main__':

    #Argument parsing

    # Load the train data
    pass  
    # Train and save a model.
    model = train_model(X_train, y_train)
    joblib.dump(model, 'lrc_census.joblib')
