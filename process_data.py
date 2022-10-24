import os
import argparse
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from ml.data import process_data

if __name__ == '__main__':
    

    
    parser = argparse.ArgumentParser(description="Pre-process and split data")
    parser.add_argument(
        'input_data',
        type=str,
        help='Path to the input csv file'
        )
    
    parser.add_argument(
        'test_size',
        type=float,
        help='Size of the test split'
    )

    parser.add_argument(
        'random_state',
        type=int,
        help='Seed for reproducible results',
        default=42
    )

    parser.add_argument(
        'target_variable',
        type=str,
        help='Name of the binary variable that the model should predict'
    )

    parser.add_argument(
        'data_output',
        type=str,
        help='Folder where the splits and encoders will be dumped'
        )
    
    args = parser.parse_args()
    #Read in the data
    census_df = pd.read_csv(args.input_data)

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(
        census_df,
        test_size=args.test_size,
        random_state=args.random_state)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    
    #Default target was salary
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label=args.target_variable, training=True
    )

    # Proces the test data with the process_data function.
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label=args.target_variable,
        training=False, encoder=encoder, lb=lb
    )

    # Dump the encoders and the binarizer
    # Needed for the inference pipeline when in production
    joblib.dump(
        encoder,
        os.path.join(args.data_output, 'ohe.joblib'))
    joblib.dump(
        lb,
        os.path.join(args.data_output, 'lb.joblib')
    )

    # Dump the training data
    X_train_df = pd.DataFrame(X_train)
    X_train_df.to_csv(os.path.join(args.data_output, 'train_features.csv'), index=False)
    y_train_df = pd.DataFrame(y_train)
    y_train_df.to_csv(os.path.join(args.data_output, 'train_targets.csv'), index=False)

    # Dump the test data
    X_test_df = pd.DataFrame(X_test)
    X_test_df.to_csv(os.path.join(args.data_output, 'test_features.csv'), index=False)
    y_test_df = pd.DataFrame(y_test)
    y_test_df.to_csv(os.path.join(args.data_output, 'test_targets.csv'), index=False)

