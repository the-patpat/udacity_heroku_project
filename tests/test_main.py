"""
Test script to test the REST api in main.py
"""
import pytest
import sys
import json
from fastapi.testclient import TestClient
import pandas as pd

sys.path.insert(0, './')
from main import app, CensusData, Batch
client = TestClient(app)

def test_greet_user():
    """Tests the root endpoint"""
    response = client.get('/')
    assert response.status_code == 200, "Server did not respond with 200 OK"
    assert response.text == """
    <html>
        <body>
            Hello dear user! Welcome to the model inference API.<br>

            Short explanation on the usage:<br>

            GET / will give you this message.<br>

            POST /inference will run inference
        </body>
    </html>
    """, "GET on root did not return expected string/greeting"


def test_do_inference():
    """
    Tests the batch inference.
    I combined the two required test cases for the post method into one,
    as I can do batch inference and then just check whether both classes
    are contained in the result
    """

    # Fill out the data fields
    # Load the training data
    train_df = pd.read_csv('data/train_unencoded.csv')
    train_df.drop('salary', axis=1, inplace=True)

    # Construct the message
    message_body = Batch(
        samples=[
            CensusData(**sample) for sample in (
                train_df.to_dict(orient='records')
            )]
    )

    # Post the request
    response = client.post(
        '/inference/',
        json=message_body.dict(by_alias=True)
    )

    # Check for the status code
    assert response.status_code == 200, (
        f"""
        Request did not return 200 OK, instead {response.status_code}.
        The error message is:
        {response.json()}
        The message body was
        {message_body.json(by_alias=True)}
        """
    )

    # Check for input data and returned input copy equality
    input_copy = [x['input'] for x in response.json()['results']]
    input_copy_df = pd.DataFrame(input_copy)
    assert input_copy_df.equals(train_df), (
        "Input and copy of input are different"
    )

    # Checked by running inference manually before, this sample results in
    # <=50K inference
    prediction_list = [x['pred_salary_class']
                       for x in response.json()['results']]
    assert set(prediction_list) == {
        '<=50K', '>50K'}, "There are unknown classes in the predictions"
