"""
This module implements the REST API for the model using FastAPI

Author: Patrick
Date: Oct 2022
"""
import fastapi
import joblib
import os
import logging
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List 
import pandas as pd
from ml.model import inference
from ml.data import process_data
from utils import MakeFileHandler


# Need to pull data if we're on heroku
# DVC can only be imported after this section
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

import dvc.api
app = fastapi.FastAPI()
params = dvc.api.params_show()

class CensusData(BaseModel):
    """
    The data model for the ingestion body

    Note that fastapi converts the underscore in these names automatically to 
    hyphens
    """
    age: int 
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(None, alias='education-num')
    marital_status: str = Field(None, alias='marital-status')
    occupation: str 
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(None, alias='capital-gain')
    capital_loss: int = Field(None, alias='capital-loss')
    hours_per_week: int = Field(None, alias='hours-per-week')
    native_country: str = Field(None, alias='native-country')
    class Config:
        schema_extra = {
            "example": {
                "age" : 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States"
            }
        }
        allow_population_by_field_name = True
class Batch(BaseModel):
    samples: List[CensusData]
    class Config:
        schema_extra = {
                "example": {"samples": [{
                "age" : 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States"
            }]}
        }
        allow_population_by_field_name = True

class InferenceResult(BaseModel):
    input: CensusData
    pred_salary_class: str
    class Config:
        schema_extra={
            "input": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States"
            },
            "pred_salary_class": "<=50K"
        }
class BatchResult(BaseModel):
    results: List[InferenceResult]
    class Config:
        schema_extra={
            "results": [
                {
                    "input": {
                        "age": 39,
                        "workclass": "State-gov",
                        "fnlgt": 77516,
                        "education": "Bachelors",
                        "education-num": 13,
                        "marital-status": "Never-married",
                        "occupation": "Adm-clerical",
                        "relationship": "Not-in-family",
                        "race": "White",
                        "sex": "Male",
                        "capital-gain": 2174,
                        "capital-loss": 0,
                        "hours-per-week": 40,
                        "native-country": "United-States"
                    },
                    "pred_salary_class": "<=50K"
                }
            ]
        }

# Set up logging
# Copied and adjusted from
# https://stackoverflow.com/questions/13733552/logger-configuration-to-log-to-file-and-print-to-stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        MakeFileHandler("logs/rest_api.log", ),
        logging.StreamHandler()
    ]
)

logging.info('Loading LabelBinarizer')
lb = joblib.load(
    os.path.join(
        params['prepare']['data_output'],
        'lb.joblib'
    )
)

logging.info('Loading one-hot-encoder')
ohe = joblib.load(
    os.path.join(
        params['prepare']['data_output'],
        'ohe.joblib'
    )
)

logging.info('Loading Linear Regression Classifier')
model = joblib.load(params['train']['model_output'])

@app.get("/", response_class=HTMLResponse)
async def greet_user():
    return """
    <html>
        <body>
            Hello dear user! Welcome to the model inference API.<br>

            Short explanation on the usage:<br>
            
            GET / will give you this message.<br>

            POST /inference will run inference
        </body>
    </html>
    """

@app.post("/inference/")
async def do_inference(samples: Batch) -> BatchResult:
    """
    This endpoint does batch inference
    """
    # Important to use the by_alias variable,
    # otherwise the hyphens are not used
    batch_df = pd.DataFrame(
        samples.dict(by_alias=True)['samples'],
        index=list(range(len(samples.samples)))
    )
    logging.debug('DataFrame head looks like this: %s', str(batch_df.head()))

    # Pre-processing
    logging.debug('Processing data')
    X_sample, _, _, _ = process_data(
        X=batch_df,
        categorical_features=params['data']['cat_features'],
        training=False,
        lb=lb,
        encoder=ohe
    )

    # Inference
    logging.debug('Running inference')
    y_preds = inference(model, X_sample)

    # Construct the response
    y_preds = lb.inverse_transform(y_preds)

    # Append the predictions so we can iterate by row
    batch_df['pred_salary_class'] = y_preds
    y_preds = batch_df.pop('pred_salary_class')

    # To dict to use with Pydantic
    input_list = batch_df.to_dict(orient='records')
    pred_list = y_preds.to_dict()

    # Construct the BatchResult list
    batch_result = []
    for inp, (_, pred) in zip(input_list, pred_list.items()):
        batch_result.append(InferenceResult(input=inp, pred_salary_class=pred))

    return BatchResult(results=batch_result)
