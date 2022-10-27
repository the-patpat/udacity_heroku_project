"""
Small script to test the live api on heroku
Author: Patrick
Date: Oct 2022
"""

# %% Imports
import requests
from main import Batch

# %%
response = requests.post(
    'https://udacity-inference-test.herokuapp.com/inference/',
    json=Batch.Config.schema_extra['example'])

print(f"Status code of request is {response.status_code}")
print(f"Response of the request is:\n{response.json()}")
