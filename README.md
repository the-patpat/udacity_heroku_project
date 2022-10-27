# Heroku Live API for Model Inference
This repository contains code for the **Deploying a Machine Learning Model on 
Heroku with FastAPI** project for the Udacity ML DevOps Engineer course.

## Repository contents
This repository contains the necessary code to deploy a model inference
application on Heroku, which users can utilize to perform inference on a model
trained for predicting the salary class of people using various personal 
attributes as features.

The **model card** can be found under `model_card.md`.

**Parameters** are listed under `params.yaml`

The actual API code is listed in `main.py`

## Submission requirements
* The screenshots are contained in the `screenshots` folder. The data files and
model files are not contained within this repository but instead are tracked
with DVC and saved in my personal GDrive.
* The performance on data slices are listed in `slice_output.txt`
* I made some **deviations** from the requirements:
    - The test cases for the different inference results are contained in one case
      for me as I designed my API endpoint to be able to handle batches of data
      
      I simply pass the training batch and then check that the resulting **set**
      contains only the two known classes.