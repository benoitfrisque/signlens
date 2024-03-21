import pandas as pd

#Find a way to load_model --> from (folder_name.file_name) import (function_name or *)
from signlens.preprocessing.registry import load_model
# Find a way to preprocess features --> from (folder_name.file_name) import (function_name or *)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

model_name = "model_epoch_08.keras"
app.state.model = load_model(model_name) # load the model with the above-imported function

# Preload the model to accelerate the predictions
# We want to avoid loading the heavy Deep Learning model from MLflow at each `get("/predict")`
# The trick is to load the model in memory when the Uvicorn server starts
# and then store the model in an `app.state.model` global variable, accessible across all routes!
# This will prove very useful for the Demo Day

@app.get("/predict")
def predict():  # needs to take landmarks or parquet files

    model = app.state.model
    assert model is not None

    # ⚠️ fastapi only accepts simple Python data types as a return value
    # among them dict, list, str, int, float, bool
    # in order to be able to convert the api response to JSON
    return dict(fare=7.456)

@app.get("/")
def root():

    return dict(greeting="Hello")
