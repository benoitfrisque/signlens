import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from signlens.preprocessing.registry import load_model
from signlens.preprocessing.preprocess import group_pad_sequences, decode_labels
from signlens.preprocessing.data import load_relevant_data_subset

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Load model
model_name = "model_epoch_08.keras"
app.state.model = load_model(model_name) # load the model with the above-imported function


# Takes in parquet path

# @app.post("/predict")
# async def predict(pq_path: str):

@app.post("/predict")
#async def predict(file: UploadFile = File(...)):
async def predict(pq_path: str):

    model = app.state.model

    # Check if file is empty

    # if not file:

    if not pq_path:
        raise HTTPException(status_code=400, detail="No file provided")

    # Preprocess the data

    # processed_data = load_relevant_data_subset(pq_path)
    # processed_data = load_relevant_data_subset("parquet/path")

    #landmarks = await file.read()
    # processed_data = load_relevant_data_subset(pq_path)

    # Convert to DF first
    processed_df = pd.DataFrame(pq_path, columns = ['file_path'])
    processed_data = group_pad_sequences(processed_df)


    # Perform prediction
    prediction = model.predict(np.array([processed_data]))

    word = decode_labels(prediction)

    return {"Predicted word": word}


@app.get("/")
def root():

    return {"API is working"}



# Run --> uvicorn signlens.api.fast:app --reload




























"""
# Load the data
from signlens.preprocessing import data
from .env import *
data = load_data_subset_csv(frac=DATA_FRAC, noface=True, balanced=False,
                     n_classes=NUM_CLASSES, n_frames=MAX_SEQ_LEN,
                     random_state=None, csv_path=TRAIN_TRAIN_CSV_PATH)

# Load the relevant data from the parquet file (load relevant subset function)
from signlens.preprocessing import preprocess
    # load_relevant_data_subset(pq_path, noface=True)
    # loads landmarks as numpy array

# Pad sequences
from signlens.preprocessing import preprocess
    # pad_sequences(sequence, n_frames=MAX_SEQ_LEN)


# Load function that transforms

# (load_relevant_data_subset_per_landmark_type)


# POST request
"""


"""
# Model.predict
@app.get("/predict")
def predict():  # needs to take landmarks or parquet files

    model = app.state.model
    assert model is not None

    model.predict
    # :warning: fastapi only accepts simple Python data types as a return value
    # among them dict, list, str, int, float, bool
    # in order to be able to convert the api response to JSON
    return
"""
