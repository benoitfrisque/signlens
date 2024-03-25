import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import json

from signlens.preprocessing.preprocess import decode_labels, pad_and_preprocess_sequence, reshape_processed_data_to_tf
from signlens.preprocessing.data import load_landmarks_json
from utils.model_utils import load_model

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
model_name = "model 20240322-173411"
model, _ = load_model(model_name)

app.state.model = model


# Takes in json path
"""
@app.post("/predict")
async def predict(landmarks_json_path: str):

    model = app.state.model

    if not landmarks_json_path:
        raise HTTPException(status_code=400, detail="No file provided")


    landmarks = load_landmarks_json(landmarks_json_path)
    data_processed = pad_and_preprocess_sequence(landmarks)
    data_tf = reshape_processed_data_to_tf(data_processed)

    prediction = model.predict(data_tf)

    word, proba = decode_labels(prediction)

    word = str(word[0])
    proba = float(proba[0])

    return {'Word:': word, 'Probability:': proba}
"""

# Takes in a JSON file
@app.post("/predict")
async def upload_file(file: UploadFile = File(...)):

    model = app.state.model

    if not file:
        raise HTTPException(status_code=400, detail="No file provided")


    json_data = await file.read()
    json_text = json_data.decode('utf-8')
    json_object = json.loads(json_text)
    json_df = pd.DataFrame(json_object)

    landmarks = load_landmarks_json(json_df)
    data_processed = pad_and_preprocess_sequence(landmarks)
    data_tf = reshape_processed_data_to_tf(data_processed)

    prediction = model.predict(data_tf)

    word, proba = decode_labels(prediction)

    word = str(word[0])
    proba = float(proba[0])

    return {'Word:': word, 'Probability:': proba}

@app.get("/")
def root():

    return {"Welcome to FastAPI"}

# Run --> uvicorn signlens.api.fast:app --reload
# Test JSON file: /Users/max/code/benoitfrisque/signlens/processed_data/07070_landmarks.json
