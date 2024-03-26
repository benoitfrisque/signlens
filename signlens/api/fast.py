import os
import json
import pandas as pd
from fastapi import FastAPI, HTTPException, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware

from signlens.preprocessing.preprocess import decode_labels, load_relevant_data_subset_from_pq, pad_and_preprocess_landmarks_array, reshape_processed_data_to_tf
from signlens.preprocessing.data import convert_landmarks_json_data_to_df
from signlens.model.model_utils import load_model

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
model_file = "model_v2_250signs.keras"
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
model_path = os.path.join(root_dir, 'models_api', model_file)
app.state.model = load_model(mode='from_path', model_path=model_path)

# Takes in a JSON file


@app.post("/predict_file")
async def upload_file(file: UploadFile = File(...)):

    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    json = await file.read()
    json_text = json_data.decode('utf-8')
    json_data = json.loads(json_text)

    landmarks_df = convert_landmarks_json_data_to_df(json_data)
    landmarks_array = load_relevant_data_subset_from_l(landmarks_df)
    data_processed = pad_and_preprocess_landmarks_array(landmarks)
    data_tf = reshape_processed_data_to_tf(data_processed)

    prediction = app.state.model.predict(data_tf)

    pred, proba = decode_labels(prediction)

    pred = str(pred[0])
    proba = float(proba[0])

    return {'sign:': pred, 'probability:': proba}


@app.post("/predict")
async def predict(request: Request):
    json_data = await request.json()

    if not json_data:
        raise HTTPException(status_code=400, detail="No data provided")

    landmarks_df = convert_landmarks_json_data_to_df(json_data)

    data_processed = pad_and_preprocess_landmarks_array(landmarks)
    data_tf = reshape_processed_data_to_tf(data_processed)

    prediction = app.state.model.predict(data_tf)

    word, proba = decode_labels(prediction)

    word = str(word[0])
    proba = float(proba[0])

    return {'Word:': word, 'Probability:': proba}


@app.get("/")
def root():

    return {"Welcome to SignLens"}
