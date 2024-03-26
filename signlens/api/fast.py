import os
import json
import pandas as pd
from fastapi import FastAPI, HTTPException, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware

from signlens.preprocessing.preprocess import decode_labels, pad_and_preprocess_sequence, reshape_processed_data_to_tf
from signlens.preprocessing.data import load_landmarks_json
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
model = load_model(mode='from_path', model_path=model_path)

app.state.model = model

# Takes in a JSON file
@app.post("/predict_file")
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

    pred, proba = decode_labels(prediction)

    pred = str(pred[0])
    proba = float(proba[0])

    return {'sign:': pred, 'probability:': proba}

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()

    model = app.state.model

    if not data:
        raise HTTPException(status_code=400, detail="No data provided")

    json_df = pd.DataFrame(data)

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

    return {"Welcome to SignLens"}

# Run --> uvicorn signlens.api.fast:app --reload
# Test JSON file: /Users/max/code/benoitfrisque/signlens/processed_data/07070_landmarks.json
