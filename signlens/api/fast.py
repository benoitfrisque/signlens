import os
import json
from fastapi import FastAPI, HTTPException, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware

from signlens.preprocessing.preprocess import preprocess_data_from_json_data, decode_labels,normalize_data_tf
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
model_file = "model_v4_250signs_filtered_pose2532.keras"
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
model_path = os.path.join(root_dir, 'models_api', model_file)

# Check if the model file exists
if os.path.exists(model_path):
    app.state.model = load_model(mode='from_path', model_path=model_path)
else:
    raise FileNotFoundError(f"Model file '{model_path}' does not exist.")

@app.post("/predict_file")
async def upload_file(file: UploadFile = File(...)):

    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    # Read json data from file
    json_file = await file.read()
    json_text = json_file.decode('utf-8')
    json_data = json.loads(json_text)

    # Preprocess data for model prediction
    data_processed_tf = preprocess_data_from_json_data(json_data)

    # Predict with loaded model
    prediction = app.state.model.predict(data_processed_tf)

    pred, proba = decode_labels(prediction)

    pred = str(pred[0])
    proba = float(proba[0])

    return {'sign': pred, 'probability': proba}


@app.post("/predict")
async def predict(request: Request):
    json_data = await request.json()

    if not json_data:
        raise HTTPException(status_code=400, detail="No data provided")

    # Preprocess data for model prediction
    data_processed_tf = preprocess_data_from_json_data(json_data)
    data_processed_tf_normalized = normalize_data_tf(data_processed_tf)

    # Predict with loaded model
    prediction = app.state.model.predict(data_processed_tf_normalized)

    pred, proba = decode_labels(prediction)

    pred = str(pred[0])
    proba = float(proba[0])

    return {'sign': pred, 'probability': proba}


@app.get("/")
def root():

    return {"Welcome to SignLens"}
