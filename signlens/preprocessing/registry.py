import os
import pickle
from tensorflow.keras import models

def load_model(model_name):

    parent_directory = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    model_path = os.path.join(parent_directory, 'training_outputs', 'models', model_name)  # Path to your model file

    print (model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_name} in {model_path} not found.")

    model = models.load_model(model_path)

    return model
