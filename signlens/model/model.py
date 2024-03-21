import tensorflow as tf
from tensorflow.keras import Model,Sequential
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense,Masking, Flatten, Dropout, SimpleRNN,Reshape, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from typing import Tuple
from colorama import Fore, Style
import time
import numpy as np
from signlens.params import *
import matplotlib.pyplot as plt



def initialize_model(frame=100,num_classes=250):

    model = Sequential()

    model.add(Reshape((frame,N_LANDMARKS_NO_FACE*3),input_shape=(frame,N_LANDMARKS_NO_FACE,3)))
    model.add(Masking(mask_value=0.0))


    model.add(SimpleRNN(units=128, return_sequences=True))
    model.add(Dropout(0.5))

    model.add(LSTM(units=64))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))
    return model

def compile_model(model: Model,learning_rate=0.001):

    optimizer=Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=256,
        patience=2,
        validation_data=None,
        validation_split=0.3,
        verbose=0
    ) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)
    # Generate a unique folder name using current timestamp
    timestamp = int(time.time())  # Get current timestamp
    folder_name = f"model_fit_at_{timestamp}"  # Generate folder name
    # Create a new directory with the generated name
    os.mkdir(MODEL_DIR + os.path.sep+folder_name)
    # Return the name of the created folder
    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    modelCheckpoint = ModelCheckpoint(
    MODEL_DIR + os.path.sep+ folder_name + os.path.sep+"model_epoch_{epoch:02d}.keras",
    monitor="val_accuracy",
    verbose=0,
    save_freq=1*int(X.shape[0]/batch_size)+1
    )
    LRreducer = ReduceLROnPlateau(monitor="val_accuracy", factor = 0.1, patience=5, verbose=1, min_lr=1e-6) #

    history = model.fit(
        X,
        y,
        validation_data=validation_data,
        validation_split=validation_split,
        epochs=10,
        batch_size=batch_size,
        callbacks=[es,modelCheckpoint,LRreducer],
        shuffle=True,
        verbose=verbose
    )

    print(f"✅ Model trained on {len(X)} rows")
    print(history)

    return model, history

def evaluate_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=64,
        verbose=0
    ) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on the dataset
    """

    print(Fore.BLUE + f"\nEvaluating model on {len(X)} rows..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(
        x=X,
        y=y,
        batch_size=batch_size,
        verbose=verbose,
        # callbacks=None,
        return_dict=True
    )

    loss = metrics["loss"]


    print(f"✅ Model evaluated, loss: {round(loss, 2)}")

    return metrics
