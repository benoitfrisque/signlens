import tensorflow as tf
from tensorflow.keras import Model,Sequential
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense,Masking, Flatten, Dropout, SimpleRNN,Reshape
from tensorflow.keras.callbacks import EarlyStopping
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

def compile_model(model: Model):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
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

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X,
        y,
        validation_data=validation_data,
        validation_split=validation_split,
        epochs=100,
        batch_size=batch_size,
        callbacks=[es],
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

def plot_history(history):
    # Accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
