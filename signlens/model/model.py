from sklearn.model_selection import train_test_split
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, Masking, Flatten, Dropout, SimpleRNN, Reshape, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from typing import Tuple
from colorama import Fore, Style
import time
import numpy as np

from signlens.params import *


def initialize_model(n_frames=MAX_SEQ_LEN, n_landmarks=N_LANDMARKS_NO_FACE, num_classes=NUM_CLASSES):
    """
    Initializes a Sequential model with specific layers for sign recognition.

    Args:
        n_frames (int, optional): The number of frames to consider in the sequence. Defaults to MAX_SEQ_LEN.
        n_landmarks (int, optional): The number of landmarks to consider in each frame. Defaults to N_LANDMARKS_NO_FACE.
        num_classes (int, optional): The number of output classes. Defaults to NUM_CLASSES.

    Returns:
        Sequential: The initialized Keras Sequential model.
    """

    model = Sequential()

    model.add(Reshape((n_frames, n_landmarks * 3),
              input_shape=(n_frames, n_landmarks, 3)))
    model.add(Masking(mask_value=0.0))

    model.add(SimpleRNN(units=128, return_sequences=True))
    model.add(Dropout(0.5))

    model.add(LSTM(units=64))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))  # output layer

    return model


def compile_model(model: Model, learning_rate=0.001):
    """
    This function compiles a given Keras Model with Adam optimizer, categorical crossentropy as loss function and accuracy as metrics.

    Args:
        model (Model): The Keras Model to be compiled.
        learning_rate (float, optional): The learning rate for the Adam optimizer. Defaults to 0.001.

    Returns:
        Model: The compiled Keras Model.
    """
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(
    model: Model,
    X: np.ndarray,
    y: np.ndarray,
    batch_size=256,
    patience=2,
    epochs=10,
    validation_data=None,
    validation_split=0.3,
    verbose=1
) -> Tuple[Model, dict]:
    """
    Trains the given Keras Model using the provided training data and parameters, and returns the trained model along with its training history.

    Args:
        model (Model): The Keras Model to be trained.
        X (np.ndarray): The input data for training.
        y (np.ndarray): The target output data for training.
        batch_size (int, optional): The number of samples per gradient update. Defaults to 256.
        patience (int, optional): The number of epochs with no improvement after which training will be stopped. Defaults to 2.
        epochs (int, optional): The number of times the learning algorithm will work through the entire training dataset. Defaults to 10.
        validation_data (tuple, optional): Data on which to evaluate the loss and any model metrics at the end of each epoch. Defaults to None.
        validation_split (float, optional): The fraction of the training data to be used as validation data. Defaults to 0.3.
        verbose (int, optional): Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. Defaults to 1.

    Returns:
        Tuple[Model, dict]: The trained model and its training history.
    """

    if validation_data is None:  # prepare valdation data with stratification
        if validation_split is not None:  # case with no validation_data but validation_split
            validation_data = train_test_split(
                X, y, test_size=validation_split, stratify=y)
            print(
                Fore.YELLOW + f"\nWarning: No validation_data provided, using validation_split={validation_split}" + Style.RESET_ALL)

        else:  # case with no validation_data and no validation_split
            print(
                Fore.RED + f"\nError: No validation_data or validation_split provided" + Style.RESET_ALL)
            return None

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    output_folder = create_output_folder()

    checkpoint = ModelCheckpoint(
        output_folder + os.path.sep + "model_epoch_{epoch:02d}.keras",
        monitor="val_accuracy",
        verbose=0,
        save_freq=(1 * int(X.shape[0] / batch_size)) + 1
    )
    LRreducer = ReduceLROnPlateau(
        monitor="val_accuracy", factor=0.1, patience=5, verbose=1, min_lr=1e-6)

    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    history = model.fit(
        X,
        y,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es, checkpoint, LRreducer],
        shuffle=True,
        verbose=verbose
    )

    print(f"✅ Model trained on {len(X)} rows")
    print(history)

    return model, history


def create_output_folder():
    # Generate a unique folder name using current timestamp
    timestamp = int(time.time())  # Get current timestamp
    folder_name = f"model_fit_at_{timestamp}"  # Generate folder name
    # Create a new directory with the generated name
    folder_path = MODEL_DIR + os.path.sep + folder_name
    os.mkdir(folder_path)

    return folder_path


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

    print(Fore.BLUE +
          f"\nEvaluating model on {len(X)} rows..." + Style.RESET_ALL)

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

    accuracy = metrics["accuracy"]

    print(f"✅ Model evaluated, accuracy: {round(accuracy, 2)}")

    return metrics
