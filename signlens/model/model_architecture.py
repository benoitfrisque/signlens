from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, Masking, Flatten, Dropout, SimpleRNN, Reshape, Bidirectional, Input, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from colorama import Fore, Style

from signlens.params import *


def initialize_model(n_frames=MAX_SEQ_LEN, n_landmarks=N_LANDMARKS_NO_FACE-N_LANDMARKS_POSE_TO_TAKE_OFF, num_classes=NUM_CLASSES):
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

    model.add(Input(shape=(n_frames, n_landmarks * N_DIMENSIONS_FOR_MODEL)))

    model.add(Masking(mask_value=MASK_VALUE))
    model.add(LSTM(units=256, return_sequences=True))
    model.add(Dropout(0.2))



    model.add(LSTM(units=256, return_sequences=True))
    model.add(Dropout(0.2))


    model.add(LSTM(units=256))
    model.add(Dropout(0.2))


    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))  # output layer

    return model


def compile_model(model, learning_rate=0.001):
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


def train_model(model, X, y,
                model_save_epoch_path,
                batch_size=256,
                patience=10,
                epochs=100,
                validation_data=None,
                validation_split=0.3,
                verbose=1,
                shuffle=True):
    """
    Trains the given Keras Model using the provided training data and parameters,
    and returns the trained model along with its training history.

    Args:
        model (Model): The Keras Model to be trained.
        X (np.ndarray): The input data for training.
        y (np.ndarray): The target output data for training.
        batch_size (int, optional): The number of samples per gradient update. Defaults to 256.
        patience (int, optional): The number of epochs with no improvement after which training will be stopped. Defaults to 2.
        epochs (int, optional): The number of times the learning algorithm will work through the entire training dataset. Defaults to 10.
        validation_data (tuple, optional): Data on which to evaluate the loss and any model metrics at the end of each epoch. Defaults to None.
        validation_split (float, optional): The fraction of the training data to be used as validation data. Defaults to 0.3. Not used if validation data is provided.
        verbose (int, optional): Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. Defaults to 1.
        shuffle (bool, optional): Whether to shuffle the training data before each epoch. Defaults to True.

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
        monitor="val_accuracy",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        model_save_epoch_path + os.path.sep + "model_epoch_{epoch:02d}.keras",
        monitor="val_accuracy",
        verbose=0,
        save_freq=(10 * int(X.shape[0] / batch_size)) + 1
    )
    LRreducer = ReduceLROnPlateau(
        monitor="val_accuracy", factor=0.5, patience=patience-1, verbose=1, min_lr=1e-6)

    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    history = model.fit(
        X,
        y,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es, checkpoint, LRreducer],
        shuffle=shuffle,
        verbose=verbose
    )

    print(f"✅ Model trained on {len(X)} rows")

    return model, history


def evaluate_model(model, X, y, batch_size=64, verbose=0):
    """
    Evaluate trained model performance on the dataset

    Args:
        model (tf.keras.Model): The trained model to evaluate.
        X (numpy.ndarray): The input data.
        y (numpy.ndarray): The target labels.
        batch_size (int, optional): The batch size for evaluation. Defaults to 64.
        verbose (int, optional): Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. Defaults to 0.

    Returns:
        dict: A dictionary containing the evaluation metrics.
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
        return_dict=True
    )

    accuracy = metrics["accuracy"]

    print(f"✅ Model evaluated, accuracy: {accuracy:.1%}")

    return metrics
