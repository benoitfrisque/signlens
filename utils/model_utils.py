import os
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider
from IPython.display import display
from colorama import Fore, Style
from tensorflow import keras
from signlens.params import *
import glob
import time
import pickle

def save_model(model):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join(MODEL_DIR, f"{timestamp}.h5")
    model.save(model_path)

def save_results(params: dict, metrics: dict) -> None:
    """
    Persist params & metrics locally on the hard drive at
    "{LOCAL_REGISTRY_PATH}/params/{current_timestamp}.pickle"
    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"
    - (unit 03 only) if MODEL_TARGET='mlflow', also persist them on MLflow
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save params locally
    if params is not None:
        params_path = os.path.join(TRAIN_OUTPUT_DIR, "params", timestamp + ".pickle")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # Save metrics locally
    if metrics is not None:
        metrics_path = os.path.join(TRAIN_OUTPUT_DIR, "metrics", timestamp + ".pickle")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    print("✅ Results saved locally")


def load_model() -> keras.Model:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)

    Return None (but do not Raise) if no model is found

    """


    print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

    # Get the latest model version name by the timestamp on disk
    local_model_directory = os.path.join(MODEL_DIR)
    local_model_paths = glob.glob(f"{MODEL_DIR}/*.h5")

    if not local_model_paths:
        return None

    most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

    print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)

    latest_model = keras.models.load_model(most_recent_model_path_on_disk)

    print(f"✅ Model loaded from local disk {most_recent_model_path_on_disk}")

    return latest_model







def plot_history(history, metric='accuracy', title=None):
    """
    Plot training history of a neural network model.

    Parameters:
    - history (keras.callbacks.History): History object returned by the fit method of a Keras model.
    - metric (str): Metric to plot, default is 'accuracy'. It should be one of the metrics monitored during training.
    - title (str): Optional title for the plot.

    Returns:
    - None: This function plots the training history but does not return any value.
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13,4))

    # Create the plots
    ax1.plot(history.history['loss'])
    ax1.plot(history.history['val_loss'])

    ax2.plot(history.history[metric])
    ax2.plot(history.history['val_' + metric])

    # Set titles and labels
    ax1.set_title('Model loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')

    ax2.set_title(metric.capitalize())
    ax2.set_ylabel(metric.capitalize())
    ax2.set_xlabel('Epoch')

    # Generate legends
    ax1.legend(['Train', 'Validation'], loc='best')
    ax2.legend(['Train', 'Validation'], loc='best')

    # Show grids
    ax1.grid(axis="x",linewidth=0.5)
    ax1.grid(axis="y",linewidth=0.5)

    ax2.grid(axis="x",linewidth=0.5)
    ax2.grid(axis="y",linewidth=0.5)

    if title:
        fig.suptitle(title)

    plt.show()


def plot_history_interactive(history, metric='accuracy', title=None):
    """
    Plot training history of a neural network model with interactive axis limits.

    Parameters:
    - history (keras.callbacks.History): History object returned by the fit method of a Keras model.
    - metric (str): Metric to plot, default is 'accuracy'. It should be one of the metrics monitored during training.
    - title (str): Optional title for the plot.

    Returns:
    - Plots with interactive widgets.
    """

    def plot_hist(y_min, y_max, epoch_min, epoch_max):
        # Setting figures
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(13,4))

        # Create the plots
        ax1.plot(history.history['loss'])
        ax1.plot(history.history['val_loss'])

        ax2.plot(history.history[metric])
        ax2.plot(history.history['val_' + metric])

        # Set titles and labels
        ax1.set_title('Model loss')
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Epoch')

        ax2.set_title(metric.capitalize())
        ax2.set_ylabel(metric.capitalize())
        ax2.set_xlabel('Epoch')

        # Generate legends
        ax1.legend(['Train', 'Validation'], loc='best')
        ax2.legend(['Train', 'Validation'], loc='best')


        ax1.set_xlim(epoch_min, epoch_max)
        ax2.set_xlim(epoch_min, epoch_max)

        ax1.set_ylim(y_min, y_max)
        ax2.set_ylim(y_min, y_max)

        # Show grids
        ax1.grid(axis="x",linewidth=0.5)
        ax1.grid(axis="y",linewidth=0.5)

        ax2.grid(axis="x",linewidth=0.5)
        ax2.grid(axis="y",linewidth=0.5)

        if title:
            fig.suptitle(title)


    y_min_slider = FloatSlider(
        value=0,
        min=0,
        max=1,
        step=0.1,
        description='Y Min:',
        continuous_update=False
    )

    y_max_slider = FloatSlider(
        value=1,
        min=0,
        max=1,
        step=0.05,
        description='Y Max:',
        continuous_update=False
    )

    epoch_min_slider = FloatSlider(
        value=0,
        min=0,
        max=len(history.history['loss'])-2,
        step=1,
        description='Min epoch:',
        continuous_update=False
    )

    epoch_max_slider = FloatSlider(
        value=len(history.history['loss'])-1,
        min=1,
        max=len(history.history['loss'])-1,
        step=1,
        description='Max epoch:',
        continuous_update=False
    )

    return interact(plot_hist, y_min=y_min_slider, y_max=y_max_slider, epoch_min=epoch_min_slider, epoch_max=epoch_max_slider)
