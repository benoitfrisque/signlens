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

import os
import time

def create_folder_model():
    """
    Creates a model directory with a unique timestamp and several subdirectories.

    The function generates a timestamp, creates a main directory with the timestamp in its name,
    and then creates several subdirectories within the main directory. The paths to these directories
    are stored in a dictionary which is returned by the function.

    Returns:
        dict: A dictionary where the keys are the names of the subdirectories with '_path' appended,
              and the values are the corresponding paths as strings.
    """
    # Generate a timestamp for the folder name
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join(TRAIN_OUTPUT_DIR, f"model {timestamp}")

    # Create the model directory
    os.mkdir(model_path)

    # Initialize a dictionary to store the paths
    paths = {}

    # Define subdirectories
    subdirs = ['model', 'plot', 'log', 'metrics', 'params']

    # Create subdirectories and store their paths
    for subdir in subdirs:
        subdir_path = os.path.join(model_path, subdir)
        paths[f'{subdir}_path'] = subdir_path
        os.mkdir(subdir_path)

    # Create 'model each epoch' as a subdirectory of 'model'
    model_each_epoch_path = os.path.join(paths['model_path'], 'model each epoch')
    paths['model_each_epoch_path'] = model_each_epoch_path
    os.mkdir(model_each_epoch_path)

    return paths


def save_model(model,model_path):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join(model_path, f"model_{timestamp}.keras")
    model.save(model_path)

def save_results(params: dict, metrics: dict,params_path,metrics_path,mode="train") -> None:
    """
    Persist params & metrics locally on the hard drive at
    "{LOCAL_REGISTRY_PATH}/params/{current_timestamp}.pickle"
    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"
    - (unit 03 only) if MODEL_TARGET='mlflow', also persist them on MLflow
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save params locally
    if params is not None:
        if mode=="train":
            params_path = os.path.join(params_path, f"training_{timestamp}.pickle")
        else:
            params_path = os.path.join(params_path, f"evaluate_{timestamp}.pickle")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # Save metrics locally
    if metrics is not None:
        if mode=="train":
            metrics_path = os.path.join(metrics_path, f"training_{timestamp}.pickle")
        else:
            metrics_path = os.path.join(metrics_path, f"evaluate_{timestamp}.pickle")

        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    print("✅ Results saved locally")


def load_model(model_name_folder=None) -> keras.Model:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)

    Return None (but do not Raise) if no model is found

    """


    print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

    # Get the latest model version name by the timestamp on disk
    if model_name_folder is None or model_name_folder=="":
        print(Fore.RED +"Please put the name of the model folder" + Style.RESET_ALL)
        return None
    local_model_paths = glob.glob(os.path.join(TRAIN_OUTPUT_DIR, f"{model_name_folder}*"))

    if not local_model_paths:
        print(Fore.RED +f"No Folder named {model_name_folder} found" + Style.RESET_ALL)
        return None

    most_recent_model_path_on_disk = sorted(local_model_paths, key=os.path.getctime)[-1]

    print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)

    keras_files = glob.glob(os.path.join(most_recent_model_path_on_disk,'model', "*.keras"))[0]

    latest_model = keras.models.load_model(keras_files)

    print(f"✅ Model loaded from local disk {most_recent_model_path_on_disk}")

    return latest_model,most_recent_model_path_on_disk







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
