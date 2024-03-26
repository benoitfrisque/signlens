import os
import glob
import time
import pickle

import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider
from colorama import Fore, Style
from tensorflow.keras import models

from signlens.params import *


def create_model_folder(training_output_dir=TRAIN_OUTPUT_DIR):
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
    model_base_dir_name = f"model_{timestamp}"

    paths = get_model_subdir_paths(model_base_dir_name, training_output_dir=training_output_dir)

    for path in paths.values():
        os.makedirs(path)

    return paths

def get_model_subdir_paths(model_base_dir_name, training_output_dir=TRAIN_OUTPUT_DIR):
    # Construct the path to the model directory
    model_base_dir = os.path.join(training_output_dir, model_base_dir_name)

    # Define subdirectories
    subdirs = ['model', 'plots', 'log', 'metrics', 'params']

    # Initialize a dictionary to store the paths
    paths = {}

    # Get the paths to the subdirectories
    for subdir in subdirs:
        subdir_path = os.path.join(model_base_dir, subdir)
        paths[subdir] = subdir_path

    sub_subdir_path = os.path.join(model_base_dir, 'model', 'iter')
    paths['iter'] = sub_subdir_path

    return paths

def save_results(params, metrics, params_path, metrics_path, mode) :
    """
    Save the parameters and metrics locally.

    Args:
        params (object): The parameters to be saved.
        metrics (object): The metrics to be saved.
        params_path (str): The path to save the parameters.
        metrics_path (str): The path to save the metrics.
        mode (str): The mode of operation (train, eval).
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save params locally
    if params is not None:
        if mode == "train":
            params_path = os.path.join(params_path, f"training_{timestamp}.pickle")
        else:
            params_path = os.path.join(params_path, f"evaluate_{timestamp}.pickle")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # Save metrics locally
    if metrics is not None:
        if mode == "train":
            metrics_path = os.path.join(metrics_path, f"training_{timestamp}.pickle")
        else:
            metrics_path = os.path.join(metrics_path, f"evaluate_{timestamp}.pickle")

        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    print("✅ Results saved locally")


def save_model(model, model_path):
    """
    Save the given model to the specified model_path.

    Args:
        model (keras.Model): The model to be saved.
        model_path (str): The path where the model should be saved.

    Returns:
        None
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join(model_path, f"model_{timestamp}.keras")
    model.save(model_path)


def load_model(mode='most_recent', model_base_dir_pattern=None, model_path=None, training_output_dir=TRAIN_OUTPUT_DIR, return_paths=False):
    """
    Load a model based on the specified mode.

    Args:
        mode (str, optional): The mode for loading the model. Defaults to 'most_recent'. Accepted values are 'most_recent' and 'from_path'.
        model_base_dir_pattern (str, optional): The pattern for the base directory of the model. Defaults to None.
        model_path (str, optional): The path to the model file. Mandatory when mode is 'from_path'. Defaults to None.
        training_output_dir (str, optional): The directory where the training output is stored. Defaults to TRAIN_OUTPUT_DIR.
        return_paths (bool, optional): Whether to return the paths of the loaded model. Defaults to False.

    Returns:
        model: The loaded model.
        model_paths (list): The paths of the loaded model (only returned if return_paths is True).

    Raises:
        ValueError: If the mode is not recognized.

    """

    if mode == 'most_recent':
        if model_base_dir_pattern is None:
            print(Fore.YELLOW + "No model name provided. Loading most recent model." + Style.RESET_ALL)

        return load_most_recent_model(model_base_dir_pattern, training_output_dir, return_paths=return_paths)

    elif mode == 'from_path':
        if model_base_dir_pattern is not None:
            print(Fore.YELLOW + f"The pattern {model_base_dir_pattern} provided with 'from_path' option will be ignored" + Style.RESET_ALL)

        if model_path is None:
            print(Fore.RED + "No model path provided. Mandatory with 'from_path' option" + Style.RESET_ALL)
            if return_paths:
                return None, None
            return None

        if return_paths:
            raise NotImplementedError("Returning model paths is not implemented for 'from_path' mode.")

        if not os.path.exists(model_path):
            print(Fore.RED + f"Model path {model_path} does not exist. Please check the path." + Style.RESET_ALL)
            if return_paths:
                return None, None
            return None

        model = models.load_model(model_path)
        return model

    else:
        raise ValueError('Mode not recognized. Please use either "most_recent" or "from_path"')

def load_most_recent_model(model_base_dir_pattern=None, training_output_dir=TRAIN_OUTPUT_DIR, return_paths=False):
    """
    Load the most recent model from the specified model base directory.

    Args:
        model_base_dir_pattern (str, optional): Pattern to match the model base directory. Defaults to None.
        training_output_dir (str, optional): Directory where the training output is stored. Defaults to TRAIN_OUTPUT_DIR.
        return_paths (bool, optional): Whether to return the paths of all models in the model base directory. Defaults to False.

    Returns:
        most_recent_model (tensorflow.keras.Model): The most recent model loaded from the model base directory.
        model_paths (list): List of paths to all models in the model base directory (if return_paths is True).
    """

    model_base_dir = get_most_recent_model_base_directory(model_base_dir_pattern, training_output_dir)

    if model_base_dir is None:
        print(Fore.RED + f"No model matching {model_base_dir_pattern}. Please check the model directory." + Style.RESET_ALL)
        if return_paths:
            return None, None
        return None

    most_recent_model_path = get_most_recent_model_path(model_base_dir)

    if most_recent_model_path is None:
        print(Fore.RED + f"No model found in {model_base_dir}. Please check the model directory." + Style.RESET_ALL)
        if return_paths:
            return None, None
        return None

    print(Fore.GREEN + f"Loading model from: '{most_recent_model_path}'"+ Style.RESET_ALL)

    most_recent_model = models.load_model(most_recent_model_path)

    if return_paths:
        model_paths = get_model_subdir_paths(model_base_dir, training_output_dir=training_output_dir)
        return most_recent_model, model_paths

    return most_recent_model


def get_most_recent_model_base_directory(model_base_dir_pattern=None, training_output_dir=TRAIN_OUTPUT_DIR):
    """
    Get the most recent model base directory that matches the given pattern.

    Args:
        model_base_dir_pattern (str, optional): Pattern to match the model base directories. If None, all directories will be considered.
        training_output_dir (str, optional): Directory where the model base directories are located.

    Returns:
        str: The path of the most recent model base directory that matches the given pattern, or None if no matching directory is found.
    """

    # If model_pattern is None, look for all directories
    if model_base_dir_pattern is None:
        model_base_dir_pattern = "*"

    # Get a list of all directories that contain the model_pattern
    model_directories = glob.glob(os.path.join(training_output_dir, f"*{model_base_dir_pattern}*"))

    # Check if there are any matching directories
    if not model_directories:
        return None

    # Get the most recent directory
    most_recent_directory = max(model_directories, key=os.path.getctime)

    return most_recent_directory

def get_most_recent_model_path(model_base_dir, model_format='.keras'):
    """
    Get the path to the most recent model file in the specified directory.

    Args:
        model_base_dir (str): The base directory where the model files are stored.
        model_format (str, optional): The format of the model files. Defaults to '.keras'.

    Returns:
        str: The path to the most recent model file, or None if no matching files are found.
    """

    model_dir = os.path.join(model_base_dir, 'model')
    model_files = glob.glob(os.path.join(model_dir, f"*{model_format}"))

    # Check if there are any matching files
    if not model_files:
        return None

    # Get the most recent file
    most_recent_model_file = max(model_files, key=os.path.getctime)

    # Return the path to the most recent model file
    return most_recent_model_file



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
