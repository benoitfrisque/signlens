import os
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider
from IPython.display import display

from signlens.params import *

def save_model(model):
    model_path = os.path.join(BASE_DIR, 'results', 'models', 'model.h5')
    model.save(model_path)

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
