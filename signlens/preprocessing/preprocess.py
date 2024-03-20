import pandas as pd
import numpy as np
from colorama import Fore, Style
from signlens.params import *
from signlens.preprocessing.data import *


def pad_sequences(sequence,n_frames=100):
    '''
    Args:
        - NumPy Array: Sequence of landmarks
        - int: number of frames
    Returns:
        - Numpy Array: Padded or cut off sequence of landmarks
    '''
    if len(sequence) < n_frames:
        pad_width = int(n_frames - len(sequence))
        sequence = np.pad(sequence, ((0, pad_width), (0, 0),(0, 0)), mode='constant')
    else:
        # TO DO: check if sign is at beginning, middle or end
        sequence = sequence[:n_frames]
    return sequence

def xy_generator(train_frame,n_frames=100):
    '''
    Yields X and y for input to model.fit
    Use X, y = next(generator) to iterate through all Xs and ys

    Args:
        - DataFrame: train_frame
        - int: number of frames
    Returns:
        - generator: X and y ()
    '''
    for i in train_frame.index:
        X = load_relevant_data_subset(train_frame['file_path'][i])
        X = pad_sequences(X, n_frames)
        y = np.expand_dims(np.array(y),0)
        yield X, y
