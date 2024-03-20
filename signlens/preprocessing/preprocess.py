import pandas as pd
import numpy as np
from colorama import Fore, Style
from signlens.params import *


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
