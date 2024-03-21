import pandas as pd
import numpy as np
from colorama import Fore, Style
from signlens.params import *
from signlens.preprocessing.data import *
# from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix

def pad_sequences(sequence, n_frames=MAX_SEQ_LEN):
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

def group_pad_sequences(df, n_frames=MAX_SEQ_LEN):
    """
    Group and pad sequences from DataFrame file paths.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing file paths.

    Returns:
    - numpy.ndarray: 4D array of grouped and padded sequences.

    This function takes a DataFrame `df` containing file paths and loads relevant data subsets
    using `load_relevant_data_subset` function for each file path. It then pads the sequences to
    ensure uniform length and shapes them into a 4D numpy array of shape (n, n_frames, N_LANDMARKS_NO_FACE, 3), where:
    - n is the number of file paths in the DataFrame.
    - n_frames is the number of frames.
    - N_LANDMARKS_NO_FACE is the number of landmarks.
    - 3 represents the number of positions(x,y and z)).

    """
    n=len(df)
    data_sparse=[]
    for i, file_path in enumerate(df):
        load_data=load_relevant_data_subset(file_path)
        padded_data = pad_sequences(load_data)
        data_sparse.append(csr_matrix(padded_data.reshape(-1)))
    return np.array([matrix.toarray().reshape(100, 75, 3) for matrix in data_sparse])

def label_dictionnary(df):
    """
    Encode the 'sign' column in the DataFrame using a label map dictionary.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing the 'sign' column to be encoded.
    Returns:
    - numpy.ndarray: One-hot encoded representations of the 'sign' column.

    This function loads converts the encoded values to one-hot encoded vectors using OnehotEncoder
    from sklearn.
    """
    encoder = OneHotEncoder()
    encoded_data = encoder.fit_transform(df[['sign']])
    y_encoded = encoded_data.toarray()
    return y_encoded

def xy_generator(train_frame, n_frames=MAX_SEQ_LEN):
    '''
    Yields X and y for input to model.fit
    Use example to iterate through all Xs and ys:
        xy = xy_generator(train_frame, 10)
        X,y = next(xy)
        print(X, y)
    Args:
        - DataFrame: train_frame
        - int: number of frames
    Returns:
        - generator: X and y
    '''
    for i in train_frame.index:
        X = load_relevant_data_subset(train_frame['file_path'][i])
        X = pad_sequences(X, n_frames)
        y = np.expand_dims(np.array(train_frame['sign'][i]),0)
        yield X, y

def x_generator(train_frame,n_frames=MAX_SEQ_LEN):
    '''
    Yields X for input to model.fit

    Args:
        - DataFrame: train_frame
        - int: number of frames
    Returns:
        - generator: X
    '''
    for i in train_frame.index:
        X = load_relevant_data_subset(train_frame['file_path'][i])
        X = pad_sequences(X, n_frames)
        yield X #, y

def y_generator(train_frame):
    '''
    Yields y for input to model.fit

    Args:
        - DataFrame: train_frame
        - int: number of frames
    Returns:
        - generator: X and y
    '''
    for i in train_frame.index:
        y = np.expand_dims(np.array(train_frame['sign'][i]),0)
        yield y

def batch_from_generator(gen, batch_size):
    '''
    Args:
        - generator: x_generator or y_generator
    Returns:
        - numpy.ndarray: batch
    '''
    # TO DO: check for len(train_frame) > batch_size
    batch = [ next(gen) for i in range(batch_size)]
    return np.array(batch)
