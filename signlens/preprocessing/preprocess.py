from json import load
import numpy as np
import multiprocessing as mp
import tensorflow as tf
from colorama import Fore, Style
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from signlens.params import *
from signlens.preprocessing.data import load_relevant_data_subset, load_glossary

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


def load_and_pad(pq_file_path):
    """
    Load data from a parquet file and pad the sequence.

    Parameters:
    - pq_file_path (str): Path to the parquet file.

    Returns:
    - numpy.ndarray: Padded data reshaped into a 1D array.
    """
    # Load data from the file
    load_data = load_relevant_data_subset(pq_file_path)

    # Pad the sequence
    padded_data = pad_sequences(load_data)

    # Reshape the data into a 1D array and return it
    return padded_data.reshape(-1)

def group_pad_sequences(pq_file_path_df, n_frames=MAX_SEQ_LEN):
    """
    Load data from multiple files, pad the sequences, and group them into a single array.
    If an error occurs during multiprocessing, falls back to sequential processing.

    Parameters:
    - pq_file_path_df (pandas.DataFrame): DataFrame containing file paths.
    - n_frames (int, optional): Number of frames. Defaults to MAX_SEQ_LEN.

    Returns:
    - tf.Tensor: 3D tensor of grouped and padded sequences.

    Raises:
    - Exception: If an error occurs during the loading or padding process.
    """

    try:
        # Create a pool of worker processes
        with mp.Pool(mp.cpu_count()) as pool:
            # Use the pool to apply `load_and_pad` to each file path in `df` in parallel
            data = pool.map(load_and_pad, pq_file_path_df)
    except Exception as e:
        print(f"An error occurred with multiprocessing: {e}")
        print("Falling back to sequential processing...")

        # Fallback to sequential processing
        data = [load_and_pad(file_path) for file_path in pq_file_path_df]

    # Reshape the data into a 3D array and return it
    data_reshaped = np.array([item.reshape(n_frames, N_LANDMARKS_NO_FACE * 3) for item in data])
    data_tf = tf.convert_to_tensor(data_reshaped)

    return data_tf

from sklearn.preprocessing import LabelEncoder


def encode_labels(y):
    """
    Encode the labels in y based on a provided glossary using TensorFlow's to_categorical.

    Parameters:
    - y (pandas.Series): Series containing labels to be encoded.

    Returns:
    - numpy.ndarray: Encoded representations of the labels.
    """
    glossary = load_glossary()

    # Extract labels from the glossary
    labels = glossary['sign'].tolist()

    # Get unique labels and their indices
    label_indices = {label: index for index, label in enumerate(labels)}

    # Encode the labels
    encoded_labels = y.map(label_indices)

    # Convert labels to one-hot encoding using TensorFlow's to_categorical
    encoded_labels = tf.keras.utils.to_categorical(encoded_labels, num_classes=NUM_CLASSES)

    return encoded_labels

def decode_labels(y_encoded):
    """
    Decode encoded labels based on the provided glossary.

    Parameters:
    - y_encoded (numpy.ndarray): Encoded representations of the labels.

    Returns:
    - pandas.Series: Decoded labels.
    """

    glossary = load_glossary()

    # Extract labels from the glossary
    labels = glossary['sign'].tolist()

    # Get the index with maximum value for each row in y_encoded
    decoded_indices = np.argmax(y_encoded, axis=1)

    # Map indices back to labels
    decoded_labels = [labels[idx] for idx in decoded_indices]

    predict_proba = np.max(y_encoded, axis=1)

    return decoded_labels, predict_proba
