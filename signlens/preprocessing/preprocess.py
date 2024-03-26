import numpy as np
import multiprocessing as mp
import tensorflow as tf
from tqdm import tqdm
from colorama import Fore, Style
import pandas as pd

from signlens.params import *
from signlens.preprocessing.data import load_relevant_data_subset, load_glossary


def pad_and_preprocess_sequence(sequence, n_frames=MAX_SEQ_LEN):
    '''
    Args:
        - NumPy Array: Sequence of landmarks
        - int: number of frames
    Returns:
        - Numpy Array: Padded or cut off sequence of landmarks
    '''
    # Replace nan values with MASK_VALUE
    sequence[np.isnan(sequence)] = MASK_VALUE

    # Pad with MASK_VALUE or cut off the sequence
    if len(sequence) < n_frames:
        pad_width = int(n_frames - len(sequence))
        sequence = np.pad(sequence, pad_width=((0, pad_width), (0, 0), (0, 0)), mode='constant', constant_values=MASK_VALUE)
    else:
        # TO DO: check if sign is at beginning, middle or end
        sequence = sequence[:n_frames]

    return sequence


def load_pad_preprocess_pq(pq_file_path, n_frames=MAX_SEQ_LEN):
    """
    Load data from a parquet file, pad and preprocess the sequence.

    Parameters:
    - pq_file_path (str): Path to the parquet file.

    Returns:
    - numpy.ndarray: Padded data reshaped into a 1D array.
    """
    # Load data from the file
    load_data = load_relevant_data_subset(pq_file_path)

    # Pad the sequence
    data_processed = pad_and_preprocess_sequence(load_data, n_frames=n_frames)

    # Reshape the data into a 1D array and return it
    return data_processed.reshape(-1)

def preprocess_and_pad_sequences_from_pq_list(pq_file_path_df, n_frames=MAX_SEQ_LEN):
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
            # Wrap the iterable with tqdm to add a progress bar
            data_processed = list(tqdm(pool.imap(load_pad_preprocess_pq, pq_file_path_df), total=len(pq_file_path_df)))


    except Exception as e:
        print(f"An error occurred with multiprocessing: {e}")
        print("Falling back to sequential processing...")

        # Fallback to sequential processing
        data_processed = [load_pad_preprocess_pq(pq_file_path, n_frames=n_frames) for pq_file_path in pq_file_path_df]

    data_processed = np.array(data_processed)
    data_tf = reshape_processed_data_to_tf(data_processed)
    return data_tf

def reshape_processed_data_to_tf(data_processed):
    """
    Reshape the processed data (3 or 4 DIMS) into a 3D tensor with 3 DIMS and convert it to a TensorFlow tensor.
    3 DIMS if batch_size=1
    for all other cases, 4 DIMS needed

    Parameters:
    - data (numpy.ndarray): Data to be reshaped.

    Returns:
    - tf.Tensor: Reshaped data as a TensorFlow tensor.
    """
    # case where we provide a single input
    if data_processed.ndim == 3:
        data_processed = np.expand_dims(data_processed, axis=0) # expand dim batch_size

    data_reshaped = np.array([item.reshape(MAX_SEQ_LEN, N_LANDMARKS_NO_FACE * 3) for item in data_processed])
    data_tf = tf.convert_to_tensor(data_reshaped)

    return data_tf


def encode_labels(y, num_classes=NUM_CLASSES):
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
    encoded_labels = tf.keras.utils.to_categorical(encoded_labels, num_classes=num_classes)

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


def filter_out_landmarks(parquet_file_path, landmark_types_to_remove, data_columns=None):
    """
    Filters out specific landmark types from a parquet file.

    Args:
        parquet_file_path (str or Path): Path to the input parquet file.
        landmark_types_to_remove (list of str): List of landmark types to be removed.
        data_columns (list of str, optional)

    Returns:
        DataFrame: DataFrame containing filtered landmarks.
    """

    if data_columns is None:
        landmarks = pd.read_parquet(parquet_file_path)
    else:
        landmarks = pd.read_parquet(parquet_file_path, columns=data_columns)

    filtered_landmarks = landmarks.copy()
    if landmark_types_to_remove=="pose":
        for landmark_type in landmark_types_to_remove:
            filtered_landmarks = landmarks[(landmarks['type'] == 'pose') & (~landmarks['landmark_index'].between(25, 32))]
    else:
        for landmark_type in landmark_types_to_remove:
            filtered_landmarks = landmarks[landmarks['type'] != landmark_type]

    return filtered_landmarks
