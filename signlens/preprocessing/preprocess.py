import numpy as np
import multiprocessing as mp
import tensorflow as tf
from tqdm import tqdm
from colorama import Fore, Style
import pandas as pd

from signlens.params import *
from signlens.preprocessing.data import load_glossary


################################################################################
# Preprocessing functions
################################################################################

# STRUCTURE
# pad_and_preprocess_sequences_from_pq_file_path_df
# └── pool.imap
#     └── load_pad_preprocess_pq
#         ├── load_relevant_data_subset_from_pq
#         │   ├── pd.read_parquet
#         │   └── filter_relevant_data_subset
#         │       └── filter_out_landmarks
#         └── pad_and_preprocess_landmarks_array
# └── reshape_processed_data_to_tf

def pad_and_preprocess_sequences_from_pq_file_path_df(pq_file_path_df, n_frames=MAX_SEQ_LEN, noface=True):
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
        data_processed = [load_pad_preprocess_pq(pq_file_path, n_frames=n_frames, noface=noface) for pq_file_path in pq_file_path_df]

    data_processed = np.array(data_processed)
    data_tf = reshape_processed_data_to_tf(data_processed, noface=noface, n_frames=n_frames)

    return data_tf


def load_pad_preprocess_pq(pq_file_path, n_frames=MAX_SEQ_LEN, noface=True):
    """
    Load data from a parquet file, pad and preprocess the sequence.

    Parameters:
    - pq_file_path (str): Path to the parquet file.
    - n_frames (int): Number of frames to pad the sequence to. Default is MAX_SEQ_LEN.

    Returns:
    - numpy.ndarray: Padded data reshaped into a 1D array.
    """
    # Load data from the file
    landmarks_array = load_relevant_data_subset_from_pq(pq_file_path, noface=noface)

    # Pad the sequence
    data_processed = pad_and_preprocess_landmarks_array(landmarks_array, n_frames=n_frames)

    # Reshape the data into a 1D array and return it
    # return data_processed.reshape(-1)

    return data_processed


def load_relevant_data_subset_from_pq(pq_path, noface=True):
    """
    Load a relevant data subset from a Parquet file.

    Args:
        pq_path (str): The path to the Parquet file.
        noface (bool, optional): Whether to exclude data without a face. Defaults to True.

    Returns:
        pandas.DataFrame: The loaded relevant data subset.
    """

    landmarks_df = pd.read_parquet(pq_path)

    landmarks_array_filtered = filter_relevant_data_subset(landmarks_df, noface=noface)

    return landmarks_array_filtered


def filter_relevant_data_subset(landmarks_df, noface=True, n_coordinates=N_DIMENSIONS_FOR_MODEL):
    '''
    Loads the relevant data from the input DataFrame.
    If noface is set to True, it excludes landmarks of type 'face'.

    Args:
        landmarks_df (DataFrame): Input DataFrame containing landmarks data.
        noface (bool): If True, excludes landmarks of type 'face'.
        n_coordinates (int): Number of coordintates (x,y,z) to use for the model.

    Returns:
        np.ndarray: NumPy array containing filtered landmarks.
    '''

    # Keep only the defined columns
    landmarks_df = landmarks_df[['type','landmark_index', 'x', 'y', 'z']]

    n_landmarks_per_frame = N_LANDMARKS_ALL

    if noface:
        # Exclude rows where 'type' is 'face' and some portion of 'pose'
        landmarks_df = filter_out_landmarks(landmarks_df, landmark_types_to_remove=['face'])

        # Calculate the number of rows per frame after removing 'face' landmarks
        n_landmarks_per_frame -= N_LANDMARKS_FACE

    if N_LANDMARKS_POSE_TO_TAKE_OFF > 0:
        landmarks_df = filter_out_landmarks(landmarks_df, landmark_types_to_remove=['pose'])

        # Calculate the number of rows per frame after removing 'pose' landmarks
        n_landmarks_per_frame -= N_LANDMARKS_POSE_TO_TAKE_OFF

    # If the model uses 2D data, drop the 'z' dimension
    if n_coordinates == 2:
        data_columns = ['x', 'y']
    else:
        # If the model uses 3D data, keep the 'z' dimension
        data_columns = ['x', 'y', 'z']

    landmarks_df = landmarks_df[data_columns]

    # Calculate the number of frames and dimensions
    n_frames = int(len(landmarks_df) / n_landmarks_per_frame)

    # Reshape to numpy array with 3 dimensions
    landmarks_array = landmarks_df.values.reshape(n_frames, n_landmarks_per_frame, n_coordinates)

    # Convert the data to a float32 array
    landmarks_array = landmarks_array.astype(np.float32)

    return landmarks_array


def filter_out_landmarks(landmarks_df, landmark_types_to_remove):
    """
    Filters out specific landmark types from a DataFrame containing landmarks. For pose, it uses the global variables.

    Args:
        landmarks_df (DataFrame): DataFrame containing landmarks.
        landmark_types_to_remove (list of str): List of landmark types to be removed.

    Returns:
        DataFrame: DataFrame containing filtered landmarks.
    """
    if isinstance(landmark_types_to_remove, str):
        landmark_types_to_remove = [landmark_types_to_remove]

    if "pose" in landmark_types_to_remove and N_LANDMARKS_POSE_TO_TAKE_OFF > 0:

        for landmark_type in landmark_types_to_remove:
            if landmark_type == 'pose':
                landmarks_df = landmarks_df[~((landmarks_df['type'] == 'pose') &
                                              (landmarks_df['landmark_index'].\
                                                  between(N_LANDMARKS_MIN_POSE_TO_TAKE_OFF, N_LANDMARKS_MAX_POSE_TO_TAKE_OFF)))
                                            ]
            else:
                landmarks_df = landmarks_df[landmarks_df['type'] != landmark_type]

    return landmarks_df


def pad_and_preprocess_landmarks_array(landmarks_array, n_frames=MAX_SEQ_LEN, padding_value=MASK_VALUE):
    '''
    Pad or cut off a sequence of landmarks to a specified number of frames.

    Args:
        landmarks_array (numpy.ndarray): Sequence of landmarks.
        n_frames (int, optional): Number of frames to pad or cut off to. Defaults to MAX_SEQ_LEN.

    Returns:
        numpy.ndarray: Padded or cut off sequence of landmarks.
    '''
    # Replace nan values with MASK_VALUE
    landmarks_array[np.isnan(landmarks_array)] = MASK_VALUE

    # Pad with MASK_VALUE or cut off the sequence
    if len(landmarks_array) < n_frames:
        pad_width = int(n_frames - len(landmarks_array))
        landmarks_array = np.pad(landmarks_array, pad_width=((0, pad_width), (0, 0), (0, 0)), mode='constant', constant_values=padding_value)
    else:
        # Cut off the sequence (TO DO: check if sign is at beginning, middle or end)
        landmarks_array = landmarks_array[:n_frames]

    return landmarks_array


def reshape_processed_data_to_tf(data_processed, noface=True, n_frames=MAX_SEQ_LEN, n_coordinates=N_DIMENSIONS_FOR_MODEL):
    """
    Reshape the processed data (3 or 4 DIMS) into a tensor with 3 DIMS and convert it to a TensorFlow tensor.
    3 DIMS if batch_size=1
    for all other cases, 4 DIMS needed

    Parameters:
    - data (numpy.ndarray): Data to be reshaped.

    Returns:
    - tf.Tensor: Reshaped data as a TensorFlow tensor.
    """
    # Case where we provide a single bacth item
    if data_processed.ndim == 3:
        data_processed = np.expand_dims(data_processed, axis=0) # expand dim batch_size

    elif data_processed.ndim != 4:
        raise ValueError(f"Data should have 3 or 4 dimensions, but has {data_processed.ndim}.")

    # Compute number of landmarks per frame
    n_landmarks = N_LANDMARKS_ALL
    if noface:
        n_landmarks -= N_LANDMARKS_FACE
    if N_LANDMARKS_POSE_TO_TAKE_OFF > 0:
        n_landmarks -= N_LANDMARKS_POSE_TO_TAKE_OFF

    if n_frames != data_processed.shape[1]:
        raise ValueError(f"Number of frames ({n_frames}) does not match the number of rows in the data ({data_processed.shape[1]})")

    if n_landmarks != data_processed.shape[2]:
        raise ValueError(f"Number of landmarks ({n_landmarks}) does not match the number of columns in the data ({data_processed.shape[2]})")

    if n_coordinates != data_processed.shape[3]:
        raise ValueError(f"Number of coordinates ({n_coordinates}) does not match the number of columns in the data ({data_processed.shape[3]})")

    # Flatten the 2 last dimensions
    data_reshaped = np.reshape(data_processed , (-1, n_frames, n_landmarks * n_coordinates))

    # Convert to TensorFlow tensor (fatser to reshape to numpy array first and then convert to tensor)
    data_tf = tf.convert_to_tensor(data_reshaped)

    return data_tf


################################################################################
# Label encoding and decoding
################################################################################

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


################################################################################
# Functions used for plotting only
################################################################################

def load_relevant_data_subset_per_landmark_type(pq_path):
    """
    Loads relevant data subset per landmark type from a Parquet file.
    Args:
    pq_path (str): Path to the Parquet file containing the data.
    Returns:
    dict: A dictionary containing data subsets for each landmark type.
          Keys are landmark types ('pose', 'left_hand', 'right_hand') and
          values are numpy arrays containing data subsets for each type.
    """
    data_columns = ['frame', 'type', 'x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = data.frame.nunique()
    data_left_hand = data[data.type == 'left_hand'][[
        'x', 'y', 'z']].values.reshape(n_frames, N_LANDMARKS_HAND, 3)
    data_right_hand = data[data.type == 'right_hand'][[
        'x', 'y', 'z']].values.reshape(n_frames, N_LANDMARKS_HAND, 3)
    data_pose = data[data.type == 'pose'][['x', 'y', 'z']
                                          ].values.reshape(n_frames, N_LANDMARKS_POSE, 3)
    data_dict = {
        'pose': data_pose,
        'left_hand': data_left_hand,
        'right_hand': data_right_hand
    }
    return data_dict


def load_relevant_data_subset_per_landmark_type_from_json(json_path):
    """
    Load a relevant data subset per landmark type from a JSON file.

    Args:
        json_path (str): The path to the JSON file.

    Returns:
        dict: A dictionary containing the loaded data subset per landmark type.
            The dictionary has the following keys:
            - 'pose': A numpy array of shape (n_frames, N_LANDMARKS_POSE, 3) containing pose data.
            - 'left_hand': A numpy array of shape (n_frames, N_LANDMARKS_HAND, 3) containing left hand data.
            - 'right_hand': A numpy array of shape (n_frames, N_LANDMARKS_HAND, 3) containing right hand data.
    """
    data = pd.read_json(json_path)

    n_frames = len(data)

    # Initialize numpy arrays
    data_pose = np.empty((n_frames, N_LANDMARKS_POSE, 3))
    data_left_hand = np.empty((n_frames, N_LANDMARKS_HAND, 3))
    data_right_hand = np.empty((n_frames, N_LANDMARKS_HAND, 3))

    # Populate numpy arrays
    for i, row in data.iterrows():
        pose_landmarks = row['pose']
        left_hand_landmarks = row['left_hand']
        right_hand_landmarks = row['right_hand']

        # Populate pose data
        for idx, landmark in enumerate(pose_landmarks):
            x = landmark['x'] if landmark.get('x') is not None else np.nan
            y = landmark['y'] if landmark.get('y') is not None else np.nan
            z = landmark['z'] if landmark.get('z') is not None else np.nan
            data_pose[i, idx, :] = [x, y, z]

        # Populate left hand data
        for idx, landmark in enumerate(left_hand_landmarks):
            x = landmark['x'] if landmark.get('x') is not None else np.nan
            y = landmark['y'] if landmark.get('y') is not None else np.nan
            z = landmark['z'] if landmark.get('z') is not None else np.nan
            data_left_hand[i, idx, :] = [x, y, z]

        # Populate right hand data
        for idx, landmark in enumerate(right_hand_landmarks):
            x = landmark['x'] if landmark.get('x') is not None else np.nan
            y = landmark['y'] if landmark.get('y') is not None else np.nan
            z = landmark['z'] if landmark.get('z') is not None else np.nan
            data_right_hand[i, idx, :] = [x, y, z]

    data_dict = {
        'pose': data_pose,
        'left_hand': data_left_hand,
        'right_hand': data_right_hand
    }
    return data_dict
