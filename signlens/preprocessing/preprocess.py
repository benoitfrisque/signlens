import numpy as np
import pandas as pd
import multiprocessing as mp
import tensorflow as tf
import json
from tqdm import tqdm
from colorama import Fore, Style

from signlens.params import *
from signlens.preprocessing.data import load_glossary
from signlens.preprocessing.glossary import load_glossary_decoding


# STRUCTURE
# pad_and_preprocess_sequences_from_pq_file_path_df
# └── pool.imap
#     └── load_pad_preprocess_pq
#         ├── load_relevant_data_subset_from_pq
#         │   ├── pd.read_parquet
#         │   └── filter_relevant_landmarks_and_coordinates
#         └── pad_and_preprocess_landmarks_array
# └── reshape_processed_data_to_tf

################################################################################
# FUNCTION FOR PARQUET FILES
################################################################################

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


def load_pad_preprocess_pq(pq_file_path, n_frames=MAX_SEQ_LEN, noface=True, n_coordinates=N_DIMENSIONS_FOR_MODEL):
    """
    Load data from a parquet file, pad and preprocess the sequence.

    Parameters:
    - pq_file_path (str): Path to the parquet file.
    - n_frames (int): Number of frames to pad the sequence to. Default is MAX_SEQ_LEN.
    - noface (bool): Flag indicating whether to exclude face landmarks. Default is True.
    - n_coordinates (int): Number of coordinates for the model. Default is N_DIMENSIONS_FOR_MODEL.

    Returns:
    - numpy.ndarray: Padded data reshaped into a 1D array.
    """
    # Load data from the file
    landmarks_array = load_relevant_data_subset_from_pq(pq_file_path, noface=noface,  n_coordinates=n_coordinates)

    # Pad the sequence
    data_processed = pad_and_preprocess_landmarks_array(landmarks_array, n_frames=n_frames)

    return data_processed


def load_relevant_data_subset_from_pq(pq_path, noface=True, n_coordinates=N_DIMENSIONS_FOR_MODEL):
    """
    Load a relevant data subset from a Parquet file.

    Args:
        pq_path (str): The path to the Parquet file.
        noface (bool, optional): Whether to exclude data without a face. Defaults to True.
        n_coordinates (int, optional): The number of coordinates (x, y, z) for the model. Defaults to N_DIMENSIONS_FOR_MODEL.

    Returns:
        pandas.DataFrame: The loaded relevant data subset.
    """

    landmarks_df = pd.read_parquet(pq_path, columns=['frame', 'type', 'landmark_index', 'x', 'y', 'z'])

    landmarks_array_filtered = filter_relevant_landmarks_and_coordinates(landmarks_df, noface=noface, n_coordinates=n_coordinates)

    return landmarks_array_filtered


################################################################################
# FUNCTIONS FOR JSON FILES (FOR THE API)
################################################################################

def preprocess_data_from_json_data(json_data, noface=True, n_coordinates=N_DIMENSIONS_FOR_MODEL, n_frames=MAX_SEQ_LEN):
    """
    Preprocesses the data from a JSON list.

    Args:
        json_data (dict): The JSON data to be processed.
        noface (bool, optional): Whether to exclude face landmarks. Defaults to True.
        n_coordinates (int, optional): The number of dimensions for the model. Defaults to N_DIMENSIONS_FOR_MODEL.
        n_frames (int, optional): The maximum sequence length. Defaults to MAX_SEQ_LEN.

    Returns:
        tf.Tensor: The preprocessed data in TensorFlow format.
    """
    landmarks_df = convert_landmarks_json_data_to_df(json_data)
    filtered_landmarks_array = filter_relevant_landmarks_and_coordinates(landmarks_df, noface=noface, n_coordinates=n_coordinates)
    data_processed = pad_and_preprocess_landmarks_array(filtered_landmarks_array, n_frames=n_frames)
    data_processed_tf = reshape_processed_data_to_tf(data_processed, noface=noface, n_frames=n_frames, n_coordinates=n_coordinates)

    return data_processed_tf


def load_landmarks_json_from_path(json_path):
    """
    Load landmarks data from a JSON file.

    Args:
        json_path (str): The path to the JSON file.

    Returns:
        pandas.DataFrame: A DataFrame containing the loaded landmarks data.
    """
    with open(json_path, 'r') as file:
        json_data = json.load(file)
        json_df = convert_landmarks_json_data_to_df(json_data)

        return json_df


def convert_landmarks_json_data_to_df(json_data):
    """
    Convert landmarks JSON data to a DataFrame.

    Args:
        json_data (list): A list of JSON objects containing landmarks data.

    Returns:
        pandas.DataFrame: A DataFrame containing the converted landmarks data.

    """
    # Initialize an empty list to store the converted data
    converted_data = []

    # Define the keys to iterate over
    landmark_types = ['pose', 'left_hand', 'right_hand']

    # Iterate over each frame in the JSON data
    for frame_index, frame in enumerate(json_data):
        # Iterate over each key
        for landmark_type in landmark_types:
            # Extract landmarks
            landmarks = frame[landmark_type]
            for landmark in landmarks:
                converted_data.append({
                    'frame': frame_index,
                    'type': landmark_type,
                    'landmark_index': landmark['landmark_index'],
                    'x': landmark['x'],
                    'y': landmark['y'],
                    'z': landmark['z'],
                })

    # Create a DataFrame from the converted data
    df = pd.DataFrame(converted_data)

    return df


################################################################################
# PREPROCESSING FUNCTIONS
################################################################################

def filter_relevant_landmarks_and_coordinates(landmarks_df, noface=True, n_coordinates=N_DIMENSIONS_FOR_MODEL):
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
    landmarks_df = landmarks_df[['frame', 'type','landmark_index', 'x', 'y', 'z']]

    # Define the order of the 'type' column
    type_order = ['face', 'pose', 'left_hand', 'right_hand']

    # Convert the 'type' column to a categorical type with the specified order
    landmarks_df['type'] = pd.Categorical(landmarks_df['type'], categories=type_order, ordered=True)

    # Sort the DataFrame by 'frame', 'type' and 'landmark_index', to ensure the correct order of landmarks
    landmarks_df.sort_values(by=['frame', 'type', 'landmark_index'], inplace=True)

    n_landmarks_per_frame = N_LANDMARKS_ALL

    landmark_types_to_remove = []

    if noface:
        # Exclude rows where 'type' is 'face' and some portion of 'pose'
        landmark_types_to_remove.append('face')

        # Calculate the number of rows per frame after removing 'face' landmarks
        n_landmarks_per_frame -= N_LANDMARKS_FACE

    if N_LANDMARKS_POSE_TO_TAKE_OFF > 0:
        landmark_types_to_remove.append('pose')

        # Calculate the number of rows per frame after removing 'pose' landmarks
        n_landmarks_per_frame -= N_LANDMARKS_POSE_TO_TAKE_OFF

    for landmark_type in landmark_types_to_remove:
        if landmark_type == 'pose' and N_LANDMARKS_POSE_TO_TAKE_OFF > 0:
            landmarks_df = landmarks_df[~((landmarks_df['type'] == 'pose') &
                                          (landmarks_df['landmark_index'].\
                                              between(N_LANDMARKS_MIN_POSE_TO_TAKE_OFF, N_LANDMARKS_MAX_POSE_TO_TAKE_OFF)))
                                        ]
        else:
            landmarks_df = landmarks_df[landmarks_df['type'] != landmark_type]

    # If the model uses 2D data, drop the 'z' dimension
    if n_coordinates == 2:
        data_columns = ['x', 'y']
    else:
        # If the model uses 3D data, keep the 'z' dimension
        data_columns = ['x', 'y', 'z']

    landmarks_df = landmarks_df[data_columns].reset_index(drop=True)

    # Calculate the number of frames and dimensions
    n_frames = int(len(landmarks_df) / n_landmarks_per_frame)

    # Reshape to numpy array with 3 dimensions
    landmarks_array = landmarks_df.values.reshape(n_frames, n_landmarks_per_frame, n_coordinates)

    # Convert the data to a float32 array
    landmarks_array = landmarks_array.astype(np.float32)

    return landmarks_array

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
# Normalising Data
################################################################################

def normalize_data_tf(tf_data):
    """
    Working only for 2dimensions x,y - 3D to uptade
    Normalize the data using max min normalization.

    Parameters:
    - tf.Tensor: Data to be normalized.

    Returns:
    - tf.Tensor: Normalized data.
    """
    mask_x = tf.not_equal(tf_data[..., ::2], MASK_VALUE)
    mask_y = tf.not_equal(tf_data[..., 1::2], MASK_VALUE)

    x_max = X_MAX
    x_min = X_MIN
    y_max = Y_MAX
    y_min = Y_MIN

    x_normalized = tf.where(mask_x, (tf_data[..., ::2] - x_min) / (x_max - x_min), MASK_VALUE)
    y_normalized = tf.where(mask_y, (tf_data[..., 1::2] - y_min) / (y_max - y_min), MASK_VALUE)


    tf_tensor_normalized = tf.reshape(tf.stack([x_normalized, y_normalized], axis=-1), tf_data.shape)


    return tf_tensor_normalized


def augment_data_by_mirror_x(tf_normalized_train):
    """
    Augment data by creating a mirror effect on x-axis.

    Parameters:
    - tf.Tensor: Normalized training data.

    Returns:
    - tf.Tensor: Data augmented with mirror effect on x-axis.
    """

    x_augmented = tf.where(
        tf_normalized_train[..., ::2] != MASK_VALUE,
        1.0 - tf_normalized_train[..., ::2],
        MASK_VALUE
    )


    y_values = tf_normalized_train[..., 1::2]
    tf_augmented_train = tf.reshape(tf.stack([x_augmented, y_values], axis=-1), tf_normalized_train.shape)

    return tf_augmented_train

def concatenate_data(tf_train, tf_augmented_train):

    """
    Concatenate the original training data with the augmented data.

    Parameters:
    - tf.Tensor: Original training data.
    - tf.Tensor: Augmented training data.

    Returns:
    - tf.Tensor: Concatenated training data.
    """
    return tf.concat([tf_train, tf_augmented_train], axis=0)



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

    glossary = load_glossary_decoding()

    # Extract labels from the glossary
    labels = glossary['sign'].tolist()

    # Get the index with maximum value for each row in y_encoded
    decoded_indices = np.argmax(y_encoded, axis=1)

    # Map indices back to labels
    decoded_labels = [labels[idx] for idx in decoded_indices]

    predict_proba = np.max(y_encoded, axis=1)

    return decoded_labels, predict_proba


################################################################################
# FUNCTIONS USED FOR PLOTTING ONLY
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
                                          ].values.reshape(n_frames, N_LANDMARKS_POSE_TOTAL, 3)
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
