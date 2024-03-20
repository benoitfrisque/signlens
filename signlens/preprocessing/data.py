from pathlib import Path
import pandas as pd
import numpy as np
import math
from tqdm import tqdm  # Import tqdm for the progress bar
import os
from colorama import Fore, Style

from signlens.params import *



import pandas as pd
import random

################################################################################
# LOAD CSV
################################################################################

def load_data_subset_csv(frac=1.0, noface=True, balanced=False, num_signs=None):
    '''
    Load a data subset, as a fraction of the original dataset. It can be balanced, and the number of signs can be limited.

    Parameters:
    - frac (float): Fraction of the original dataset to load. Defaults to 1.0, loading the entire dataset.
    - noface (bool): If True, use the noface dataset. Defaults to True.
    - balanced (bool): If True, balance the dataset based on the distribution of sign categories. Defaults to False.
    - num_signs (int or None): Number of random sign categories to include. Defaults to None, including all sign categories.

    Returns:
    - DataFrame: Subset of the training data according to the specified parameters.

    Notes:
    - If balanced is set to True, the dataset is balanced based on the specified number of sign categories (num_signs).
    - The balanced dataset will have an equal number of samples for each selected sign category, up to the original distribution.
    '''
    train = pd.read_csv(TRAIN_CSV_PATH)

    # replace path train_landmark_files with train_landmark_files_noface
    if noface:
        train[['path']] = train[['path']].apply(lambda x: x.str.replace(
            'train_landmark_files', 'train_landmark_files_noface'))

    train['file_path'] = TRAIN_DATA_DIR + os.path.sep + train['path']

    # Balance the data if requested
    if balanced:
        if num_signs is not None:
            # Randomly select num_signs from all available sign categories in the dataset
            include_signs = random.sample(list(train['sign'].unique()), num_signs)
        else:
            include_signs = train['sign'].unique()
            num_signs = len(include_signs)

        # Filter the dataset to include only the selected sign categories
        train_subset = train[train['sign'].isin(include_signs)]

        # Calculate the target number of samples after balancing
        initial_size = len(train)
        target_size = int(initial_size * frac)
        target_size_per_sign = target_size // num_signs

        # Calculate how many samples are remaining after distributing equally among sign categories
        remaining_samples = target_size % num_signs

        min_size_per_sign = min(train_subset.sign.value_counts()) # min number of elements per sign, in the selected signs

        # If not enough samples, we reduce the sampling size to that value
        if min_size_per_sign < target_size_per_sign:
            print(f'Warning: total size smaller than requested, with {min_size_per_sign} per sign instead of {target_size_per_sign}')
            target_size_per_sign = min_size_per_sign
            remaining_samples = 0 # don't add extra samples in this case, we put min_size_per_sign for each sign

        # Initiate the data before concatenation
        remaining_samples_added = 0
        balanced_data = pd.DataFrame()

        # For each selected sign category, adjust the number of samples to match the target size
        for sign_category in include_signs:
            sign_data = train[train['sign'] == sign_category]

            if remaining_samples_added < remaining_samples:

                sign_data = sign_data.sample(target_size_per_sign+1)  # add 1 aditional sample to reach the exact total
                remaining_samples_added += 1
            else:
                sign_data = sign_data.sample(target_size_per_sign) # add aditional sample

            balanced_data = pd.concat([balanced_data, sign_data], ignore_index=True)

        size = len(balanced_data)
        size_ratio = size / initial_size
        print(f"Size reduced from {initial_size} to {size} ({size_ratio*100:.1f}%)")

        return balanced_data.reset_index(drop=True)

    else:
        train = train.sample(frac=frac)
        return train.reset_index(drop=True)


def load_frame_number_parquet(train, csv_path=TRAIN_CSV_PATH):
    """
    Enhances the input 'train' DataFrame by adding a 'frame_parquet' column which calculates the number of frames
    for each parquet file referenced in the DataFrame. If a CSV file at the specified path ('csv_path') named 'train_frame.csv'
    already exists, this function loads the DataFrame from that CSV instead of re-processing the parquet files.

    Parameters:
    - train (pd.DataFrame): The input DataFrame containing a column 'file_path' with paths to the parquet files.
    - csv_path (str, optional): The directory path where 'train_frame.csv' will be saved or loaded from.
      Defaults to 'TRAIN_DATA_DIR'.

    Returns:
    - pd.DataFrame: The enhanced DataFrame with a 'frame_parquet' column indicating the number of frames
      for each file. If 'train_frame.csv' exists, returns the DataFrame loaded from this CSV.

    Note:
    - This function checks for the existence of 'train_frame.csv' in the specified 'csv_path'.
      If the file does not exist, it calculates the frame difference for each parquet file and saves the result as a CSV.
      If the file exists, it loads and returns the DataFrame from the CSV, bypassing the calculation.
    - Make sure 'TRAIN_DATA_DIR' is defined and accessible in your environment before using this function.

    """

    # Check if csv file already exist
    if not os.path.exists(csv_path):
        # If not existing create the column and save the data frame
        for i in range(len(train)):
            df = pd.read_parquet(train.loc[i, "file_path"]).copy()
            train.at[i, "frame_parquet"] = df["frame"].iloc[-1] - df["frame"].iloc[0] + 1
        train.to_csv(csv_path, index=False)
        print(f" ✅ File has been saved at : {csv_path}")
    else:
        full_df = pd.read_csv(csv_path)
        train = full_df[full_df['sequence_id'].isin(train['sequence_id'])]
        print("File already exists, loaded matching 'sequence_id' rows.")

    return train

def filter_out_parquet_frame(df, n_frame=100):
    """
    Filters the DataFrame by the 'frame_parquet' column to include only rows where the number of frames is less than or equal to the specified threshold.
    This function is intended to be used on DataFrames that have already been processed by the 'load_frame_number_parquet'
    function, which adds the 'frame_parquet' column indicating the number of frames for each parquet file.

    Parameters:
    - df (pd.DataFrame): The DataFrame to filter. It must include a 'frame_parquet' column.
    - n_frame (int): The maximum number of frames allowed for a row to be included in the filtered DataFrame.

    Returns:
    - pd.DataFrame: A new DataFrame consisting of rows from the original DataFrame where the 'frame_parquet' value is less than or equal to 'n_frame'.
    The index of the DataFrame will be reset.

    """
    return df[df["frame_parquet"]<=n_frame].reset_index(drop=True)


################################################################################
# LOAD PARQUET FILES
################################################################################

def load_relevant_data_subset(pq_path, noface=True):
    '''
    loads the relevant data from the parquet file.
    If noface is set to True, it excludes landmark 'face'.
    Fills NaN values with 0.

    Args:
        file_path (str or Path): Path to the input parquet file.

    Returns:
        NumPy array: NumPy array containing filtered landmarks.
    '''
    data_columns = ['x', 'y', 'z', 'type']  # Include the 'type' column

    data = pd.read_parquet(pq_path, columns=data_columns)

    if noface:
        # Exclude rows where 'type' is 'face'
        data = filter_out_landmarks(pq_path, landmark_types_to_remove=['face'], data_columns=data_columns)

        data = data[data['type'] != 'face']
        # N_LANDMARKS_NOFACE 75
        frame_rows = N_LANDMARKS_NO_FACE

    else:
        frame_rows = N_LANDMARKS_ALL

    data = data.drop(columns=['type'])
    data_columns = data_columns[:-1]

    # Replace NaN values with 0
    data.fillna(0, inplace=True)
    n_frames = int(len(data) / frame_rows)
    n_dim = len(data_columns)
    data = data.values.reshape(n_frames, frame_rows, n_dim)

    return data.astype(np.float32)

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
    data_columns = ['frame','type','x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = data.frame.nunique()
    data_left_hand = data[data.type == 'left_hand'][['x', 'y', 'z']].values.reshape(n_frames, N_LANDMARKS_HAND, 3)
    data_right_hand = data[data.type == 'right_hand'][['x', 'y', 'z']].values.reshape(n_frames, N_LANDMARKS_HAND, 3)
    data_pose = data[data.type == 'pose'][['x', 'y', 'z']].values.reshape(n_frames, N_LANDMAKRS_POSE, 3)
    data_dict = {
        'pose': data_pose,
        'left_hand': data_left_hand,
        'right_hand': data_right_hand
    }
    return data_dict

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

    for landmark_type in landmark_types_to_remove:
        filtered_landmarks = landmarks[landmarks['type'] != landmark_type]

    return filtered_landmarks
