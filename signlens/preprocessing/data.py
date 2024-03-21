from pathlib import Path
from termios import N_SLIP
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

def load_data_subset_csv(frac=DATA_FRAC, noface=True, balanced=False, n_classes=NUM_CLASSES, n_frames=MAX_SEQ_LEN):
    '''
    Load a data subset, as a fraction of the original dataset. It can be balanced, and the number of classes can be limited.

    Parameters:
    - frac (float): Fraction of the original dataset to load. Defaults to 1.0, loading the entire dataset.
    - noface (bool): If True, use the noface dataset. Defaults to True.
    - balanced (bool): If True, balance the dataset based on the distribution of sign categories. Defaults to False.
    - n_classes (int): Number of random classes to include. Defaults to NUM_CLASSES.
    - n_frames (int): Maximum number of frames allowed for a row to be included in the filtered DataFrame. Defaults to MAX_SEQ_LEN.

    Returns:
    - DataFrame: Subset of the training data according to the specified parameters.

    Notes:
    - If balanced is set to True, the dataset is balanced based on the specified number of classes (n_classes).
    - The balanced dataset will have an equal number of samples for each selected class, up to the original distribution.
    '''
    train = pd.read_csv(TRAIN_CSV_PATH)

    total_size = len(train) # total size
    size = total_size # current size (will be modified by the filters)

    # Replace path train_landmark_files with train_landmark_files_noface
    if noface:
        train[['path']] = train[['path']].apply(lambda x: x.str.replace(
            'train_landmark_files', 'train_landmark_files_noface'))

    train['file_path'] = TRAIN_DATA_DIR + os.path.sep + train['path']

    # Filter out parquet files with more than n_frames
    if n_frames is not None:
        train = load_frame_number_parquet(train) # Add n_frames column
        train = filter_out_parquet_frame(train, n_frames=n_frames)
        new_size = len(train)
        size_ratio = new_size / size
        print(f"✅ Filtered on n_frames = {n_frames}. Size reduced from {size} to {new_size} ({size_ratio*100:.1f}%)")
        size = new_size

    # Balance the data if requested
    if balanced:
        # Filter the dataset to include only the selected sign categories
        if n_classes is not None:
            # Randomly select n_classes from all available classes in the dataset
            include_classes = random.sample(list(train['sign'].unique()), n_classes)
            train = train[train['sign'].isin(include_classes)]
            new_size = len(train)
            size_ratio = new_size / size
            print(f"✅ Filtered on n_classes = {n_classes}. Size reduced from {size} to {new_size} ({size_ratio*100:.1f}%)")
            size = new_size
        else:
            # include_classes = train['sign'].unique()
            n_classes = len(include_classes)

        # Calculate the target number of samples after balancing
        target_size = int(size * frac)
        target_size_per_class = target_size // n_classes

        # Calculate how many samples are remaining after distributing equally among sign categories
        remaining_samples = target_size % n_classes

        min_size_per_class = min(train.sign.value_counts()) # min number of elements per sign, in the selected classes

        # If not enough samples, we reduce the sampling size to that value
        if min_size_per_class < target_size_per_class:
            print(f'⚠️ Total size smaller than requested, with {min_size_per_class} per sign instead of {target_size_per_class}')
            target_size_per_class = min_size_per_class
            remaining_samples = 0 # don't add extra samples in this case, we put min_size_per_sign for each sign

        # Initiate the data before concatenation
        remaining_samples_added = 0
        train_balanced = pd.DataFrame()

        # For each selected sign category, adjust the number of samples to match the target size
        for class_ in include_classes:
            train_class = train[train['sign'] == class_]

            if remaining_samples_added < remaining_samples:
                train_class = train_class.sample(target_size_per_class + 1)  # add 1 aditional sample to reach the exact total
                remaining_samples_added += 1
            else:
                train_class = train_class.sample(target_size_per_class)

            train_balanced = pd.concat([train_balanced, train_class], ignore_index=True)

        size_per_class = len(train_balanced) / n_classes

        new_size = len(train_balanced)
        size_ratio = new_size / size
        print(f"✅ Balanced data, with average of {size_per_class} elements per class. Size reduced from {size} to {new_size} ({size_ratio*100:.1f}%)")

        total_size_ratio = new_size / total_size
        print(f"✅ Loaded {size} rows ({total_size_ratio *100:.1f}% of the original {total_size} rows) from the dataset.")
        return train_balanced.sample(frac=1).reset_index(drop=True) # resample to shuffle

    # Case if not balanced but n_classes is specified
    elif n_classes is not None:
        include_classes = random.sample(list(train['sign'].unique()), n_classes)
        train = train[train['sign'].isin(include_classes)]
        train = train.sample(frac=frac)

        new_size = len(train)
        size_ratio = new_size / size
        print(f"✅ Filtered on n_classes = {n_classes}. Size reduced from {size} to {new_size} ({size_ratio*100:.1f}%)")

        size = new_size
        total_size_ratio = size / total_size
        print(f"✅ Loaded {size} rows ({total_size_ratio *100:.1f}% of the original {total_size} rows) from the dataset.")

        return train.reset_index(drop=True)

    # Case if not balanced and n_classes is not specified
    else:
        train = train.sample(frac=frac)

        new_size = len(train)
        size_ratio = new_size / size
        print(f"✅ Loaded {size} rows ({total_size_ratio *100:.1f}% of the original {total_size} rows) from the dataset.")

        return train.reset_index(drop=True)


def load_frame_number_parquet(train, csv_path=TRAIN_DATA_DIR):
    """
    Enhances the input DataFrame by adding a 'frame_parquet' column indicating the number of frames
    for each referenced parquet file. If 'train_frame.csv' exists at 'csv_path', loads DataFrame from it.

    Parameters:
    - train (pd.DataFrame): Input DataFrame with 'file_path' column containing parquet file paths.
    - csv_path (str, optional): Directory path for 'train_frame.csv'. Defaults to 'TRAIN_DATA_DIR'.

    Returns:
    - pd.DataFrame: Enhanced DataFrame with 'frame_parquet' column. Loads from CSV if exists.

    Note:
    - Checks for 'train_frame.csv' existence. Calculates frames if not. Ensure 'TRAIN_DATA_DIR' is defined.
    """
    csv_path = TRAIN_DATA_DIR + os.path.sep + "train_frame.csv"

    # Check if csv file already exists
    if not os.path.exists(csv_path):

        train_with_frame_count = train.copy()
        # If not existing create the column and save the data frame
        with tqdm(total=len(train), desc="Reading parquet files to count frames") as pbar:
            for i in range(len(train)):
                pq_file_path = train.loc[i, "file_path"]
                parquet_df = pd.read_parquet(pq_file_path )
                n_frames = parquet_df["frame"].nunique() # calculate the number of frames
                n_frames2 = parquet_df["frame"].iloc[-1] - parquet_df["frame"].iloc[0] + 1 # calculate the number of frames from start to end
                if n_frames != n_frames2:
                    print(f"!! Warning, file {pq_file_path} might have missing frames")

                train_with_frame_count.loc[i, "n_frames"] = n_frames
                train_with_frame_count.loc[i, "n_frames2"] = n_frames2

                pbar.update(1)

        train_with_frame_count = train_with_frame_count[["sequence_id", "n_frames", "n_frames2"]]
        train_with_frame_count.to_csv(csv_path, index=False)
        print(f" ✅ File with frame_parquet has been saved at : {csv_path}")

    # if file exists, load it
    else:
        train_with_frame_count = pd.read_csv(csv_path)
        print("✅ File with frames already exists, loaded matching 'sequence_id' rows.")

    train = pd.merge(train, train_with_frame_count, how="left", on='sequence_id')

    return train

def filter_out_parquet_frame(df, n_frames=MAX_SEQ_LEN):
    """
    Filters the DataFrame by the 'n_frames' column to include only rows where the number of frames is less than or equal to the specified threshold.
    This function is intended to be used on DataFrames that have already been processed by the 'load_frame_number_parquet'
    function, which adds the 'n_frames' column indicating the number of frames for each parquet file.

    Parameters:
    - df (pd.DataFrame): The DataFrame to filter. It must include a 'n_frames' column.
    - n_frames (int): The maximum number of frames allowed for a row to be included in the filtered DataFrame.

    Returns:
    - pd.DataFrame: A new DataFrame consisting of rows from the original DataFrame where the 'n_frames' value is less than or equal to 'n_frame'.
    The index of the DataFrame will be reset.

    """
    return df[df["n_frames"] <= n_frames].reset_index(drop=True)


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
