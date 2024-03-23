# from pathlib import Path
# from termios import N_SLIP
import pandas as pd
import numpy as np
from tqdm import tqdm  # Import tqdm for the progress bar
import os
import sys
from colorama import Fore, Style

from signlens.params import *

################################################################################
# LOAD CSV
################################################################################

def load_data_subset_csv(frac=DATA_FRAC, noface=True, balanced=False, n_classes=NUM_CLASSES, n_frames=MAX_SEQ_LEN, random_state=None, csv_path=TRAIN_TRAIN_CSV_PATH):
    '''
    Load a data subset, as a fraction of the original dataset. It can be balanced, and the number of classes can be limited.

    Parameters:
    - frac (float): Fraction of the original dataset to load. Defaults to 1.0, loading the entire dataset.
    - noface (bool): If True, use the noface dataset. Defaults to True.
    - balanced (bool): If True, balance the dataset based on the distribution of sign categories. Defaults to False.
    - n_classes (int): Number of random classes to include. Defaults to NUM_CLASSES.
    - n_frames (int): Maximum number of frames allowed for a row to be included in the filtered DataFrame. Defaults to MAX_SEQ_LEN.
    - random_state (int, or None): Random seed for reproducibility. Defaults to None.

    Returns:
    - DataFrame: Subset of the training data according to the specified parameters.

    Notes:
    - If balanced is set to True, the dataset is balanced based on the specified number of classes (n_classes).
    - The balanced dataset will have an equal number of samples for each selected class, up to the original distribution.
    '''
    if not os.path.exists(csv_path):
        return ValueError(f"❌ File {csv_path} does not exist. Did you do the train_test_split?.")

    print(Fore.BLUE + f"Loading data subset from {os.path.basename(csv_path)}" + Style.RESET_ALL)

    train = pd.read_csv(csv_path) # load the specified document

    total_size = len(train) # total size
    size = total_size # current size (will be modified by the filters)

    # Replace path train_landmark_files with train_landmark_files_noface
    if noface:
        train[['path']] = train[['path']].apply(lambda x: x.str.replace(
            'train_landmark_files', 'train_landmark_files_noface'))

    train['file_path'] = TRAIN_DATA_DIR + os.path.sep + train['path']
    if 'n_frames' not in train.columns or 'n_frames2' not in train.columns:
        train = load_frame_number_parquet(train, csv_path=TRAIN_CSV_PATH) # Add n_frames column

    train = filter_sequences_with_missing_frames(train) # Filter out sequences with missing frames

    # Filter out parquet files with more than n_frames
    if n_frames is not None:
        train = filter_out_parquet_frame(train, n_frames=n_frames)
        new_size = len(train)
        size_ratio = new_size / size
        print(f"✅ Filtered on n_frames = {n_frames}. Size reduced from {size} to {new_size} ({size_ratio*100:.2f}%)")
        size = new_size

    # Balance the data if requested
    if balanced:
        # Filter the dataset to include only the selected sign categories
        if n_classes is not None:
            # Select the first n_classes from the glossary
            all_classes = load_glossary()
            include_classes = all_classes[:n_classes]['sign'].to_list()
            train = train[train['sign'].isin(include_classes)]
            new_size = len(train)
            size_ratio = new_size / size
            print(f"✅ Filtered on n_classes = {n_classes}. Size reduced from {size} to {new_size} ({size_ratio*100:.2f}%)")
            size = new_size
        else:
            include_classes = load_glossary()
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
                train_class = train_class.sample(target_size_per_class + 1, random_state=random_state)  # add 1 aditional sample to reach the exact total
                remaining_samples_added += 1
            else:
                train_class = train_class.sample(target_size_per_class, random_state=random_state)

            train_balanced = pd.concat([train_balanced, train_class], ignore_index=True)

        size_per_class = len(train_balanced) / n_classes

        new_size = len(train_balanced)
        size_ratio = new_size / size
        print(f"✅ Balanced data, with average of {size_per_class} elements per class. Size reduced from {size} to {new_size} ({size_ratio*100:.2f}%)")

        total_size_ratio = new_size / total_size
        print(f"✅ Loaded {size} rows ({total_size_ratio *100:.2f}% of the original {total_size} rows) from the dataset.")
        return train_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True) # resample to shuffle

    # Case if not balanced but n_classes is specified
    elif n_classes is not None:
        include_classes = pd.Series(train['sign'].unique()).sample(n=n_classes, random_state=random_state)
        train = train[train['sign'].isin(include_classes)]
        train = train.sample(frac=frac, random_state=random_state)

        new_size = len(train)
        size_ratio = new_size / size
        print(f"✅ Filtered on n_classes = {n_classes}. Size reduced from {size} to {new_size} ({size_ratio*100:.2f}%)")

        size = new_size
        total_size_ratio = size / total_size
        print(f"✅ Loaded {size} rows ({total_size_ratio *100:.2f}% of the original {total_size} rows) from the dataset.")

        return train.reset_index(drop=True)

    # Case if not balanced and n_classes is not specified
    else:
        train = train.sample(frac=frac, random_state=random_state)

        new_size = len(train)
        size_ratio = new_size / size
        print(f"✅ Loaded {size} rows ({total_size_ratio *100:.2f}% of the original {total_size} rows) from the dataset.")

        return train.reset_index(drop=True)

def unique_train_test_split():
    test_size = 0.2

    print(Fore.BLUE + f"Loading unqiue test set with test_size={test_size}" + Style.RESET_ALL)

    test_data = load_data_subset_csv(frac=test_size, noface=False, balanced=True, n_classes=250, n_frames=100, random_state=42, csv_path=TRAIN_CSV_PATH)

    print(Fore.BLUE + "\nLoading training test set" + Style.RESET_ALL)
    all_data = load_data_subset_csv(frac=1, noface=False, balanced=False, n_classes=250, n_frames=None, random_state=None, csv_path=TRAIN_CSV_PATH)
    train_data = all_data[~all_data.isin(test_data)]

    total_len = len(all_data)
    train_len = len(train_data)
    test_len = len(test_data)

    test_ratio =  test_len  / total_len
    train_ratio = train_len / total_len

    print(Fore.BLUE + f"\nTotal loaded rows : {total_len} \
        \nTotal training rows : {train_len} ({train_ratio*100:.2f}%) \
        \nTotal test rows : {test_len} ({test_ratio*100:.2f}%)" + Style.RESET_ALL)

    train_data.drop(columns=['file_path']).to_csv(TRAIN_TRAIN_CSV_PATH, index=False)
    test_data.drop(columns=['file_path']).to_csv(TRAIN_TEST_CSV_PATH, index=False)
    print(Fore.BLUE + f"\nTrain and test data saved at {TRAIN_TRAIN_CSV_PATH} and {TRAIN_TEST_CSV_PATH}" + Style.RESET_ALL)


def write_train_test_csv():
    test_data, train_data = unique_train_test_split()
    test_data.to_csv(TRAIN_TEST_CSV_PATH, index=False)
    train_data.to_csv(TRAIN_TRAIN_CSV_PATH, index=False)


def load_frame_number_parquet(train, csv_path):
    """
    Enhances the input DataFrame by adding 'n_frames' and 'n_frames2' columns indicating the number of frames
    for each referenced parquet file. If '<filename>_frame.csv' exists at 'csv_path', loads DataFrame from it.

    Parameters:
    - train (pd.DataFrame): Input DataFrame with 'file_path' column containing parquet file paths.
    - csv_path (str, optional): Directory path for '<filename>.csv'. Defaults to 'TRAIN_DATA_DIR'.

    Returns:
    - pd.DataFrame: Enhanced DataFrame with 'n_frames' and 'n_frames2' columns. Loads from CSV if exists.

    Note:
    - Checks for '<filename>_frame.csv' existence. Calculates frames if not. Ensure 'TRAIN_DATA_DIR' is defined.
    - 'n_frames' is the number of unique frames in the parquet file.
    - 'n_frames2' is the number of frames from start to end in the parquet file.
    - If 'n_frames' and 'n_frames2' are not equal, a warning is printed indicating that the parquet file might have missing frames.
    """

    dir_path = os.path.dirname(csv_path)
    filename, file_extension = os.path.splitext(os.path.basename(csv_path))
    new_filename = f"{filename}_frame{file_extension}"
    frame_csv_path = os.path.join(dir_path, new_filename)

    # Check if csv file already exists
    if not os.path.exists(frame_csv_path):

        train_with_frame_count = train.copy()
        # If not existing create the column and save the data frame
        with tqdm(total=len(train), desc="Reading parquet files to count frames") as pbar:
            for i in range(len(train)):
                pq_file_path = train.loc[i, "file_path"]
                parquet_df = pd.read_parquet(pq_file_path )
                n_frames = parquet_df["frame"].nunique() # calculate the number of frames
                n_frames2 = parquet_df["frame"].iloc[-1] - parquet_df["frame"].iloc[0] + 1 # calculate the number of frames from start to end

                train_with_frame_count.loc[i, "n_frames"] = n_frames
                train_with_frame_count.loc[i, "n_frames2"] = n_frames2

                pbar.update(1)

        train_with_frame_count = train_with_frame_count[["sequence_id", "n_frames", "n_frames2"]]
        train_with_frame_count.to_csv(frame_csv_path, index=False)
        print(f" ✅ File with frame_parquet has been saved at : {frame_csv_path }")

    # if file exists, load it
    else:
        train_with_frame_count = pd.read_csv(frame_csv_path)
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

def filter_sequences_with_missing_frames(df, threshold=10):
    """
    Filters out sequences that have missing frames (count of frames that is too different from the last_frame - first_frame + 1)

    Parameters:
    - df (DataFrame): The DataFrame containing the sequences to filter.
    - threshold (int, optional): The maximum allowed difference in frame counts. Sequences with a difference greater than this threshold will be filtered out. Default is 10.

    Returns:
    - DataFrame: A DataFrame with sequences filtered based on the frame count difference.
    """
    delta = abs(df["n_frames"] - df["n_frames2"])

    return df[delta < threshold].reset_index(drop=True)

################################################################################
# LOAD GLOSSARY CSV
################################################################################

def load_glossary(csv_path=GLOSSARY_CSV_PATH):
    """
    Load a glossary from a CSV file into a pandas DataFrame.

    Parameters:
    - csv_path (str): The file path to the CSV file containing the glossary. Default is GLOSSARY_CSV_PATH.

    Returns:
    pandas.DataFrame: A DataFrame containing the loaded glossary data.
    """

    return pd.read_csv(csv_path)

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
