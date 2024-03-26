import os
import pandas as pd
import numpy as np
import json

from tqdm import tqdm  # Import tqdm for the progress bar
from pathlib import Path
from colorama import Fore, Style

from signlens.params import *
from signlens.preprocessing.glossary import load_glossary

################################################################################
# LOAD CSV WITH LIST OF PARQUET FILES
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
    - csv_path (str): Path to the CSV file containing the dataset. Defaults to TRAIN_TRAIN_CSV_PATH.

    Returns:
    - DataFrame: Subset of the training data according to the specified parameters.

    Notes:
    - If balanced is set to True, the dataset is balanced based on the specified number of classes (n_classes).
    - The balanced dataset will have an equal number of samples for each selected class, up to the original distribution.
    - If the CSV file specified by csv_path does not exist, a ValueError is raised.

    '''
    if not os.path.exists(csv_path):
        return ValueError(f"❌ File {csv_path} does not exist. Did you do the train_test_split?.")

    print(Fore.BLUE +
          f"Loading data subset from {os.path.basename(csv_path)}" + Style.RESET_ALL)

    if random_state is not None:
        print(f"    ℹ️ Random state set for data loading : {random_state}")

    train = pd.read_csv(csv_path)  # load the specified document

    total_size = len(train)  # total size
    size = total_size  # current size (will be modified by the filters)

    # Replace path train_landmark_files with train_landmark_files_noface
    if noface:
        train[['path']] = train[['path']].apply(lambda x: x.str.replace(
            'train_landmark_files', 'train_landmark_files_noface'))

    # Add file_path column with absolute path
    train['file_path'] = TRAIN_DATA_DIR + os.path.sep + train['path']

    # Add n_frames column
    if 'n_frames' not in train.columns or 'n_frames2' not in train.columns:
        train = load_frame_number_parquet(train)

    # Filter out sequences with missing frames
    train = filter_sequences_with_missing_frames(train)

    new_size = len(train)
    size_ratio = new_size / size
    print(
        f"    ℹ️ Filtered sequences with missing frames. Size reduced from {size} to {new_size} ({size_ratio*100:.2f}%)")
    size = new_size

    # Filter out parquet files with more than n_frames
    if n_frames is not None:
        train = filter_out_parquet_frame(train, n_frames=n_frames)
        new_size = len(train)
        size_ratio = new_size / size
        print(
            f"    ℹ️ Filtered on n_frames = {n_frames}. Size reduced from {size} to {new_size} ({size_ratio*100:.2f}%)")
        size = new_size

    # Balance the data if requested
    if balanced:
        train_balanced, size = balance_data(train, n_classes, frac, random_state, size)
        train_balanced = train_balanced.sample(frac=1, random_state=random_state) # shuffle

        if n_classes is None:
            n_classes = len(train_balanced.sign.unique())

        size_per_class = len(train_balanced) / n_classes
        new_size = len(train_balanced)
        size_ratio = new_size / size
        print(
            f"    ℹ️ Balanced data, with average of {size_per_class:.1f} elements per class. Size reduced from {size} to {new_size} ({size_ratio*100:.2f}%)")
        size = new_size
        total_size_ratio = size / total_size
        print(
            f"✅ Loaded {size} rows ({total_size_ratio *100:.2f}% of the original {total_size} rows) from the dataset.")
        return train_balanced.reset_index(drop=True)

    # Case if not balanced but n_classes is specified
    elif n_classes is not None:
        train_filtered, _ = filter_by_classes(train, n_classes)
        train_filtered = train_filtered.sample(
            frac=frac, random_state=random_state)
        new_size = len(train_filtered)
        size_ratio = new_size / size
        print(
            f"    ℹ️ Filtered on n_classes = {n_classes}. Size reduced from {size} to {new_size} ({size_ratio*100:.2f}%)")

        size = new_size
        total_size_ratio = size / total_size
        print(
            f"✅ Loaded {size} rows ({total_size_ratio *100:.2f}% of the original {total_size} rows) from the dataset.")

        return train_filtered.reset_index(drop=True)

    # Case if not balanced and n_classes is not specified
    else:
        train = train.sample(frac=frac, random_state=random_state)

        new_size = len(train)
        size = new_size
        total_size_ratio = size / total_size

        print(
            f"✅ Loaded {size} rows ({total_size_ratio *100:.2f}% of the original {total_size} rows) from the dataset.")

        return train.reset_index(drop=True)


def balance_data(train, n_classes, frac, random_state, size):
    """
    Balances the data by adjusting the number of samples for each sign category to match the target size.

    Args:
        train (pd.DataFrame): The input dataset.
        n_classes (int): The number of sign categories to include. If None, all sign categories will be included.
        frac (float): The fraction of the target size to use for balancing.
        random_state (int): The random seed for reproducibility.
        size (int): The original size of the dataset.

    Returns:
        pd.DataFrame: The balanced dataset.
        int: The size of the balanced dataset.
    """
    # Filter the dataset to include only the selected sign categories
    if n_classes is not None:
        # Select the first n_classes from the glossary
        train, include_classes = filter_by_classes(train, n_classes)
        new_size = len(train)
        size_ratio = new_size / size
        print(
            f"    ℹ️ Filtered on n_classes = {n_classes}. Size reduced from {size} to {new_size} ({size_ratio*100:.2f}%)")

        size = new_size
    else:
        include_classes = load_glossary().sign
        n_classes = len(include_classes)

    # Calculate the target number of samples after balancing
    target_size = int(size * frac)
    target_size_per_class = target_size // n_classes

    # Calculate how many samples are remaining after distributing equally among sign categories
    remaining_samples = target_size % n_classes

    # min number of elements per sign, in the selected classes
    min_size_per_class = min(train.sign.value_counts())

    # If not enough samples, we reduce the sampling size to that value
    if min_size_per_class < target_size_per_class:
        print(
            f'    ⚠️ Total size smaller than requested, with {min_size_per_class} per sign instead of {target_size_per_class}')
        target_size_per_class = min_size_per_class
        # don't add extra samples in this case, we put min_size_per_sign for each sign
        remaining_samples = 0

    # Initiate the data before concatenation
    remaining_samples_added = 0
    train_balanced = pd.DataFrame()

    # For each selected sign category, adjust the number of samples to match the target size
    for class_ in include_classes:
        train_class = train[train['sign'] == class_]
        if remaining_samples_added < remaining_samples and len(train_class) > target_size_per_class:
            # add 1 additional sample to reach the exact total
            train_class = train_class.sample(
                target_size_per_class + 1, random_state=random_state)
            remaining_samples_added += 1
        else:
            train_class = train_class.sample(
                target_size_per_class, random_state=random_state)

        train_balanced = pd.concat(
            [train_balanced, train_class], ignore_index=True)

    # Shuffle the data
    train_balanced.sample(frac=1, random_state=random_state)

    return train_balanced, size


def filter_by_classes(train, n_classes):
    """
    Filters the training data based on the specified number of classes.

    Args:
        train (pandas.DataFrame): The training data.
        n_classes (int): The number of classes to include.

    Returns:
        tuple: A tuple containing the filtered training data and the list of included classes.
    """
    # Select the first n_classes from the glossary
    all_classes = load_glossary().sign
    include_classes = all_classes[:n_classes].to_list()
    train = train[train['sign'].isin(include_classes)]

    return train, include_classes


def count_frames(pq_file_path):
    """
    Count the number of unique frames and the number of frames from start to end in a parquet file.

    Parameters:
    - pq_file_path (str): Path to the parquet file.

    Returns:
    - pd.Series: A series containing the number of unique frames and the number of frames from start to end.
    """
    parquet_df = pd.read_parquet(pq_file_path)
    n_frames = parquet_df["frame"].nunique()
    n_frames2 = parquet_df["frame"].iloc[-1] - parquet_df["frame"].iloc[0] + 1
    return pd.Series([n_frames, n_frames2])


def load_frame_number_parquet(train, frame_count_csv_path=TRAIN_FRAME_CSV_PATH):
    """
    Enhances the input DataFrame by adding 'n_frames' and 'n_frames2' columns indicating the number of frames
    for each referenced parquet file. If '<filename>_frame.csv' exists at 'csv_path', loads DataFrame from it.

    Parameters:
    - train (pd.DataFrame): Input DataFrame with 'file_path' column containing parquet file paths.
    - frame_count_csv_path (str, optional): Directory path for '<filename>.csv'. Defaults to 'TRAIN_FRAME_CSV_PATH'.

    Returns:
    - pd.DataFrame: Enhanced DataFrame with 'n_frames' and 'n_frames2' columns. Loads from CSV if exists.

    Note:
    - Checks for '<filename>_frame.csv' existence. Calculates frames if not. Ensure 'TRAIN_DATA_DIR' is defined.
    - 'n_frames' is the number of unique frames in the parquet file.
    - 'n_frames2' is the number of frames from start to end in the parquet file.
    - If 'n_frames' and 'n_frames2' are not equal, a warning is printed indicating that the parquet file might have missing frames.
    """

    # Check if csv file already exists
    if not os.path.exists(frame_count_csv_path):
        tqdm.pandas(desc="Reading parquet files to count frames")
        train[['n_frames', 'n_frames2']
              ] = train['file_path'].progress_apply(count_frames)
        train_with_frame_count = train[[
            "sequence_id", "n_frames", "n_frames2"]]
        train_with_frame_count.to_csv(frame_count_csv_path, index=False)
        print(
            f" ✅ File with frame_parquet has been saved at : {frame_count_csv_path }")
        return train

    # if file exists, load it
    else:
        train_with_frame_count = pd.read_csv(frame_count_csv_path)
        print(f"    ℹ File with frames already exists, loaded matching 'sequence_id' rows.")
        train = pd.merge(train, train_with_frame_count,
                         how="left", on='sequence_id')
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


def unique_train_test_split(force_rewrite=False):
    """
    Splits the data into unique training and test sets.

    If the training and test data already exist, the function returns without performing any operations.

    Returns:
        None
    """
    if not force_rewrite and os.path.exists(TRAIN_TRAIN_CSV_PATH) and os.path.exists(TRAIN_TEST_CSV_PATH):
        return print(Fore.BLUE + "Train and test data already exist." + Style.RESET_ALL)

    if force_rewrite:
        print(Fore.RED + "Force rewrite is enabled. Overwriting the existing train and test data." + Style.RESET_ALL)

    test_size = 0.2

    print(Fore.BLUE + Style.BRIGHT +
          f"\nCreating unique test set with test_size = {test_size}" + Style.RESET_ALL)

    # Unique test set defined with constant random_state and parameters independent of the environment
    test_data = load_data_subset_csv(frac=test_size, noface=False, balanced=True,
                                     n_classes=250, n_frames=100, random_state=42, csv_path=TRAIN_CSV_PATH)

    print(Fore.BLUE + Style.BRIGHT + "\nCreating training set" + Style.RESET_ALL)

    all_data = load_data_subset_csv(frac=1, noface=False, balanced=False,
                                    n_classes=250, n_frames=None, random_state=42, csv_path=TRAIN_CSV_PATH)

    # take the difference between the two sets
    train_data = all_data[~all_data['sequence_id'].isin(test_data['sequence_id'])]

    total_len = len(all_data)
    train_len = len(train_data)
    test_len = len(test_data)

    train_ratio = train_len / total_len
    test_ratio = test_len / total_len

    print(Fore.BLUE + f"\nTotal loaded rows : {total_len} \
        \nTotal training rows : {train_len} ({train_ratio*100:.2f}%) \
        \nTotal test rows : {test_len} ({test_ratio*100:.2f}%)" + Style.RESET_ALL)

    train_data = train_data.drop(columns=['file_path'])
    test_data = test_data.drop(columns=['file_path'])

    train_data.to_csv(TRAIN_TRAIN_CSV_PATH, index=False)
    test_data.to_csv(TRAIN_TEST_CSV_PATH, index=False)

    print(Fore.BLUE +
          f"\nTrain and test data saved at {TRAIN_TRAIN_CSV_PATH} and {TRAIN_TEST_CSV_PATH}" + Style.RESET_ALL)

################################################################################
# LOAD VIDEOS (OTHER DATASET)
################################################################################

def load_video_list_json(video_list_json_path: str = WLASL_JSON_PATH, filter_glossary: bool = True, random_seed: int = 42) -> pd.DataFrame:
    """
    Reads the list of video paths from the specified file.

    Args:
        video_list_path (str): Path to the file containing the list of video paths.
        filter_glossary (bool): Whether to filter the glossary.
        random_seed (int): Seed for the random number generator.

    Returns:
        DataFrame: A DataFrame of video paths.
    """
    df = pd.read_json(video_list_json_path)

    # Explode the 'instances' column
    df_unstacked = df.explode('instances').reset_index(drop=True)

    # Normalize the 'instances' column containing dictionaries
    instances_df = pd.json_normalize(df_unstacked['instances'])

    # Combine the normalized 'instances' DataFrame with the original DataFrame
    videos_df = pd.concat([df_unstacked.drop(columns=['instances']), instances_df], axis=1)

    videos_df['video_path'] = videos_df['video_id'].apply(lambda row : os.path.join(WLASL_VIDEO_DIR, f"{row}.mp4"))

    videos_df['file_exists'] = videos_df['video_path'].apply(lambda row : Path(row).is_file())

    videos_df = videos_df [videos_df['file_exists']] # filter out videos that do not exist

    videos_df = videos_df.rename(columns={'gloss': 'sign'})

    videos_df = videos_df.drop(columns=['file_exists'])

    if filter_glossary:
        glossary = load_glossary()
        videos_df = videos_df[videos_df['sign'].isin(glossary['sign'])]

    videos_df = videos_df.sample(frac=1, random_state=random_seed) # shuffle to avoid having them grouped by sign

    videos_df = videos_df.reset_index(drop=True)

    return videos_df
