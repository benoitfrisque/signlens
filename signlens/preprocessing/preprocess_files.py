from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm  # Import tqdm for the progress bar

from signlens.params import *

def load_subset_data(frac=1,noface=True,balanced=False):
    '''
    Load subset of data based on the fraction of the original dataset.
    Uses the noface dataset if noface is set to True.
    Also balances the dataset if balanced is set to True.

    Returns:
        DataFrame: subset of the original dataset
    '''
    train = pd.read_csv(TRAIN_CSV_PATH)

    # replace path train_landmark_files with train_landmark_files_noface
    if noface:
        train[['path']] = train[['path']].apply(lambda x: x.str.replace(
            'train_landmark_files', 'train_landmark_files_noface'))
    train['file_path'] = str(TRAIN_DATA_DIR) + '/' + train['path']

    # random subset of the data by percent
    if frac < 1:
        train = train.sample(frac=frac)

    # balance the data
    if balanced:
        train = balance_data(train)

    return train

def balance_data(train):
    '''
    Balances the dataset to the smallest class.
    '''
    # smallest sign
    min_sign_count = train['sign'].value_counts().sort_values().min()

    # select random sample of min_sign_count
    return train.groupby('sign').apply(lambda x: x.sample(min_sign_count)).reset_index(drop=True)

def load_relevant_data_subset(pq_path,noface=True):
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

    # ROWS_PER_FRAME = 543  from documentation
    frame_rows = 543
    if noface:
        # Exclude rows where 'type' is 'face'
        data = data[data['type'] != 'face']
        # N_LANDMARKS_NOFACE 75
        frame_rows = 75
    data = data.drop(columns=['type'])
    data_columns = data_columns[:-1]

    # Replace NaN values with 0
    data.fillna(0, inplace=True)
    n_frames = int(len(data) / frame_rows)
    n_dim=len(data_columns)
    data = data.values.reshape(n_frames, frame_rows, n_dim)
    return data.astype(np.float32)
