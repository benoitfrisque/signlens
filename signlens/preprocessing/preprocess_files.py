from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm  # Import tqdm for the progress bar
import os
from colorama import Fore, Style
from signlens.params import *

def load_subset_data(frac=1.0,noface=True,balanced=False):
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

def load_frame_number_parquet(train, csv_path=TRAIN_DATA_DIR):
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

    csv_filename=csv_path+"/train_frame.csv"
    # Check if csv file already exist
    if not os.path.exists(csv_filename):
        # If not existing create the column and save the data frame
        for i in range(len(train)):
            df = pd.read_parquet(train.loc[i, "file_path"]).copy()
            train.at[i, "frame_parquet"] = df["frame"].iloc[-1] - df["frame"].iloc[0] + 1
        train.to_csv(csv_filename, index=False)
        print(f" ✅ File has been saved at : {csv_filename}")
    else:
        train = pd.read_csv(csv_filename)
        print("File already exist")

    return train

def filter_out_parquet_frame(df,n_frame=100):
    """
    Filters the DataFrame by the 'frame_parquet' column to include only rows where the number of frames is less than or equal to the specified threshold.
    This function is intended to be used on DataFrames that have already been processed by the 'load_frame_number_parquet'
    function, which adds the 'frame_parquet' column indicating the number of frames for each parquet file.

    Parameters:
    - df (pd.DataFrame): The DataFrame to filter. It must include a 'frame_parquet' column.
    - n_frame (int): The maximum number of frames allowed for a row to be included in the filtered DataFrame.

    Returns:
    - pd.DataFrame: A new DataFrame consisting of rows from the original DataFrame where the 'frame_parquet' value is less than or equal to 'frame_threshold'.
    The index of the DataFrame will be reset.

    """
    return df[df["frame_parquet"]<=n_frame].reset_index(drop=True)
