import pandas as pd
from pathlib import Path
from tqdm import tqdm  # Import tqdm for the progress bar

from signlens.params import *


def filter_relevant_landmarks(parquet_file_path):
    """
    Filter out face landmarks from parquet file.

    Args:
        file_path (str or Path): Path to the input parquet file.

    Returns:
        DataFrame: DataFrame containing filtered landmarks.
    """
    # Read the parquet file containing landmarks
    landmarks = pd.read_parquet(parquet_file_path)
    # Filter out rows where the type is not 'face'
    filtered_landmarks = landmarks[landmarks['type'] != 'face']
    return filtered_landmarks

def get_light_parquet_path(parquet_file_path, dirname_suffix="_light", filename_suffix=""):
    """
    Generate the path for the new light parquet file.

    Args:
        parquet_file_path (str or Path): Path to the input parquet file.
        dirname_suffix (str, optional): Suffix to append to the parent directory's name. Defaults to "_light".
        filename_suffix (str, optional): Suffix to append to the stem of the file name. Defaults to "".

    Returns:
        Path: Path to the new light parquet file.
    """
    file_path = Path(parquet_file_path)
    parent_dir = file_path.parent
    parent_parent_dir = parent_dir.parent

    # Create a new directory with the appended suffix, if it doesn't exist
    new_parent_parent_dir = parent_parent_dir.parent / (parent_parent_dir.name + dirname_suffix)
    new_parent_parent_dir.mkdir(parents=True, exist_ok=True)
    new_parent_dir = new_parent_parent_dir / parent_dir.name
    new_parent_dir.mkdir(parents=True, exist_ok=True)

    # Generate the new file name with the appended suffix
    new_parquet_file_name = file_path.stem + filename_suffix + file_path.suffix
    return new_parent_dir / new_parquet_file_name

def write_light_parquet_file(parquet_file_path, dirname_suffix="_light", filename_suffix=""):
    """
    Write filtered landmarks to a new light parquet file.

    Args:
        parquet_file_path (str or Path): Path to the input parquet file.
        dirname_suffix (str, optional): Suffix to append to the parent directory's name. Defaults to "_light".
        filename_suffix (str, optional): Suffix to append to the stem of the file name. Defaults to "".
    """
    # Generate the path for the new light parquet file
    new_parquet_file_path = get_light_parquet_path(parquet_file_path, dirname_suffix, filename_suffix)

    if not new_parquet_file_path.exists():
         # Filter relevant landmarks from the input parquet file
        filtered_landmarks = filter_relevant_landmarks(parquet_file_path)
        # Write the filtered landmarks to the new parquet file
        filtered_landmarks.to_parquet(new_parquet_file_path)

def write_light_parquet_files(parquet_file_list, dirname_suffix="_light", filename_suffix=""):
    """
    Write filtered landmarks to new light parquet files for a list of input files.

    Args:
        parquet_file_list (list of str or Path): List of paths to the input parquet files.
        dirname_suffix (str, optional): Suffix to append to the parent directory's name. Defaults to "_light".
        filename_suffix (str, optional): Suffix to append to the stem of the file name. Defaults to "".
    """
    # Initialize tqdm with the total number of files
    with tqdm(total=len(parquet_file_list), desc="Writing files") as pbar:
        # Iterate through each parquet file in the list
        for parquet_file_path in parquet_file_list:
            # Write the light parquet file for the current input file
            write_light_parquet_file(parquet_file_path, dirname_suffix, filename_suffix)
            # Update the progress bar for each file processed
            pbar.update(1)

if __name__ == "__main__":
    train = pd.read_csv(TRAIN_CSV_PATH)
    train['file_path'] = str(TRAIN_DATA_DIR) + '/' + train['path']
    user_input = input('You are about to convert all files to light files. Do you wish to continue? [Y/N] ')
    if user_input.lower() in ('y', 'yes'):
        write_light_parquet_files(train['file_path'], dirname_suffix="_light", filename_suffix="")
