import pandas as pd
from pathlib import Path
from tqdm import tqdm

from signlens.params import *
from signlens.preprocessing.preprocess import filter_relevant_landmarks_and_coordinates_df


def get_light_parquet_path(parquet_file_path, dirname_suffix, filename_suffix):
    """
    Generates the path for a new lightweight parquet file.

    Args:
        parquet_file_path (str or Path): Path to the input parquet file.
        dirname_suffix (str): Suffix to append to the parent directory's name.
        filename_suffix (str): Suffix to append to the stem of the file name.

    Returns:
        Path: Path to the new lightweight parquet file.
    """
    file_path = Path(parquet_file_path)
    parent_dir = file_path.parent
    parent_parent_dir = parent_dir.parent

    new_parent_parent_dir = parent_parent_dir.parent / (parent_parent_dir.name + dirname_suffix)
    new_parent_parent_dir.mkdir(parents=True, exist_ok=True)
    new_parent_dir = new_parent_parent_dir / parent_dir.name
    new_parent_dir.mkdir(parents=True, exist_ok=True)

    new_parquet_file_name = file_path.stem + filename_suffix + file_path.suffix
    return new_parent_dir / new_parquet_file_name


def write_light_parquet_files(parquet_file_list, landmark_types_to_remove, dirname_suffix, filename_suffix=""):
    """
    Writes filtered landmarks to new lightweight parquet files for a list of input files.

    Args:
        parquet_file_path (str or Path): Path to the input parquet file.
        landmark_types_to_remove (list of str): List of landmark types to be removed.
        dirname_suffix (str): Suffix to append to the parent directory's name.
        filename_suffix (str, optional): Suffix to append to the stem of the file name. Defaults to "".
    """
    with tqdm(total=len(parquet_file_list), desc="Writing files") as pbar:
        for parquet_file_path in parquet_file_list:
            new_parquet_file_path = get_light_parquet_path(parquet_file_path, dirname_suffix, filename_suffix)

            if not new_parquet_file_path.exists():
                landmarks_df = pd.read_parquet(parquet_file_path)
                filtered_landmarks = filter_relevant_landmarks_and_coordinates_df(landmarks_df, noface=True)
                filtered_landmarks.to_parquet(new_parquet_file_path)

            pbar.update(1)



if __name__ == "__main__":
    train = pd.read_csv(TRAIN_CSV_PATH)
    train['file_path'] = str(TRAIN_DATA_DIR) + '/' + train['path']
    user_input = input('You are about to convert all files to lightweight files. Do you wish to continue? [Y/N] ')
    if user_input.lower() in ('y', 'yes'):
        write_light_parquet_files(train['file_path'], landmark_types_to_remove=['face'], dirname_suffix="_noface", filename_suffix="")
