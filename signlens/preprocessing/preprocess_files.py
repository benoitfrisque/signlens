import pandas as pd
from pathlib import Path
from tqdm import tqdm  # Import tqdm for the progress bar

from signlens.params import *

def load_subset_data(frac=1,noface=True,balanced=False):
    '''
    Load subset of data based on the fraction of the original dataset.
    Uses only the noface dataset if noface is set to True.
    Also balances the dataset if balanced is set to True.
    '''
    train = pd.read_csv(TRAIN_CSV_PATH)
    train['file_path'] = str(TRAIN_DATA_DIR) + '/' + train['path']
    train = train.sample(frac=frac)

    # replace path train_landmark_files with train_landmark_files_noface
    if noface:
        train[['path']] = train[['path']].apply(lambda x: x.str.replace(
            'train_landmark_files', 'train_landmark_files_noface'))

    # random subset of the data by percent
    FRAC = 0.2
    train = train.sample(frac=FRAC)

    return train
