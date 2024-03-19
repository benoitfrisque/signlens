import os
import numpy as np

##################  VARIABLES  ##################

##################  CONSTANTS  #####################

TRAIN_DATA_DIR   = os.path.join('..', '..', 'raw_data', 'asl-signs')
TRAIN_CSV_PATH   = os.path.join(TRAIN_DATA_DIR, 'train.csv')
LANDMARK_DIR     = os.path.join(TRAIN_DATA_DIR,'train_landmark_files')
LABEL_MAP_PATH   = os.path.join(TRAIN_DATA_DIR,'sign_to_prediction_index_map.json')
