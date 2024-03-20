import os
import numpy as np

##################  VARIABLES  ##################

##################  CONSTANTS  #####################

BASE_DIR = os.path.dirname(os.path.dirname(__file__)) # go 2 levels up

TRAIN_DATA_DIR   = os.path.join(BASE_DIR, 'raw_data', 'asl-signs')
TRAIN_CSV_PATH   = os.path.join(TRAIN_DATA_DIR, 'train.csv')
LANDMARK_DIR     = os.path.join(TRAIN_DATA_DIR,'train_landmark_files')
LABEL_MAP_PATH   = os.path.join(TRAIN_DATA_DIR,'sign_to_prediction_index_map.json')

N_LANDMARKS_HAND    = 21
N_LANDMAKRS_POSE    = 33
N_LANDMARKS_FACE    = 468
N_LANDMARKS_ALL     = 543   # N_LANDMARKS_HAND * 2 +  N_LANDMAKRS_POSE + N_LANDMARKS_FACE
N_LANDMARKS_NO_FACE = 75    # N_LANDMAKRS_POSE + 2 * N_LANDMARKS_HAND
