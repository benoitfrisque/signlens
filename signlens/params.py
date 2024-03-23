import os

##################  VARIABLES  ##################
 # Number of classes to analyze
if os.environ.get("NUM_CLASSES") == 'all':
    NUM_CLASSES = 250
else:
    NUM_CLASSES = int(os.environ.get("NUM_CLASSES"))

# Max number of frames in a sequence. Used to pad sequences, and also filter out sequences
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN"))

# Percentage of the data to load
DATA_FRAC = float(os.environ.get("DATA_FRAC"))

##################  CONSTANTS  #####################
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'raw_data')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'processed_data')

# Landmark directories
TRAIN_DATA_DIR          = os.path.join(RAW_DATA_DIR, 'asl-signs')
TRAIN_CSV_PATH          = os.path.join(TRAIN_DATA_DIR, 'train.csv')

LANDMARK_DIR            = os.path.join(TRAIN_DATA_DIR,'train_landmark_files')
LABEL_MAP_PATH          = os.path.join(TRAIN_DATA_DIR,'sign_to_prediction_index_map.json')

# Video directories
WLASL_DIR = os.path.join(RAW_DATA_DIR, 'WLASL')
WLASL_JSON_PATH = os.path.join(WLASL_DIR, 'WLASL_v0.3.json')
WLASL_VIDEO_DIR = os.path.join(WLASL_DIR, 'videos')

# Processed data dir
TRAIN_TRAIN_CSV_PATH    = os.path.join(PROCESSED_DATA_DIR , 'train_train.csv')
TRAIN_TEST_CSV_PATH     = os.path.join(PROCESSED_DATA_DIR , 'train_test.csv')
TRAIN_FRAME_CSV_PATH    = os.path.join(PROCESSED_DATA_DIR , 'train_frame.csv')
GLOSSARY_CSV_PATH       = os.path.join(PROCESSED_DATA_DIR, 'glossary.csv')


TRAIN_OUTPUT_DIR        = os.path.join(BASE_DIR, 'training_outputs')

# Mediapipe landmarks
N_LANDMARKS_HAND    = 21
N_LANDMAKRS_POSE    = 33
N_LANDMARKS_FACE    = 468
N_LANDMARKS_ALL     = 543   # N_LANDMARKS_HAND * 2 +  N_LANDMAKRS_POSE + N_LANDMARKS_FACE
N_LANDMARKS_NO_FACE = 75    # N_LANDMAKRS_POSE + 2 * N_LANDMARKS_HAND
