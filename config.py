import os

# ── base paths ──────────────────────────────────────────────────
# PROJECT_ROOT is the folder this file lives in — everything else is relative to it.
# DATASET_PATH is the only thing you need to change for your machine.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def _find_daisee():
    """Auto-discover the DAiSEE dataset. Checks DAISEE_PATH env var first,
    then walks the project tree looking for the Labels/ and DataSet/ folders."""
    env = os.environ.get('DAISEE_PATH')
    if env and os.path.isdir(env):
        return env

    for root, dirs, _ in os.walk(PROJECT_ROOT):
        if 'Labels' in dirs and 'DataSet' in dirs:
            labels_dir = os.path.join(root, 'Labels')
            if os.path.isfile(os.path.join(labels_dir, 'TrainLabels.csv')):
                return root
    raise FileNotFoundError(
        "DAiSEE dataset not found. Place it anywhere inside the project folder "
        "or set the DAISEE_PATH environment variable."
    )

DATASET_PATH = _find_daisee()

PROJECT_PATH = os.path.join(PROJECT_ROOT, 'projects')   # saved models
SAVE_DIR     = os.path.join(PROJECT_ROOT, 'processed_data')  # tfrecords

TRAIN_PATH  = os.path.join(DATASET_PATH, 'DataSet', 'Train')
VAL_PATH    = os.path.join(DATASET_PATH, 'DataSet', 'Validation')
TEST_PATH   = os.path.join(DATASET_PATH, 'DataSet', 'Test')
LABELS_PATH = os.path.join(DATASET_PATH, 'Labels')

def _latest_model(project_dir):
    """Return the most recently saved .keras model, or None if none exist."""
    if not os.path.isdir(project_dir):
        return None
    models = sorted(
        (f for f in os.listdir(project_dir) if f.endswith('.keras')),
        reverse=True  # newest timestamp first (filenames are YYYYMMDD_HHMM)
    )
    return os.path.join(project_dir, models[0]) if models else None

MODEL_PATH      = _latest_model(PROJECT_PATH)
BEST_MODEL_PATH = MODEL_PATH

IMG_SIZE   = (224, 224) # size of the input images
BATCH      = 16 # batch size for training
N_CLASSES  = 2 # number of classes
LABELS     = ['Not Engaged', 'Engaged'] # labels for the classes
EPOCHS     = 40 # number of epochs for training
LR         = 1e-5 # learning rate
PATIENCE   = 4 # patience for early stopping and learning rate reduction
FRAME_STEP = 4 # step size for frame sampling 
MIN_FRAMES = 5 # minimum number of frames required for a video to be included in the dataset