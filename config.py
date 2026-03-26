DATASET_PATH = r'/mnt/c/Users/raysh/OneDrive - University of Reading (1)/DAiSEE-PROJ/DAiSEE' # path to the dataset
PROJECT_PATH = '/home/raysh/DAiSEE-PROJ/projects' # path to save models and logs

TRAIN_PATH  = f'{DATASET_PATH}/DataSet/Train' # path to training data
VAL_PATH    = f'{DATASET_PATH}/DataSet/Validation' # path to validation data
TEST_PATH   = f'{DATASET_PATH}/DataSet/Test' # path to test data
LABELS_PATH = f'{DATASET_PATH}/Labels' # path to labels

SAVE_DIR   = '/home/raysh/DAiSEE-PROJ/processed_data' # path to save processed data
MODEL_PATH = '/home/raysh/DAiSEE-PROJ/projects/model_20260319_1747.keras' # path to a pre-trained model
BEST_MODEL_PATH = '/home/raysh/DAiSEE-PROJ/projects/model_20260313_1941.keras' # path to save the best model (currently i have presaved this to 1941 model)

IMG_SIZE   = (224, 224) # size of the input images
BATCH      = 16 # batch size for training
N_CLASSES  = 2 # number of classes
LABELS     = ['Not Engaged', 'Engaged'] # labels for the classes
EPOCHS     = 40 # number of epochs for training
LR         = 1e-5 # learning rate
PATIENCE   = 4 # patience for early stopping and learning rate reduction
FRAME_STEP = 4 # step size for frame sampling 
MIN_FRAMES = 5 # minimum number of frames required for a video to be included in the dataset