import os
from config import *
from preprocessing import FacePreprocessor, load_labels, plot_distribution, load_or_process

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs('outputs/previews', exist_ok=True)
os.makedirs('outputs/plots', exist_ok=True)

# another .py that runs preprocessing.py basically
if __name__ == '__main__':
    train_df, val_df, test_df = load_labels()
    plot_distribution(train_df)

    preprocessor = FacePreprocessor()

    load_or_process(TRAIN_PATH, train_df, SAVE_DIR, 'train', preprocessor, preview=True)
    load_or_process(VAL_PATH,val_df, SAVE_DIR, 'val',preprocessor, preview=False)
    load_or_process(TEST_PATH, test_df,SAVE_DIR, 'test',preprocessor, preview=False)

    print(f"Done. Saved to {SAVE_DIR}")