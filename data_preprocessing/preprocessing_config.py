import os
DATA_DIR = 'data/downloaded_dataset'
DEST_DIR = 'data/rearranged'
CSV_PATH = f'{DATA_DIR}/metadata.csv'
N_SPLITS = 5
TEST_SIZE = 0.2 # it's a ratio therefore has to be between 0 and 1
OVERSAMPLE = False
LOG_LEVEL = 'INFO'
LABEL_COL = 'family'