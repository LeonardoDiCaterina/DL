import os
DATA_DIR = 'data/downloaded_dataset'
DEST_DIR = 'data/rearranged'
CSV_PATH = f'{DATA_DIR}/metadata.csv'
N_SPLITS = 5
TEST_SIZE = 0.2 # it's a ratio therefore has to be between 0 and 1
OVERSAMPLE = False
BATCH_SIZE = 32
IMAGE_SIZE = (256, 256)
LOG_LEVEL = 'INFO'
LABEL_COL = 'family'
OVERCLASS_COL = 'phylum' # set to None if not needed