import pandas as pd
from config import CSV_PATH, DATA_DIR, DEST_DIR
from splitting import create_split
from augmentation import oversample_and_save
from logger import get_logger

logger = get_logger(__name__)

if __name__ == '__main__':
    logger.info("Starting preprocessing pipeline")
    df = pd.read_csv(CSV_PATH)[['file_path', 'phylum']]
    logger.info(f"Loaded metadata with {len(df)} entries")
    
    folds, test_df = create_split(df['file_path'], df['phylum'])
    logger.info("Data splitting completed")

    oversample_and_save(folds[0], DATA_DIR, DEST_DIR)
    logger.info("Oversampling completed")