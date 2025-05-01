import pandas as pd
from .preprocessing_config import CSV_PATH, DATA_DIR, DEST_DIR, N_SPLITS, TEST_SIZE, OVERSAMPLE,LOG_LEVEL, LABEL_COL
from .splitting import create_split
from .augmentation import save_to_split_to_directory
from .logger import get_logger
import os

logger = get_logger(__name__)
os.environ['LOG_LEVEL'] = LOG_LEVEL  



if __name__ == '__main__':
    # if DEST_DIR exists, remove it
    if os.path.exists(DEST_DIR):
        logger.info(f"Removing existing directory {DEST_DIR}")
        
    # read the CSV
    logger.info("Starting preprocessing pipeline")
    df = pd.read_csv(CSV_PATH)[['file_path', LABEL_COL]]
    logger.info(f"Loaded metadata with {len(df)} entries")
    
    folds, test_df = create_split(df['file_path'], df[LABEL_COL], n_folds=N_SPLITS, test_ratio=TEST_SIZE, random_state=42)
    logger.info("Data splitting completed")
    
    for index_dir,fold in enumerate(folds):
        # destination directory for each fold
        fold_dir = os.path.join(DEST_DIR, f'fold_{index_dir}')
        logger.info(f"Creating directory {fold_dir}")
        os.makedirs(fold_dir, exist_ok=True)
        logger.info(f"Renaming and saving the fold {index_dir}")
        # fold a dataframe with columns ['file_path', LABEL_COL]
        save_to_split_to_directory(fold,DATA_DIR, fold_dir, oversample=OVERSAMPLE)
        fold_metadata_name = os.path.join(fold_dir, f'metadata_{str(index_dir).zfill(2)}.csv')
        fold_df = pd.DataFrame(fold, columns=['file_path', LABEL_COL])
        fold_df.to_csv(fold_metadata_name, index=False)
        logger.info(f"Saved fold metadata to {fold_metadata_name}")
        logger.info(f"Saved fold {index_dir} to {fold_dir}")
    
    test_dir = os.path.join(DEST_DIR, 'test')
    logger.info(f"Creating directory {test_dir}")
    os.makedirs(test_dir, exist_ok=True)
    logger.info("Renaming and saving test set")
    # test_df is a list of tuples (path, label)
    fold_dir = os.path.join(DEST_DIR, 'test')
    save_to_split_to_directory(test_df, DATA_DIR, fold_dir, oversample=False)
    fold_metadata_name = os.path.join(test_dir, 'metadata_test.csv')
    test_df.to_csv(fold_metadata_name, index=False)
    logger.info(f"Saved test metadata to {fold_metadata_name}")

    logger.info(f"Saved test set to {test_dir}")
    logger.info("Preprocessing pipeline completed")
    
