import os
import sys
import uuid
import pandas as pd
from data_utils import get_n_copies
from utils.io_utils import load_image, save_image, copy_and_rename_file
from logger import get_logger
from config import OVERSAMPLE, LABEL_COL

logger = get_logger(__name__)

def rename_and_save(df: pd.DataFrame, origin_root: str, dest_root: str, oversample: bool = True):
    logger.info(f"oversampling: {oversample}")
    logger.info(f"OVERSAMPLE: {OVERSAMPLE}")
    
    if oversample == True:
        # not implemented yet
        logger.warning("Oversampling is not implemented yet, set oversample=False to skip this step")
        return
    logger.debug(f"origin_root: {origin_root}-- dest_root: {dest_root}")
    logger.debug(f"Dataframe shape: {df.shape}")
    logger.debug(f"Dataframe columns: {df.columns}")
    logger.debug(f"OVERSAMPLE: {OVERSAMPLE}")
    logger.debug("Starting oversampling and saving process")
    df['copies'] = get_n_copies(df[f'{LABEL_COL}']).reindex(df[f'{LABEL_COL}']).values

    for index_row, row in df.iterrows():
        label = row[f'{LABEL_COL}']
        src = os.path.join(origin_root, row['file_path'])
        
        if not os.path.isfile(src):
            logger.warning(f"Source file does not exist: {src}")
            continue
        
        if not oversample:
            logger.debug(f"Copying without oversampling for {label}")
            dest_dir = os.path.join(dest_root, label)
            os.makedirs(dest_dir, exist_ok=True)
            new_filename = f"{label}_{str(index_row).zfill(6)}00.jpg"
            copy_and_rename_file(src, dest_dir, new_filename)
            continue
        if row['copies'] <= 1:
            logger.debug(f"No oversampling needed for {label}, copies: {row['copies']}")
            new_filename = f"{label}_{str(index_row).zfill(6)}00.jpg"
            copy_and_rename_file(src, dest_root, new_filename)
            continue
        logger.debug(f"Oversampling {label}, copies-: {row['copies']}")
        # Oversample the image
        # Assuming the oversampling process involves creating multiple copies
        new_filename = f"{label}_{str(index_row).zfill(6)}00.jpg"
        copy_and_rename_file(src, dest_root, new_filename)
        continue
        
        
        
    """        if row['copies'] < 5:
            for index_copy in range(row['copies']):
                dest_dir = os.path.join(dest_root, label)
                os.makedirs(dest_dir, exist_ok=True)
                new_filename = f"{label}_{str(index_row).zfill(6)}{str(index_copy).zfill(2)}.jpg"
                copy_and_rename_file(src, dest_dir, new_filename)
                logger.debug(f"Saved oversampled image: {new_filename}")
        
        for _ in range(row['copies']):
            dest_dir = os.path.join(dest_root, label)
            os.makedirs(dest_dir, exist_ok=True)
            new_filename = f"{label}_{uuid.uuid4().hex}.jpg"
            image = load_image(src)
            save_image(os.path.join(dest_dir, new_filename), image)
            logger.debug(f"Saved oversampled image: {new_filename}")
    """
