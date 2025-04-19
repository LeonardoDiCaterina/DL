import os
import sys
import uuid
import pandas as pd
from data_utils import get_n_copies
from utils.io_utils import load_image, save_image, copy_and_rename_file
from utils.modification_utils import distinct_multiple_transformations
from logger import get_logger
from config import OVERSAMPLE, LABEL_COL

logger = get_logger(__name__)




def save_to_split_to_directory(df: pd.DataFrame, origin_root: str, dest_root: str, oversample: bool = True):
    
    logger.info("Starting oversampling and saving process")
    logger.info(f"origin_root: {origin_root} --> dest_root: {dest_root}")
    logger.debug(f"Dataframe shape: {df.shape}")
    logger.debug(f"Dataframe columns: {df.columns}")
    logger.debug(f"OVERSAMPLE: {OVERSAMPLE}")

    label_counts = get_n_copies(df[LABEL_COL])
    aligned_copies = label_counts.reindex(df[LABEL_COL])

    n_missing = aligned_copies.isnull().sum()
    if n_missing > 0:
        logger.warning(f"{n_missing} label(s) had no matching copy count. Filling missing values with 0.")

    aligned_copies = aligned_copies.fillna(0).astype(int)
    df['copies'] = aligned_copies.values
    
    for index_row, row in df.iterrows():
        label = row[f'{LABEL_COL}']
        # src is ehere the original image is stored
        src = os.path.join(origin_root, row['file_path'])
        
        if not os.path.isfile(src):
            # not good if the image does not exist
            logger.warning(f"Source file does not exist: {src}")
            continue
        
        logger.debug(f"Copying without oversampling for {label}")
        dest_dir = os.path.join(dest_root, label)
        # if the directory named after the label does not exist I create it
        os.makedirs(dest_dir, exist_ok=True)
        # the new filename is the label + the index of the row in the dataframe in 6 digits
        # and 00 at the end to allow for the copies
        new_filename = f"{label}_{str(index_row).zfill(6)}00.jpg"
        # to optimise the copy I do not upload the image to memory but copy it directly
        # it's a potential risk so I put it in a try except block not to crash the program
        try:
            # Copy the original file to the destination directory
            copy_and_rename_file(src, dest_dir, new_filename)
        except Exception as e:
            logger.error(f"Error copying file {src} to {dest_dir}: {e}")
            continue
        
        
        # I check if I have to oversample the image
        n_children = 0
        if OVERSAMPLE == True:
            # n_children has to be at least 5 (maybe I can parameterize it)
            n_children = min(5, row['copies'])
    
        if n_children > 0:
            image = load_image(src)
            # Apply distinct transformations
            transformed_images = distinct_multiple_transformations(image, n_children)
            # Save the transformed images
            for index_copy in range(n_children):
                new_filename = f"{label}_{str(index_row).zfill(6)}{str(index_copy).zfill(2)}.jpg"
                try:
                    save_image(os.path.join(dest_dir, new_filename), transformed_images[index_copy])
                except Exception as e:
                    logger.error(f"Error saving transformed image {new_filename}: {e}")
                    continue
                logger.debug(f"Saved oversampled image: {new_filename}")
            
    logger.info("Oversampling and saving process completed")

