import os
import uuid
import pandas as pd
from data_utils import get_n_copies
from utils.io_utils import load_image, save_image
from logger import get_logger

logger = get_logger(__name__)

def oversample_and_save(df: pd.DataFrame, origin_root: str, dest_root: str):
    logger.info("Starting oversampling and saving process")
    df['copies'] = get_n_copies(df['phylum']).reindex(df['phylum']).values

    for _, row in df.iterrows():
        label = row['phylum']
        src = os.path.join(origin_root, row['file_path'])

        for _ in range(row['copies']):
            dest_dir = os.path.join(dest_root, label)
            os.makedirs(dest_dir, exist_ok=True)
            new_filename = f"{label}_{uuid.uuid4().hex}.jpg"
            image = load_image(src)
            save_image(os.path.join(dest_dir, new_filename), image)
            logger.debug(f"Saved oversampled image: {new_filename}")