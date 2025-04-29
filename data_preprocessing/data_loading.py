from data_preprocessing.preprocessing_configcessing_config import DEST_DIR, N_SPLITS, LOG_LEVEL
from data_preprocessing.logger import get_logger
import os
import tensorflow as tf
from functools import reduce
from tqdm import tqdm


logger = get_logger(__name__)
os.environ['LOG_LEVEL'] = LOG_LEVEL  

def load_data(batch_size=32, image_size=(256, 256)):

    logger.info(f"Loading data from {DEST_DIR}")
    test_folder = os.path.join(DEST_DIR, "test")
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        directory=test_folder,
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=image_size
    )
    logger.info(f"Loaded test dataset with {len(test_ds)} batches")
    logger.info(f"Test dataset shape: {test_ds.element_spec}")
    folds = [os.path.join(DEST_DIR, f"fold_{i}") for i in range(N_SPLITS)]
    
    train_folds = []
    for i in tqdm(range(N_SPLITS), desc="Loading folds"):
        logger.info(f"Loading fold {i} from {folds[i]}")
        fold_i = tf.keras.utils.image_dataset_from_directory(
            directory=folds[i],
            labels='inferred',
            label_mode='categorical',
            batch_size=batch_size,
            image_size=image_size
        )
        train_folds.append(fold_i)
    
    logger.info(f"Loaded {len(train_folds)} folds")    
    return train_folds, test_ds

