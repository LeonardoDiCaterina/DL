import tensorflow as tf
import pandas as pd
import os
from tqdm import tqdm

#from data_preprocessing.preprocessing_config import LABEL_COL, OVERCLASS_COL
from .preprocessing_config import DEST_DIR, N_SPLITS, LOG_LEVEL,LABEL_COL, OVERCLASS_COL,BATCH_SIZE, IMAGE_SIZE
from data_preprocessing.logger import get_logger


def lookup_class_overclass_generator():
    """
    This function creates a lookup table for family and phylum.
    It takes two lists as input: families and phylums.
    It returns a lookup table that maps each family to its corresponding phylum.
    """
    if OVERCLASS_COL is None:
        raise ValueError("OVERCLASS_COL is not set. Please set it to the name of the column containing the overclass labels.")
    
    # Read the CSV file
    df = pd.read_csv('data/downloaded_dataset/lookup_table.csv')
    label_class = df[LABEL_COL].tolist()
    label_overclass = df[OVERCLASS_COL].tolist()
    
    init = tf.lookup.KeyValueTensorInitializer(
        keys=label_class,
        values=label_overclass,
        key_dtype=tf.string,
        value_dtype=tf.string
    )

    default_value = "unknown"
    # Static hash table
    family_phylum_lookup = tf.lookup.StaticHashTable(
        initializer=init,
        default_value=default_value
    )
    return family_phylum_lookup


logger = get_logger(__name__)
os.environ['LOG_LEVEL'] = LOG_LEVEL  

def load_data(batch_size= BATCH_SIZE, image_size= IMAGE_SIZE):

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


def unify_and_rebatch(datasets, batch_size = BATCH_SIZE, shuffle_buffer=1000):
    """
    This function takes a list of datasets and unifies them into a single dataset.
    It also batches the dataset and shuffles it.

    Args:
        datasets (List[]): List of datasets to be unified.
        batch_size (int, optional): _description_. Defaults to BATCH_SIZE.
        shuffle_buffer (int, optional): _description_. Defaults to 1000.

    Returns:
        _type_: _description_
    """
    
    #for i, ds in enumerate(datasets):
    #    for x, y in ds.take(1):
    #        if len(x.shape) != 4:
    #            raise ValueError(f"Dataset {i} has unexpected input shape: {x.shape}")
    #        if len(y.shape) != 2:
    #            raise ValueError(f"Dataset {i} has unexpected label shape: {y.shape}")
    
    unbatched = [ds.unbatch() for ds in datasets]
    merged_dataset = unbatched[0]
    for ds in unbatched[1:]:
        merged_dataset = merged_dataset.concatenate(ds)
    return merged_dataset.shuffle(shuffle_buffer).batch(batch_size).prefetch(tf.data.AUTOTUNE)
