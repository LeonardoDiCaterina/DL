import tensorflow as tf
import pandas as pd
from data_preprocessing.preprocessing_config import LABEL_COL, OVERCLASS_COL
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