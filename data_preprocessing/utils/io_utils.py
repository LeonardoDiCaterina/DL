import tensorflow as tf
import shutil
import os
from logger import get_logger

logger = get_logger(__name__)

def load_image(filepath: str) -> tf.Tensor:
    logger.debug(f"Loading image: {filepath}")
    image = tf.io.read_file(filepath)
    return tf.image.decode_jpeg(image, channels=3)

def save_image(dest_path: str, image: tf.Tensor):
    logger.debug(f"Saving image to: {dest_path}")
    encoded = tf.image.encode_jpeg(image)
    tf.io.write_file(dest_path, encoded)
    
def copy_and_rename_file(source_path, destination_dir, new_filename):
    """
    Copies a file from source_path to destination_dir, renaming it to new_filename.
    
    Args:
        source_path (str): Path to the original file.
        destination_dir (str): Folder where the file should be copied.
        new_filename (str): New name for the copied file (with extension).
    """
    if not os.path.isfile(source_path):
        print(f"Source file does not exist: {source_path}")
        return
    
    os.makedirs(destination_dir, exist_ok=True)  # Create destination if it doesn't exist

    destination_path = os.path.join(destination_dir, new_filename)
    shutil.copy2(source_path, destination_path)  # copy2 preserves metadata
    logger.debug(f"Copied {source_path} to {destination_path}")