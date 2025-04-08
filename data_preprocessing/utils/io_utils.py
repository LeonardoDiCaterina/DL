import tensorflow as tf
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