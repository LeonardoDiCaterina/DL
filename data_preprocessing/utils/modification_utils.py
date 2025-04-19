import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots, show
import cv2 as cv
from PIL import Image, ImageEnhance, ImageFilter
import random
import tensorflow as tf  # Only used for checking tensor type


def to_pil_image(img):
    """
    Ensures the image is a PIL.Image.
    Accepts PIL.Image, NumPy array, or TensorFlow tensor.
    """
    if isinstance(img, tf.Tensor):
        img = img.numpy()
    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
    return img


def horizontal_flip(img):
    """
    Applies horizontal flip to the image.
    """
    img = to_pil_image(img)
    return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)


def contrast_enhancement(img):
    """
    Applies contrast enhancement to the image.
    """
    img = to_pil_image(img)
    factor = random.uniform(0.5, 2.0)
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)


def brightness_enhancement(img):
    """
    Applies brightness enhancement to the image.
    """
    img = to_pil_image(img)
    factor = random.uniform(0.5, 1.5)
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


def color_enhancement(img):  # saturation
    """
    Applies color enhancement to the image.
    """
    img = to_pil_image(img)
    factor = random.uniform(0.5, 1.5)
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(factor)


def sharpening(img):
    """
    Applies sharpening to the image.
    """
    img = to_pil_image(img)
    factor = random.uniform(0.5, 2.0)
    enhancer = ImageEnhance.Sharpness(img)
    return enhancer.enhance(factor)


def filter_edge_enhance(img):
    """
    Applies edge enhancement to the image.
    """
    img = to_pil_image(img)
    return img.filter(ImageFilter.EDGE_ENHANCE)


def filter_edge_enhance_more(img):
    """
    Applies strong edge enhancement to the image.
    """
    img = to_pil_image(img)
    return img.filter(ImageFilter.EDGE_ENHANCE_MORE)


def combination_transformation(img):
    """
    Applies a combination of transformations to the image.
    """
    img = to_pil_image(img)
    transformations_list = [
        horizontal_flip,
        contrast_enhancement,
        brightness_enhancement,
        color_enhancement,
        sharpening,
        filter_edge_enhance,
        filter_edge_enhance_more,
    ]

    # Random boolean array indicating which transforms to apply
    random_list = np.random.randint(0, 2, len(transformations_list), dtype=bool)

    for i in range(len(transformations_list)):
        if random_list[i]:
            img = transformations_list[i](img)
    return img


def distinct_multiple_transformations(img, n_children=5):
    """
    Applies n distinct transformations to the image.
    """
    img = to_pil_image(img)
    images = []

    for _ in range(n_children):
        child = combination_transformation(img)
        images.append(child)

    return images
