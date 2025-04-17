import numpy as np
import random
from PIL import Image, ImageEnhance, ImageFilter



def horizontal_flip(img):
    """
    Applies horizontal flip to the image.
    """
    
    return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

def contrast_enhancement(img):
    """
    Applies contrast enhancement to the image.
    """
    factor = random.uniform(0.5, 2.0)
        
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)

def brightness_enhancement(img):
    """
    Applies brightness enhancement to the image.
    """
    # random factor between 0.5 and 1.5
    factor = random.uniform(0.5, 1.5)
    # 0.5 = 50% darker, 1.5 = 50% brighter
    # 1.0 = original brightness
    # 0.0 = black image
    # 2.0 = white image
    
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)

def color_enhancement(img): #saturation
    """
    Applies color enhancement to the image.
    """
    factor = random.uniform(0.5, 1.5)
    
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(factor)

def sharpening(img):
    """
    Applies sharpening to the image.
    """
    factor = random.uniform(0.5, 2.0)
    
    enhancer = ImageEnhance.Sharpness(img)
    return enhancer.enhance(factor)

def filter_edge_enhance(img):
    """
    Applies edge enhancement to the image.
    """
    return img.filter(ImageFilter.EDGE_ENHANCE)
def filter_edge_enhance_more(img):
    """
    Applies edge enhancement more to the image.
    """
    return img.filter(ImageFilter.EDGE_ENHANCE_MORE)


def combination_transformation (img):
    """
    Applies a combination of transformations to the image.
    """
    transfirmations_list = [horizontal_flip, contrast_enhancement, brightness_enhancement, color_enhancement, sharpening,
                             filter_edge_enhance, filter_edge_enhance_more]
    
    # random list of ones and zeros
    random_list = np.random.randint(0, 2, len(transfirmations_list), dtype=bool)
    
    for i in range(len(transfirmations_list)):
        if random_list[i]:
            img = transfirmations_list[i](img)
    return img


def distinct_multiple_transformations(img, n_children=5):
    """
    Applies n distinct transformations to the image.
    """
    
    images = []
    
    for i in range(n_children):
        child = combination_transformation(img)
        images.append(child)
    

    return images
