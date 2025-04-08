import tensorflow as tf
import matplotlib.pyplot as plt


def preprocess_image(image, final_size= [244,244], rotate = 0):

    '''
    Args:
        image: image file
        final_size: size of the final image
    
    Returns:
    
        image: preprocessed image
    '''
    
    
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, final_size)
    image /= 255.0 
    
    for _ in range(rotate):
        image = tf.image.rot90(image)
    return image

def load_and_preprocess_image(path, rotate = 0):
    '''
    Args:
        path: path to the image file
        rotate: number of times to rotate the image
    
    Returns:
    
        image: preprocessed image
    '''
    
    image = tf.io.read_file(path)
    return preprocess_image(image, rotate = rotate)

def show_image(image, title):
    '''
    Args:
        image: image file
        title: title of the image
    
    Plots the image
    
    '''
    plt.imshow(image)
    plt.axis('off')
    plt.title(title)
    plt.show()
