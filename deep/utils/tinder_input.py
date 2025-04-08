import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import threading
import time

import cv2
import numpy as np

def show_image(image, title):
    """
    Args:
        image: Image file (Tensor or NumPy array)
        title: Title of the image
    
    Displays the image and waits for a key press.
    """

    # Convert TensorFlow tensor to NumPy array (if needed)
    if not isinstance(image, np.ndarray):
        image = image.numpy()

    # Convert RGB to BGR (OpenCV uses BGR format)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Show the image
    cv2.imshow(title, image_bgr)

    # Wait for a key press for 7 seconds (7000 ms)
    cv2.waitKey(2000)

    # Close the image window
    cv2.destroyAllWindows()

path = '../Data/rare_species 1'
df = pd.read_csv(path + '/metadata.csv')
df.head()

first_image = df['file_path'][0]
paths_list = df['file_path'].tolist()

checkpoint = 0
# open a file to save the answers

with open("answers.txt", "r", encoding="utf-8") as file:
    #  look for the last line in the file
    for line in file:
        checkpoint += 1

for index, img in enumerate(paths_list):
    if index < checkpoint - 1:
        continue

    full_path_image = f"{path}/{img}"
    img_raw = tf.io.read_file(full_path_image)
    img_tensor = tf.image.decode_image(img_raw)

    # Show image in the main thread
    show_image(img_tensor, f'Image {index}')

    # Take user input AFTER the image is closed
    user_said = input('L = Like, D = Dislike, H = Hard:\n if tired, type Q\n')
    

    if user_said.upper() == 'Q':
        break

    # Write the answer to a file
    with open('answers.txt', 'a') as file:
        file.write(f"{index}*{user_said}\n")
    






    