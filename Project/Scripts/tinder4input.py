import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


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

path = '../Data/rare_species 1'
df = pd.read_csv(path + '/metadata.csv')
df.head()

first_image = df['file_path'][0]
paths_list = df['file_path'].tolist()

answer = []

checkpoint = 0
# open a file to save the answers

with open('answers.txt', 'w') as file:
    #  look for the last line in the file
    for line in file:
        checkpoint += 1

for index, img in enumerate(paths_list):
    if index < checkpoint:
        continue
    
    full_path_image = path + '/' + img
    img_raw = tf.io.read_file(full_path_image)
    img_tensor = tf.image.decode_image(img_raw)
    show_image(img_tensor, 'Image {}'.format(index))
    
    user_said = input('L = like, D = dislike, H = Hard: ')

    # write the answer to the file
    with open('answers.txt', 'a') as file:
        file.write(str(index) + '*' + user_said + '\n')
    

df['answer'] = answer





    