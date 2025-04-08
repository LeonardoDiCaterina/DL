import pandas as pd
import tensorflow as tf
from deep.utils.animal_classification_funcs import show_image

path = '../data'
df = pd.read_csv(path + '/metadata.csv')

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

    img_id = df[df['file_path'] == img]['rare_species_id'].values[0]
    full_path_image = f"{path}/{img}"
    img_raw = tf.io.read_file(full_path_image)
    img_tensor = tf.image.decode_image(img_raw)

    # Show image in the main thread
    show_image(img_tensor, f'Image {index}')

    # Take user input AFTER the image is closed
    while True:
        user_said = input('1 = Animal, 0 = No Animal:\n to see picture again, type R\n if tired, type Q\n')

        if user_said in ['1', '0', 'Q', 'q']:
            break

        if user_said.upper() == 'R':
            show_image(img_tensor, f'Image {index}')
    
    if user_said.upper() == 'Q':
        break    

    # Write the answer to a file
    with open('answers.txt', 'a') as file:
        file.write(f"{img_id}, {user_said}\n") 