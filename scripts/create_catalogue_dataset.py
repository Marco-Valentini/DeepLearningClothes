import os
import pandas as pd
from PIL import Image

# set the working directory to the path of the file
os.chdir(os.path.dirname(os.path.abspath(__file__)))
'''
this python script takes images from a catalogue and move them in the folder 'dataset_catalogue'
'''
root_dir = '../dataset/images/'
csv_file_path = '../reduced_data/reduced_catalogue.csv'
data = pd.read_csv(csv_file_path)

images = list(data.loc[:, 'ID'].values)
labels = list(data.loc[:, 'Semantic_category'].values)

# organise the train set in folders
for idx in range(len(images)):
    print(f"Saving image {idx}, total {idx/len(images)}")
    image_name = str(images[idx]) + '.jpg'
    image_path = os.path.join(root_dir, image_name)
    image = Image.open(image_path)
    new_image_path = f'../dataset_catalogue/{image_name}'
    image.save(new_image_path)
