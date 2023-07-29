import os
import pandas as pd
from PIL import Image

# set the working directory to the path of the file
os.chdir(os.path.dirname(os.path.abspath(__file__)))
'''
this python script takes images from a catalogue and move them in the folder 'dataset_catalogue'
'''
root_dir = '../dataset/images/'

reduced_compatibility = pd.read_csv('../reduced_data/reduced_compatibility.csv')  # this dataset cotains both compatible and incompatible outfits
reduced_compatibility.drop(columns='compatibility', inplace=True)

# create a list of all the ids in the reduced_compatibility and unified_dataset_MLM datasets
images = []
for outfit in reduced_compatibility.values:
    images.extend(list(outfit))

# remove duplicates
images = list(set(images))

for idx in range(len(images)):
    print(f"Saving image {idx}, total {idx/len(images)}")
    image_name = str(images[idx]) + '.jpg'
    image_path = os.path.join(root_dir, image_name)
    image = Image.open(image_path)
    new_image_path = f'../dataset_catalogue/{image_name}'
    image.save(new_image_path)
