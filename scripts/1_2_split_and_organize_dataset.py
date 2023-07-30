import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

# set the working directory to the path of the file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

'''
this python script takes a dataset of images, splits it into train, val and test sets and organises them in folders 
with the name of the label associated with each image
'''
root_dir = '../dataset/images/'
csv_file_path = '../reduced_data/reduced_catalogue.csv'
data = pd.read_csv(csv_file_path)

images = list(data.loc[:, 'ID'].values)
labels = list(data.loc[:, 'Semantic_category'].values)

# Split the dataset into train and testing/validation sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Further split the train set into train and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2,
                                                                      random_state=42)

# organise the train set in folders
for idx in range(len(train_images)):
    image_name = str(train_images[idx]) + '.jpg'
    image_path = os.path.join(root_dir, image_name)
    image = Image.open(image_path)
    label = train_labels[idx]
    new_image_path = f'../dataset_cnn_fine_tuning/train/{label}/{image_name}'
    image.save(new_image_path)

# organise the val set in folder
for idx in range(len(val_images)):
    image_name = str(val_images[idx]) + '.jpg'
    image_path = os.path.join(root_dir, image_name)
    image = Image.open(image_path)
    label = val_labels[idx]
    new_image_path = f'../dataset_cnn_fine_tuning/val/{label}/{image_name}'
    image.save(new_image_path)

# organise the test set in folder
for idx in range(len(test_images)):
    image_name = str(test_images[idx]) + '.jpg'
    image_path = os.path.join(root_dir, image_name)
    image = Image.open(image_path)
    label = test_labels[idx]
    new_image_path = f'../dataset_cnn_fine_tuning/test/{label}/{image_name}'
    image.save(new_image_path)
