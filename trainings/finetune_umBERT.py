import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd

np.random.seed(42)  # for reproducibility

# set the working directory to the path of the file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

device = torch.device("mps" if torch.has_mps else "cpu")  # use the mps device if available
print(f'Working on device: {device}')

catalogue = pd.read_csv('../reduced_data/reduced_catalogue.csv')  # load the catalogue
print("Catalogue loaded")

# import the dataset (to be created)

# create training and validation sets and dataloaders

# train the model

# print training results

# test the model

# save the results

