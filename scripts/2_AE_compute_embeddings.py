# obtain the embeddings of the dataset using the trained model trained_fashion_VAE_resnet18_128.pth
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from utility.custom_image_dataset import CustomImageDataset
from torchvision.transforms import transforms
from models.model_autoencoder_ssimval import AutoEncoder
from utility.AE_image_to_embedding import AE_image_to_embedding
from datetime import datetime

# set the working directory to the path of the file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# use GPU if available
device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
# device = torch.device("cpu")
print("Device used: ", device)
# define the size of the embeddings

# load the checkpoint
checkpoint = torch.load(f'../checkpoints/trained_fashion_VAE_128.pth')
print(f"Loading the trained model")
model = AutoEncoder()
model.load_state_dict(checkpoint['model_state_dict'])  # load the weights of the trained model

model.eval()  # set the model to evaluation mode
model.to(device)  # set the model to run on the device

print("Model loaded")

print("Loading the image dataset")

# load the dataset (just for now, we will use the test dataset)
data_transform = transforms.Compose([  # define the transformations to be applied to the images
    transforms.Resize(128),  # resize the image to 256x256
    transforms.ToTensor(),  # convert the image to a tensor
])

data_dir = '../dataset_catalogue'  # define the directory of the dataset
image_dataset = CustomImageDataset(root_dir=data_dir, data_transform=data_transform)  # create the dataset

dataloaders = DataLoader(image_dataset, batch_size=32, num_workers=0)  # create the dataloader
print("Dataset loaded")

print("Computing the embeddings of the dataset")
# get the embeddings of the dataset, the labels and the ids
embeddings, IDs = AE_image_to_embedding(dataloaders, model, device)
print("Embeddings computed")
print("Saving the embeddings IDs")
now = datetime.now()
dt_string = now.strftime("%Y_%m_%d")
with open(f"{dt_string}AE_IDs_list", "w") as fp:
    json.dump(IDs, fp)
print("IDs saved")
dim_embeddings = embeddings.shape[1]
with open(f'./{dt_string}AE_embeddings_128.npy', 'wb') as f:
    np.save(f, embeddings)
print("Embeddings saved")
