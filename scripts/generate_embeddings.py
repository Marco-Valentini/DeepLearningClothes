import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
from torchvision.transforms import transforms
from utility.image_to_embedding import image_to_embedding
from utility.custom_image_dataset import CustomImageDataset

# use GPU if available
device = torch.device("mps" if torch.has_mps else "cpu")

catalogue = pd.read_csv('./reduced_data/reduced_catalogue.csv')  # load the catalogue

# first step: obtain the embeddings of the dataset using the fine-tuned model finetuned_fashion_resnet18.pth

# load the model finetuned_fashion_resnet18.pth
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features  # get the number of input features for the last fully connected layer
model.fc = nn.Linear(num_ftrs, 4)  # modify the last fully connected layer to have the desired number of classes
model.load_state_dict(torch.load(
    './models/finetuned_fashion_resnet18.pth'))  # load the weights of the model finetuned_fashion_resnet18.pth
model.eval()  # set the model to evaluation mode
model.to(device)  # set the model to run on the device

# load the dataset (just for now, we will use the test dataset)
data_transform = transforms.Compose([  # define the transformations to be applied to the images
    transforms.Resize((224, 224)),  # resize the image to 224x224
    transforms.ToTensor()  # convert the image to a tensor
])
data_dir = './dataset_catalogue'

image_dataset = CustomImageDataset(root_dir=data_dir, data_transform=data_transform)  # create the dataset
dataloader = DataLoader(image_dataset, batch_size=32, shuffle=False, num_workers=0)  # create the dataloader

# get the embeddings of the dataset, the labels and the ids
embeddings, labels, ids = image_to_embedding(dataloader, model, device)
# show the embeddings
print(embeddings)
print(embeddings.shape)

# create a dictionary with the embeddings and the ids
catalogue_embeddings = {str(ids[i]): embeddings[i] for i in range(len(ids))}

