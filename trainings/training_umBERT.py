from BERT_architecture.umBERT import umBERT

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
from torchvision.transforms import transforms

from utility.image_to_embedding import image_to_embedding

# use GPU if available
device = torch.device("mps" if torch.has_mps else "cpu")

catalogue = pd.read_csv('../reduced_data/reduced_catalogue.csv')  # load the catalogue

# first step: obtain the embeddings of the dataset using the fine-tuned model finetuned_fashion_resnet18.pth

# load the model finetuned_fashion_resnet18.pth
fashion_resnet18 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = fashion_resnet18.fc.in_features  # get the number of input features for the last fully connected layer
fashion_resnet18.fc = nn.Linear(num_ftrs, 4)  # modify the last fully connected layer to have the desired number of classes
fashion_resnet18.load_state_dict(torch.load('../models/finetuned_fashion_resnet18.pth'))  # load the weights of the model finetuned_fashion_resnet18.pth
fashion_resnet18.eval()  # set the model to evaluation mode
fashion_resnet18.to(device)  # set the model to run on the device

# load the dataset (just for now, we will use the test dataset)
data_transform = transforms.Compose([  # define the transformations to be applied to the images
    transforms.Resize((224, 224)),  # resize the image to 224x224
    transforms.ToTensor()  # convert the image to a tensor
])
data_dir = '../dataset_cnn_fine_tuning'
test_dir = data_dir + '/test'
image_dataset = ImageFolder(test_dir, transform=data_transform)  # create the dataset

dataloaders = DataLoader(image_dataset, batch_size=32, shuffle=False, num_workers=0)  # create the dataloader

# get the embeddings of the dataset, the labels and the ids
embeddings, labels, indexes = image_to_embedding(dataloaders, catalogue, fashion_resnet18, device)

# define the umBERT model
umBERT = umBERT(catalogue_size=catalogue['ID'].size, d_model=embeddings.shape[1], num_encoders=6, num_heads=8, dropout=0, dim_feedforward=None)

# prepare the data for the umBERT model

