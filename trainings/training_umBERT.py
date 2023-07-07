from BERT_architecture.umBERT import umBERT

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
from torchvision.transforms import transforms
from torch.optim import Adam
import os
from utility.image_to_embedding import image_to_embedding
from utility.custom_image_dataset import CustomImageDataset
from utility.create_tensor_dataset_for_BC_from_dataframe import create_tensor_dataset_for_BC_from_dataframe
import numpy as np
from utility.masking_input import masking_input

# set the working directory to the path of the file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

np.random.seed(42)  # for reproducibility

# use GPU if available
device = torch.device("mps" if torch.has_mps else "cpu")
print("Device used: ", device)

catalogue = pd.read_csv('../reduced_data/reduced_catalogue.csv')  # load the catalogue
print("Catalogue loaded")

# first step: obtain the embeddings of the dataset using the fine-tuned model finetuned_fashion_resnet18.pth

# load the model finetuned_fashion_resnet18.pth
print("Loading the model finetuned_fashion_resnet18.pth")
fashion_resnet18 = resnet18()
num_ftrs = fashion_resnet18.fc.in_features  # get the number of input features for the last fully connected layer
fashion_resnet18.fc = nn.Linear(num_ftrs,
                                4)  # modify the last fully connected layer to have the desired number of classes
fashion_resnet18.load_state_dict(torch.load(
    '../models/finetuned_fashion_resnet18.pth'))  # load the weights of the model finetuned_fashion_resnet18.pth
fashion_resnet18.eval()  # set the model to evaluation mode
fashion_resnet18.to(device)  # set the model to run on the device

print("Model loaded")

print("Loading the image dataset")

# load the dataset (just for now, we will use the test dataset)
data_transform = transforms.Compose([  # define the transformations to be applied to the images
    transforms.Resize((224, 224)),  # resize the image to 224x224
    transforms.ToTensor()  # convert the image to a tensor
])
data_dir = '../dataset_catalogue'  # define the directory of the dataset
image_dataset = CustomImageDataset(root_dir=data_dir, data_transform=data_transform)  # create the dataset

dataloaders = DataLoader(image_dataset, batch_size=8, shuffle=False, num_workers=0)  # create the dataloader
print("Dataset loaded")

print("Computing the embeddings of the dataset")
# get the embeddings of the dataset, the labels and the ids
embeddings, labels, IDs = image_to_embedding(dataloaders, fashion_resnet18, device)

print("Embeddings computed")

# create MASK and CLS token embeddings as random tensors with the same shape of the embeddings
CLS = np.random.rand(1, embeddings.shape[1])  # TODO controlla seed resti lo stesso
MASK = np.random.rand(1, embeddings.shape[1])
print("CLS and MASK token embeddings created")

# define the umBERT model
model = umBERT(catalogue_size=catalogue['ID'].size, d_model=embeddings.shape[1], num_encoders=6, num_heads=8, dropout=0,
               dim_feedforward=None)

# import the training set
train_dataframe = pd.read_csv('../reduced_data/reduced_compatibility_train.csv')
compatibility_train = train_dataframe['compatibility'].values
train_dataframe.drop(columns='compatibility', inplace=True)

print("Creating the training set")
training_set = create_tensor_dataset_for_BC_from_dataframe(train_dataframe, embeddings, IDs, CLS)
trainloader_BC = DataLoader(training_set.transpose(0,1), batch_size=8, num_workers=0, shuffle=False)
print("Training set for binary classifiation created")
training_set_MLM, masked_positions_train, actual_masked_values_train = masking_input(training_set, train_dataframe,
                                                                                     MASK, with_CLS=False)
trainloader_MLM = DataLoader(training_set_MLM, batch_size=8, num_workers=0, shuffle=False)
print("Training set for masked language model created")

# import the validation set
valid_dataframe = pd.read_csv('../reduced_data/reduced_compatibility_valid.csv')
compatibility_valid = valid_dataframe['compatibility'].values
valid_dataframe.drop(columns='compatibility', inplace=True)

print("Creating the validation set")
validation_set = create_tensor_dataset_for_BC_from_dataframe(valid_dataframe, embeddings, IDs, CLS)
validloader_BC = DataLoader(validation_set.transpose(0,1), batch_size=8, num_workers=0, shuffle=False)
print("Validation set for binary classifiation created")
validation_set_MLM, masked_positions_valid, actual_masked_values_valid = masking_input(validation_set, valid_dataframe,
                                                                                       MASK, with_CLS=False)
validloader_MLM = DataLoader(validation_set_MLM, batch_size=8, num_workers=0, shuffle=False)
print("Validation set for masked language model created")

# import the test set

test_dataframe = pd.read_csv('../reduced_data/reduced_compatibility_test.csv')
compatibility_test = test_dataframe['compatibility'].values
test_dataframe.drop(columns='compatibility', inplace=True)

print("Creating the test set")
test_set = create_tensor_dataset_for_BC_from_dataframe(test_dataframe, embeddings, IDs, CLS)
testloader_BC = DataLoader(test_set.transpose(0,1), batch_size=8, num_workers=0, shuffle=False)
print("Test set for binary classifiation created")
test_set_MLM, masked_positions_test, actual_masked_values_test = masking_input(test_set, test_dataframe, MASK,
                                                                               with_CLS=False)
testloader_MLM = DataLoader(test_set_MLM, batch_size=8, num_workers=0, shuffle=False)
print("Test set for masked language model created")

# create the dictionary to work with modularity
dataloaders_BC = {'train': trainloader_BC, 'val': validloader_BC}
dataloaders_MLM = {'train': trainloader_MLM, 'val': validloader_MLM}
masked_indices = {'train': masked_positions_train, 'val': masked_positions_valid}
masked_labels = {'train': actual_masked_values_train, 'val': actual_masked_values_valid}
compatibility = {'train': compatibility_train, 'val': compatibility_valid}
# define the optimizer
optimizer = Adam(params=model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01)
# put the model on the GPU
model.to(device)
criterion_1 = nn.BCELoss()

model.pre_train_BC(optimizer, criterion_1, dataloaders_BC, compatibility, n_epochs=100,device=device)

torch.save(model.state_dict(), '../models/umBERT_pretrained_1.pth')

criterion_2 = nn.CrossEntropyLoss()

model.pre_train_MLM(optimizer, criterion_2, dataloaders_MLM, masked_labels, masked_indices,
                    n_epochs=100,device=device)

# save the model into a checkpoint file
torch.save(model.state_dict(), '../models/umBERT_pretrained_2.pth')
