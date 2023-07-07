from BERT_architecture.umBERT import umBERT

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import torch.nn as nn
from torchvision.transforms import transforms
from torch.optim import Adam
import os
from utility.image_to_embedding import image_to_embedding
from utility.custom_image_dataset import CustomImageDataset
from utility.create_tensor_dataset_from_dataframe import create_tensor_dataset_for_BC_from_dataframe
from utility.masking_input import masking_input
import numpy as np
from torch.nn import CrossEntropyLoss

# set the working directory to the path of the file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

np.random.seed(42)  # for reproducibility

# use GPU if available
device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
print('Using device:', device)

catalogue = pd.read_csv('../reduced_data/reduced_catalogue.csv')  # load the catalogue

# first step: obtain the embeddings of the dataset using the fine-tuned model finetuned_fashion_resnet18.pth

# load the model finetuned_fashion_resnet18.pth
fashion_resnet18 = resnet18()
num_ftrs = fashion_resnet18.fc.in_features  # get the number of input features for the last fully connected layer
fashion_resnet18.fc = nn.Linear(num_ftrs,
                                4)  # modify the last fully connected layer to have the desired number of classes
fashion_resnet18.load_state_dict(torch.load(
    '../models/finetuned_fashion_resnet18.pth'))  # load the weights of the model finetuned_fashion_resnet18.pth
fashion_resnet18.eval()  # set the model to evaluation mode
fashion_resnet18.to(device)  # set the model to run on the device

# load the dataset (just for now, we will use the test dataset)
data_transform = transforms.Compose([  # define the transformations to be applied to the images
    transforms.Resize((224, 224)),  # resize the image to 224x224
    transforms.ToTensor()  # convert the image to a tensor
])
data_dir = '../dataset_catalogue'  # define the directory of the dataset
image_dataset = CustomImageDataset(root_dir=data_dir, data_transform=data_transform)  # create the dataset

dataloaders = DataLoader(image_dataset, batch_size=32, shuffle=False, num_workers=0)  # create the dataloader

# get the embeddings of the dataset, the labels and the ids
print('Getting the embeddings of the dataset...')
embeddings, labels, IDs = image_to_embedding(dataloaders, fashion_resnet18, device)
print('Done!')
# create MASK and CLS token embeddings as random tensors with the same shape of the embeddings
print('Creating the MASK and CLS token embeddings...')
CLS = np.random.rand(1, embeddings.shape[1])
MASK = np.random.rand(1, embeddings.shape[1])
print('Done!')
# define the umBERT model
model = umBERT(catalogue_size=catalogue['ID'].size, d_model=embeddings.shape[1], num_encoders=6, num_heads=8, dropout=0,
               dim_feedforward=None)
model.to(device)  # set the model to run on the device

# import the training set
train_dataframe = pd.read_csv('../reduced_data/reduced_compatibility_train.csv')
compatibility_train = train_dataframe['compatibility'].values
train_dataframe.drop(columns='compatibility', inplace=True)

# create the tensor dataset for the training set (which contains the CLS embedding)
print('Creating the tensor dataset for the training set...')
training_set = create_tensor_dataset_for_BC_from_dataframe(train_dataframe, embeddings, IDs, CLS)
# mask the input (using the MASK embedding)
print('Masking the input...')
training_set, masked_indexes_train, masked_labels_train = masking_input(training_set, train_dataframe, MASK)

# create the dataloader for the training set
print('Creating the dataloader for the training set...')
trainloader = DataLoader(training_set, batch_size=32, shuffle=False, num_workers=0)
print('Done!')

# import the validation set
valid_dataframe = pd.read_csv('../reduced_data/reduced_compatibility_valid.csv')
compatibility_valid = valid_dataframe['compatibility'].values
valid_dataframe.drop(columns='compatibility', inplace=True)

# create the tensor dataset for the validation set (which contains the CLS embedding)
print('Creating the tensor dataset for the validation set...')
validation_set = create_tensor_dataset_for_BC_from_dataframe(valid_dataframe, embeddings, IDs, CLS)
# mask the input (using the MASK embedding)
print('Masking the input...')
validation_set, masked_indexes_valid, masked_labels_valid = masking_input(validation_set, valid_dataframe, MASK)

# create the dataloader for the validation set
print('Creating the dataloader for the validation set...')
validloader = DataLoader(validation_set, batch_size=32, shuffle=False, num_workers=0)
print('Done!')

# create the dictionary containing the dataloaders, the masked indices, the masked labels and the compatibility labels
# for the training and validation set
dataloaders = {'train': trainloader, 'val': validloader}
masked_indices = {'train': masked_indexes_train, 'val': masked_indexes_valid}
masked_labels = {'train': masked_labels_train, 'val': masked_labels_valid}
compatibility = {'train': compatibility_train, 'val': compatibility_valid}

# import the test set
test_dataframe = pd.read_csv('../reduced_data/reduced_compatibility_test.csv')
compatibility_test = test_dataframe['compatibility'].values
test_dataframe.drop(columns='compatibility', inplace=True)

# create the tensor dataset for the test set (which contains the CLS embedding)
print('Creating the tensor dataset for the test set...')
test_set = create_tensor_dataset_for_BC_from_dataframe(test_dataframe, embeddings, IDs, CLS)
# mask the input (using the MASK embedding)
print('Masking the input...')
test_set, masked_indexes_test, masked_labels_test = masking_input(test_set, test_dataframe, MASK)

# create the dataloader for the test set
print('Creating the dataloader for the test set...')
testloader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)
print('Done!')

optimizer = Adam(params=model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01)
criterion = CrossEntropyLoss()

print('Start pre-training the model')
model.pre_train_BERT_like(optimizer=optimizer, criterion=criterion, dataloaders=dataloaders,
                          labels_classification=compatibility, labels_ids=masked_labels,
                          masked_positions=masked_indices, device=device, n_epochs=100)
print('Pre-training completed')
# save the model into a checkpoint file
torch.save(model.state_dict(), '../models/umBERT_pretrained_jointly.pth')
