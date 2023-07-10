import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
import json
from BERT_architecture.umBERT import umBERT
from utility.create_tensor_dataset_for_BC_from_dataframe import create_tensor_dataset_for_BC_from_dataframe
from utility.dataset_augmentation import create_permutations_per_outfit, mask_one_item_per_time
from torch.utils.data import DataLoader
from torch.optim import Adam
from utility.umBERT_trainer import umBERT_trainer

np.random.seed(42)  # for reproducibility


def freeze(model, n=None):
    """
    this function freezes the given number of layers of the given model in order to fine-tune it.
    :param model: the model to be frozen
    :param n: the number of layers to be frozen, if None, all the layers are frozen
    :return: the frozen model
    """
    if n is None:
        for param in model.parameters():
            param.requires_grad = False
    else:
        count = 0
        for param in model.parameters():
            if count < n:
                param.requires_grad = False
            count += 1


# set the working directory to the path of the file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

device = torch.device("mps" if torch.has_mps else "cpu")  # use the mps device if available
print(f'Working on device: {device}')

catalogue = pd.read_csv(
    '../reduced_data/reduced_catalogue.csv')  # load the catalogue of ID (related positions will be the labels_train) and classes
print("Catalogue loaded")

# import embeddings and IDs
with open("../reduced_data/IDs_list", "r") as fp:
    IDs = json.load(fp)
print("IDs loaded")

with open('../reduced_data/embeddings.npy', 'rb') as f:
    embeddings = np.load(f)

print("Embeddings loaded")

# create the MASK
print('Creating the MASK token embeddings...')
CLS = np.random.randn(1, embeddings.shape[1])
MASK = np.random.randn(1, embeddings.shape[1])
print('Done!')

# import the training set
train_dataframe = pd.read_csv('../reduced_data/reduced_compatibility_train.csv')
train_dataframe.drop(columns='compatibility', inplace=True)

print("Creating the training set")
training_set = create_tensor_dataset_for_BC_from_dataframe(train_dataframe, embeddings, IDs, CLS)

# augment the training set with all the permutations of the outfits
print('Augmenting the training set with all the permutations of the outfits...')
training_set = create_permutations_per_outfit(training_set, with_CLS=False)  # to remove CLS embeddings

# scale the training set using z-score (layer normalization + batch normalization)
print('Scaling the training set using z-score...')
mean = training_set.mean(dim=0).mean(dim=0)
std = training_set.std(dim=0).std(dim=0)
training_set = (training_set - mean) / std

train_dataframe = pd.DataFrame(np.repeat(train_dataframe.values, 24, axis=0), columns=train_dataframe.columns)

masked_outfit_train, masked_indexes_train, labels_train = mask_one_item_per_time(training_set, train_dataframe, MASK,
                                                                                 with_CLS=False)

train_labels = torch.Tensor(labels_train).reshape(len(labels_train), 1)
masked_positions_tensor_train = torch.Tensor(masked_indexes_train).reshape(len(masked_indexes_train), 1)
train_labels = torch.concat((train_labels, masked_positions_tensor_train), dim=1)
training_dataset = torch.utils.data.TensorDataset(training_set, train_labels)
trainloader = DataLoader(training_dataset, batch_size=8, num_workers=0, shuffle=True)
print("Training set for fill in the blank fine tuning created")

# create validation sets and dataloaders
# import the validation set
valid_dataframe = pd.read_csv('../reduced_data/reduced_compatibility_valid.csv')
valid_dataframe.drop(columns='compatibility', inplace=True)

print("Creating the validation set")
validation_set = create_tensor_dataset_for_BC_from_dataframe(valid_dataframe, embeddings, IDs, CLS)
# augment the validation set with all the permutations of the outfits
print('Augmenting the validation set with all the permutations of the outfits...')
validation_set = create_permutations_per_outfit(validation_set, with_CLS=False)  # to remove CLS embeddings
# scale the validation set using z-score (layer+batch normalization) (using the mean and std of the training set)
print('Scaling the validation set using z-score...')
validation_set = (validation_set - mean) / std
# repeat each row of the train dataframe 24 times (all the permutations of the outfit)
valid_dataframe = pd.DataFrame(np.repeat(valid_dataframe.values, 24, axis=0), columns=train_dataframe.columns)
# mask one item per time
masked_outfit_val, masked_indexes_val, labels_val = mask_one_item_per_time(validation_set, valid_dataframe, MASK,
                                                                           with_CLS=False)
# create the validation set for the fill in the blank task
valid_labels = torch.Tensor(labels_val).reshape(len(labels_val), 1)
masked_positions_tensor_valid = torch.Tensor(masked_indexes_val).reshape(len(masked_indexes_val), 1)
valid_labels = torch.concat((valid_labels, masked_positions_tensor_valid), dim=1)
validation_dataset_MLM = torch.utils.data.TensorDataset(validation_set, valid_labels)
validloader = DataLoader(validation_dataset_MLM, batch_size=8, num_workers=0, shuffle=True)
print("Validation set for fill in the blank fine tuning created")

# create the dictionary to work with modularity
dataloaders = {'train': trainloader, 'val': validloader}

# define the umBERT model
model = umBERT(catalogue_size=catalogue['ID'].size, d_model=embeddings.shape[1], num_encoders=6, num_heads=8,
               dropout=0.1,
               dim_feedforward=None)
# load the model weights
model.load_state_dict(torch.load('../models/umBERT.pth'))

# define the optimizer
optimizer = Adam(params=model.parameters(), lr=1e-4, betas=(0.9, 0.999),
                 weight_decay=0.01)  # TODO valutare anche adamW o LiON
# put the model on the GPU
model.to(device)
# define the loss function
criterion = nn.CrossEntropyLoss()
# train the model
print('Start fine tuning...')
trainer = umBERT_trainer(model, optimizer, criterion, device, n_epochs=10)
trainer.fine_tuning(dataloaders)
print('Fine tuning completed!')

# save the results
model.save('../models/umBERT_finetuned.pth')

# test the model
print('Testing the model...')
# TODO define test set and evaluate the model
