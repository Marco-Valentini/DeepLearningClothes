import random

import torch
import torch.nn as nn
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
from constants import get_special_embeddings

# set the seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)

# import the MASK and CLS tokens
dim_embeddings = 512
CLS, MASK = get_special_embeddings(dim_embeddings)


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

with open('../reduced_data/embeddings_512.npy', 'rb') as f:
    embeddings = np.load(f)

print("Embeddings loaded")


# import the training set
train_dataframe = pd.read_csv('../reduced_data/reduced_compatibility_train.csv')
# remove all the outfits with compatibility = 0 (we are interested only in the compatible outfits)
train_dataframe = train_dataframe[train_dataframe['compatibility'] == 1]
train_dataframe.drop(columns='compatibility', inplace=True)

print("Creating the training set")
training_set = create_tensor_dataset_for_BC_from_dataframe(train_dataframe, embeddings, IDs, CLS)

# remove the CLS layer from the training set
CLS_layer = training_set[0, :, :].unsqueeze(0)
training_set = training_set[1:, :, :]
# scale the training set using z-score (layer normalization + batch normalization)
print('Scaling the training set using z-score...')
mean = training_set.mean(dim=0).mean(dim=0)
std = training_set.std(dim=0).std(dim=0)
torch.save(mean, '../reduced_data/mean_fine_tuning.pth')
torch.save(std, '../reduced_data/std_fine_tuning.pth')
training_set = (training_set - mean) / std
# reattach the CLS layer
training_set = torch.cat((CLS_layer, training_set), dim=0)

# augment the training set with all the permutations of the outfits
print('Augmenting the training set with all the permutations of the outfits...')
training_set = create_permutations_per_outfit(training_set, with_CLS=False)  # to remove CLS embeddings
print('making the training set for fill in the blank fine tuning...')
train_dataframe = pd.DataFrame(np.repeat(train_dataframe.values, 24, axis=0), columns=train_dataframe.columns)

training_set, masked_indexes_train, labels_train = mask_one_item_per_time(training_set,
                                                                                 train_dataframe,
                                                                                 MASK,
                                                                                 input_contains_CLS=False,
                                                                                 device=device,
                                                                                 output_in_batch_first=True)

labels_train_tensor = torch.Tensor(labels_train).unsqueeze(1)
masked_positions_tensor_train = torch.Tensor(masked_indexes_train).unsqueeze(1)
labels_train_tensor = torch.concat((labels_train_tensor, masked_positions_tensor_train), dim=1)
training_dataset = torch.utils.data.TensorDataset(training_set, labels_train_tensor)
trainloader = DataLoader(training_dataset, batch_size=8, num_workers=0, shuffle=True)
print("Training set for fill in the blank fine tuning created")

# create validation sets and dataloaders
# import the validation set
valid_dataframe = pd.read_csv('../reduced_data/reduced_compatibility_valid.csv')
# remove all the outfits with compatibility = 0 (we are interested only in the compatible outfits)
valid_dataframe = valid_dataframe[valid_dataframe['compatibility'] == 1]
valid_dataframe.drop(columns='compatibility', inplace=True)

print("Creating the validation set")
validation_set = create_tensor_dataset_for_BC_from_dataframe(valid_dataframe, embeddings, IDs, CLS)
print('Scaling the validation set using z-score...')
# remove the CLS layer from the validation set
CLS_layer = validation_set[0, :, :].unsqueeze(0)
validation_set = validation_set[1:, :, :]
validation_set = (validation_set - mean) / std
# reattach the CLS layer
validation_set = torch.cat((CLS_layer, validation_set), dim=0)
# augment the validation set with all the permutations of the outfits
print('Augmenting the validation set with all the permutations of the outfits...')
validation_set = create_permutations_per_outfit(validation_set, with_CLS=False)  # to remove CLS embeddings
# scale the validation set using z-score (layer+batch normalization) (using the mean and std of the training set)
# repeat each row of the train dataframe 24 times (all the permutations of the outfit)
valid_dataframe = pd.DataFrame(np.repeat(valid_dataframe.values, 24, axis=0), columns=train_dataframe.columns)
# mask one item per time
validation_set, masked_indexes_val, labels_val = mask_one_item_per_time(validation_set,
                                                                           valid_dataframe,
                                                                           MASK,
                                                                           input_contains_CLS=False,
                                                                           device=device,
                                                                           output_in_batch_first=True)
# create the validation set for the fill in the blank task
labels_val_tensor = torch.Tensor(labels_val).unsqueeze(1)
masked_positions_tensor_valid = torch.Tensor(masked_indexes_val).unsqueeze(1)
valid_labels = torch.concat((labels_val_tensor, masked_positions_tensor_valid), dim=1)
validation_dataset_MLM = torch.utils.data.TensorDataset(validation_set, valid_labels)
validloader = DataLoader(validation_dataset_MLM, batch_size=8, num_workers=0, shuffle=True)
print("Validation set for fill in the blank fine tuning created")

# create the dictionary to work with modularity
dataloaders = {'train': trainloader, 'val': validloader}

# define the umBERT model
checkpoint = torch.load('../models/umBERT_pretrained_BERT_like.pth')

model = umBERT(catalogue_size=checkpoint['catalogue_size'], d_model=checkpoint['d_model'],
               num_encoders=checkpoint['num_encoders'], num_heads=checkpoint['num_heads'],
               dropout=checkpoint['dropout'], dim_feedforward=checkpoint['dim_feedforward'])
# load the model weights
model.load_state_dict(checkpoint['model_state_dict'])

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

# test the model then in the main on the test set