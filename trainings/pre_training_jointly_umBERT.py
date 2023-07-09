import json
from BERT_architecture.umBERT import umBERT
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import os
from utility.create_tensor_dataset_for_BC_from_dataframe import create_tensor_dataset_for_BC_from_dataframe
from utility.masking_input import masking_input
from utility.dataset_augmentation import create_permutations_per_outfit
import numpy as np
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from utility.umBERT_trainer import umBERT_trainer

# set the working directory to the path of the file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

np.random.seed(42)  # for reproducibility

# use GPU if available
device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
print('Using device:', device)

catalogue = pd.read_csv('../reduced_data/reduced_catalogue.csv')  # load the catalogue

# first step: obtain the embeddings of the dataset using the fine-tuned model finetuned_fashion_resnet18.pth
with open("../reduced_data/IDs_list", "r") as fp:
    IDs = json.load(fp)
print("IDs loaded")

with open('../reduced_data/embeddings.npy', 'rb') as f:
    embeddings = np.load(f)

print("Embeddings loaded")

# create MASK and CLS token embeddings as random tensors with the same shape of the embeddings
print('Creating the MASK and CLS token embeddings...')
CLS = np.random.rand(1, embeddings.shape[1])
MASK = np.random.rand(1, embeddings.shape[1])
print('Done!')

# import the training set
train_dataframe = pd.read_csv('../reduced_data/reduced_compatibility_train.csv')
compatibility_train = train_dataframe['compatibility'].values
train_dataframe.drop(columns='compatibility', inplace=True)

# create the tensor dataset for the training set (which contains the CLS embedding)
print('Creating the tensor dataset for the training set...')
training_set = create_tensor_dataset_for_BC_from_dataframe(train_dataframe, embeddings, IDs, CLS)
# augment the training set with all the permutations of the outfits
print('Augmenting the training set with all the permutations of the outfits...')
training_set = create_permutations_per_outfit(training_set)
# scale the training set using z-score (layer normalization + batch normalization)
print('Scaling the training set using z-score...')
CLS_layer_train = training_set[0, :, :]  # CLS is the first element of the tensor, preserve it
training_set = training_set[1:, :, :]  # remove CLS from the tensor
mean = training_set.mean(dim=0).mean(dim=0)
std = training_set.std(dim=0).std(dim=0)
training_set = (training_set - mean) / std
training_set = torch.cat((CLS_layer_train.unsqueeze(0), training_set), dim=0)  # concatenate CLS to the scaled tensor
# mask the input (using the MASK embedding)
print('Masking the input...')
# repeat each row of the train dataframe 24 times (all the permutations of the outfit)
train_dataframe = pd.DataFrame(np.repeat(train_dataframe.values, 24, axis=0), columns=train_dataframe.columns)
training_set, masked_indexes_train, masked_labels_train = masking_input(training_set, train_dataframe, MASK)
# labels for BC are the same as the compatibility labels, labels for MLM are the masked labels
# repeat each label 24 times (all the permutations of the outfit)
compatibility_train = np.repeat(compatibility_train, 24, axis=0)
BC_train_labels = torch.Tensor(compatibility_train).unsqueeze(1)
MLM_train_labels = torch.Tensor(masked_labels_train).unsqueeze(1)
masked_train_positions = torch.Tensor(masked_indexes_train).unsqueeze(1)
# concatenate the labels
train_labels = torch.concat((BC_train_labels, MLM_train_labels, masked_train_positions), dim=1)
# create a Tensor Dataset
training_set = torch.utils.data.TensorDataset(training_set, train_labels)
# create the dataloader for the training set
print('Creating the dataloader for the training set...')
trainloader = DataLoader(training_set, batch_size=32, shuffle=True, num_workers=0)

# import the validation set
valid_dataframe = pd.read_csv('../reduced_data/reduced_compatibility_valid.csv')
compatibility_valid = valid_dataframe['compatibility'].values
valid_dataframe.drop(columns='compatibility', inplace=True)

# create the tensor dataset for the validation set (which contains the CLS embedding)
print('Creating the tensor dataset for the validation set...')
validation_set = create_tensor_dataset_for_BC_from_dataframe(valid_dataframe, embeddings, IDs, CLS)
# augment the validation set with all the permutations of the outfits
print('Augmenting the validation set with all the permutations of the outfits...')
validation_set = create_permutations_per_outfit(validation_set)
# scale the validation set using z-score (layer normalization) (using the mean and std of the training set)
print('Scaling the validation set using z-score...')
CLS_layer_val = validation_set[0, :, :]  # CLS is the first element of the tensor, preserve it
validation_set = validation_set[1:, :, :]  # remove CLS from the tensor
validation_set = (validation_set - mean) / std
validation_set = torch.cat((CLS_layer_val.unsqueeze(0), validation_set), dim=0)  # concatenate CLS to the scaled tensor
# mask the input (using the MASK embedding)
print('Masking the input...')
# repeat each row of the valid dataframe 24 times (all the permutations of the outfit)
valid_dataframe = pd.DataFrame(np.repeat(valid_dataframe.values, 24, axis=0), columns=valid_dataframe.columns)
validation_set, masked_indexes_valid, masked_labels_valid = masking_input(validation_set, valid_dataframe, MASK)
# labels for BC are the same as the compatibility labels, labels for MLM are the masked labels
# repeat each label 24 times (all the permutations of the outfit)
compatibility_valid = np.repeat(compatibility_valid, 24, axis=0)
BC_valid_labels = torch.Tensor(compatibility_valid).unsqueeze(1)
MLM_valid_labels = torch.Tensor(masked_labels_valid).unsqueeze(1)
masked_val_positions = torch.Tensor(masked_indexes_valid).unsqueeze(1)
# concatenate the labels
valid_labels = torch.concat((BC_valid_labels, MLM_valid_labels, masked_val_positions), dim=1)
# create a Tensor Dataset
validation_set = torch.utils.data.TensorDataset(validation_set, valid_labels)

# create the dataloader for the validation set
print('Creating the dataloader for the validation set...')
validloader = DataLoader(validation_set, batch_size=32, shuffle=True, num_workers=0)
print('Done!')

# create the dictionary containing the dataloaders, the masked indices, the masked labels and the compatibility labels
# for the training and validation set
dataloaders = {'train': trainloader, 'val': validloader}
masked_indices = {'train': masked_indexes_train, 'val': masked_indexes_valid}

# import the test set
test_dataframe = pd.read_csv('../reduced_data/reduced_compatibility_test.csv')
compatibility_test = test_dataframe['compatibility'].values
test_dataframe.drop(columns='compatibility', inplace=True)

# # create the tensor dataset for the test set (which contains the CLS embedding)
# print('Creating the tensor dataset for the test set...')
# test_set = create_tensor_dataset_for_BC_from_dataframe(test_dataframe, embeddings, IDs, CLS)
# # mask the input (using the MASK embedding)
# print('Masking the input...')
# test_set, masked_indexes_test, masked_labels_test = masking_input(test_set, test_dataframe, MASK)
#
# # labels for BC are the same as the compatibility labels, labels for MLM are the masked labels
# BC_test_labels = torch.Tensor(compatibility_test).unsqueeze(1)
# MLM_test_labels = torch.Tensor(masked_labels_test).unsqueeze(1)
# masked_test_positions = torch.Tensor(masked_indexes_test).unsqueeze(1)
# # concatenate the labels
# test_labels = torch.concat((BC_test_labels, MLM_test_labels, masked_test_positions), dim=1)
# # create a Tensor Dataset
# test_set = torch.utils.data.TensorDataset(test_set, test_labels)
# # create the dataloader for the test set
# print('Creating the dataloader for the test set...')
# testloader = DataLoader(test_set, batch_size=16, shuffle=True, num_workers=0)
# print('Done!')

# define the umBERT model
model = umBERT(catalogue_size=catalogue['ID'].size, d_model=embeddings.shape[1], num_encoders=6, num_heads=8,
               dropout=0.3, dim_feedforward=None)

# use Adam as optimizer as suggested in the paper
optimizer = Adam(params=model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-08)
criterion_MLM = CrossEntropyLoss()
criterion_BC = BCEWithLogitsLoss()
criteria = {'BC': criterion_BC, 'MLM': criterion_MLM}

print('Start pre-training the model')
trainer = umBERT_trainer(model=model, optimizer=optimizer, criterion=criteria, device=device, n_epochs=500)
trainer.pre_train_BERT_like(dataloaders=dataloaders)
print('Pre-training completed')
# save the model into a checkpoint file
torch.save(model.state_dict(), '../models/umBERT_pretrained_jointly.pth')
