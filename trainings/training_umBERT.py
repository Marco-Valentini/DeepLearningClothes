from BERT_architecture.umBERT import umBERT
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
import os
from utility.create_tensor_dataset_for_BC_from_dataframe import create_tensor_dataset_for_BC_from_dataframe
import numpy as np
from utility.masking_input import masking_input
from utility.umBERT_trainer import umBERT_trainer
import json

# set the working directory to the path of the file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

np.random.seed(42)  # for reproducibility

# use GPU if available
device = torch.device("mps" if torch.has_mps else "cpu")
print("Device used: ", device)

catalogue = pd.read_csv('../reduced_data/reduced_catalogue.csv')  # load the catalogue
print("Catalogue loaded")

# first step: load the embeddings of the dataset obtained from fine-tuned model finetuned_fashion_resnet18
with open("../reduced_data/IDs_list", "r") as fp:
    IDs = json.load(fp)
print("IDs loaded")

with open('../reduced_data/embeddings.npy', 'rb') as f:
    embeddings = np.load(f)

print("Embeddings loaded")

# create MASK and CLS token embeddings as random tensors with the same shape of the embeddings
CLS = np.random.rand(1, embeddings.shape[1])  # TODO controlla seed resti lo stesso
MASK = np.random.rand(1, embeddings.shape[1])
print("CLS and MASK token embeddings created")

# define the umBERT model
model = umBERT(catalogue_size=catalogue['ID'].size, d_model=embeddings.shape[1], num_encoders=6, num_heads=8, dropout=0.1,
               dim_feedforward=None)

# import the training set
train_dataframe = pd.read_csv('../reduced_data/reduced_compatibility_train.csv')
print(f'target variable count in train_dataframe : {train_dataframe["compatibility"].value_counts()}')
compatibility_train = train_dataframe['compatibility'].values
train_dataframe.drop(columns='compatibility', inplace=True)

print("Creating the training set")
training_set = create_tensor_dataset_for_BC_from_dataframe(train_dataframe, embeddings, IDs, CLS)
training_dataset_BC = torch.utils.data.TensorDataset(training_set.transpose(0, 1), torch.Tensor(compatibility_train))
trainloader_BC = DataLoader(training_dataset_BC, batch_size=8, num_workers=0, shuffle=True)
print("Training set for binary classification created")
# training set is a tensor of shape (2736,4,512), masked_positions_train is a list of length 2736,
# actual_masked_values_train is a list of length 2736 with the positions of clothes in catalogue
training_set_MLM, masked_positions_train, actual_masked_values_train = masking_input(training_set, train_dataframe,
                                                                                     MASK, with_CLS=False)
MLM_train_labels = torch.Tensor(actual_masked_values_train).reshape(len(actual_masked_values_train), 1)
masked_positions_tensor_train = torch.Tensor(masked_positions_train).reshape(len(masked_positions_train), 1)
train_labels_MLM = torch.concat((MLM_train_labels,masked_positions_tensor_train), dim=1)
training_dataset_MLM = torch.utils.data.TensorDataset(training_set_MLM, train_labels_MLM)
trainloader_MLM = DataLoader(training_dataset_MLM, batch_size=8, num_workers=0, shuffle=True)
print("Training set for masked language model created")

# import the validation set
valid_dataframe = pd.read_csv('../reduced_data/reduced_compatibility_valid.csv')
print(f'target variable count in valid_dataframe : {valid_dataframe["compatibility"].value_counts()}')
compatibility_valid = valid_dataframe['compatibility'].values
valid_dataframe.drop(columns='compatibility', inplace=True)

print("Creating the validation set")
validation_set = create_tensor_dataset_for_BC_from_dataframe(valid_dataframe, embeddings, IDs, CLS)
vallidation_dataset_BC = torch.utils.data.TensorDataset(validation_set.transpose(0,1), torch.Tensor(compatibility_valid))
validloader_BC = DataLoader(vallidation_dataset_BC, batch_size=8, num_workers=0, shuffle=True)
print("Validation set for binary classifiation created")
validation_set_MLM, masked_positions_valid, actual_masked_values_valid = masking_input(validation_set, valid_dataframe,
                                                                                       MASK, with_CLS=False)
MLM_valid_labels = torch.Tensor(actual_masked_values_valid).reshape(len(actual_masked_values_valid), 1)
masked_positions_tensor_valid = torch.Tensor(masked_positions_valid).reshape(len(masked_positions_valid), 1)
valid_labels_MLM = torch.concat((MLM_valid_labels,masked_positions_tensor_valid), dim=1)
validation_dataset_MLM = torch.utils.data.TensorDataset(validation_set_MLM, valid_labels_MLM)
validloader_MLM = DataLoader(validation_dataset_MLM, batch_size=8, num_workers=0, shuffle=True)
print("Validation set for masked language model created")

# import the test set

# test_dataframe = pd.read_csv('../reduced_data/reduced_compatibility_test.csv')
# print(f'target variable count in test_dataframe : {test_dataframe["compatibility"].value_counts()}')
# compatibility_test = test_dataframe['compatibility'].values
# test_dataframe.drop(columns='compatibility', inplace=True)

# print("Creating the test set")
# test_set = create_tensor_dataset_for_BC_from_dataframe(test_dataframe, embeddings, IDs, CLS)
# test_dataset_BC = torch.utils.data.TensorDataset(test_set.transpose(0,1), torch.Tensor(compatibility_test))
# testloader_BC = DataLoader(test_dataset_BC, batch_size=8, num_workers=0, shuffle=True)
# print("Test set for binary classifiation created")
# test_set_MLM, masked_positions_test, actual_masked_values_test = masking_input(test_set, test_dataframe, MASK,
#                                                                                with_CLS=False)
# MLM_test_labels = torch.Tensor(actual_masked_values_test).reshape(len(actual_masked_values_test), 1)
# masked_positions_tensor_test = torch.Tensor(masked_positions_test).reshape(len(masked_positions_test), 1)
# test_labels_MLM = torch.concat((MLM_test_labels,masked_positions_tensor_test), dim=1)
# test_dataset_MLM = torch.utils.data.TensorDataset(test_set_MLM, test_labels_MLM)
# testloader_MLM = DataLoader(test_dataset_MLM, batch_size=8, num_workers=0, shuffle=True)
# print("Test set for masked language model created")

# create the dictionary to work with modularity
dataloaders_BC = {'train': trainloader_BC, 'val': validloader_BC}
dataloaders_MLM = {'train': trainloader_MLM, 'val': validloader_MLM}
# masked_indices = {'train': masked_positions_train, 'val': masked_positions_valid} # questo qui non dovrebbe servire più
# masked_labels = {'train': actual_masked_values_train, 'val': actual_masked_values_valid}
# compatibility = {'train': compatibility_train, 'val': compatibility_valid}
# define the optimizer
optimizer = Adam(params=model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01)
# put the model on the GPU
model.to(device)
criterion_1 = nn.BCEWithLogitsLoss()

print('Start pre-training BC the model')

trainer = umBERT_trainer(model=model, optimizer=optimizer, criterion=criterion_1, device=device, n_epochs=100)
trainer.pre_train_BC(dataloaders=dataloaders_BC)
print('Pre-training on BC completed')

torch.save(model.state_dict(), '../models/umBERT_pretrained_1.pth')

criterion_2 = nn.CrossEntropyLoss()
trainer.criterion = criterion_2
print('Start pre-training MLM the model')
trainer.pre_train_MLM(dataloaders=dataloaders_MLM)
print('Pre-training on MLM completed')

# save the model into a checkpoint file
torch.save(model.state_dict(), '../models/umBERT_pretrained_2.pth')
