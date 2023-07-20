import random
import os
import json
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss, CosineEmbeddingLoss
from torch.optim import Adam
from BERT_architecture.umBERT2 import umBERT2
from torch.utils.data import DataLoader
from utility.create_tensor_dataset_for_BC_from_dataframe import create_tensor_dataset_for_BC_from_dataframe
from utility.umBERT2_trainer import umBERT2_trainer
from  constants import *

# set the seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
SEED = 42

dim_embeddings = 128

# set the working directory to the path of the file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# use GPU if available
device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
print('Device used: ', device)

# read the dataset
df = pd.read_csv('../reduced_data/reduced_compatibility.csv')

# load the IDs of the images
with open("../reduced_data/IDs_list", "r") as fp:
    IDs = json.load(fp)
# load the embeddings
with open(f'../reduced_data/embeddings_{str(dim_embeddings)}.npy', 'rb') as f:
    embeddings = np.load(f)

# generate the CLS and MASK embeddings
CLS, _ = generate_special_embeddings_randomly(dim_embeddings)  # generate the CLS embedding randomly
MASK = initialize_mask_embedding_zeros(dim_embeddings)  # initialize the MASK embedding to zeros

# split the dataset in train, valid and test set (80%, 10%, 10%) in a stratified way on the compatibility column
compatibility = df['compatibility'].values
df = df.drop(columns=['compatibility'])
df_train, df_test, compatibility_train, compatibility_test = train_test_split(df, compatibility, test_size=0.2,
                                                                              stratify=compatibility,
                                                                              random_state=42,
                                                                              shuffle=True)
df_valid, df_test, compatibility_valid, compatibility_test = train_test_split(df_test, compatibility_test,
                                                                              test_size=0.5,
                                                                              stratify=compatibility_test,
                                                                              random_state=42,
                                                                              shuffle=True)
# reset the index of the dataframes
df_train = df_train.reset_index(drop=True)
df_valid = df_valid.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# reset index of the labels

# create the tensor dataset for the training set (which contains the CLS embedding)
print('Creating the tensor dataset for the training set...')
training_set = create_tensor_dataset_for_BC_from_dataframe(df_train, embeddings, IDs, CLS)
# scale the training set using z-score (layer normalization + batch normalization)
print('Scaling the training set using z-score...')
CLS_layer_train = training_set[0, :, :]  # CLS is the first element of the tensor, preserve it
training_set = training_set[1:, :, :]  # remove CLS from the tensor
mean = training_set.mean(dim=0).mean(dim=0)
std = training_set.std(dim=0).std(dim=0)
torch.save(mean, '../reduced_data/mean.pth')
torch.save(std, '../reduced_data/std.pth')
training_set = (training_set - mean) / std
training_set = torch.cat((CLS_layer_train.unsqueeze(0), training_set), dim=0)  # concatenate CLS to the scaled tensor

# # create the labels tensors for the task of compatibility classification and embedding reconstruction
# # convert the compatibility labels to tensor
# compatibility_train = torch.tensor(compatibility_train.values)
# # from df_train, create a tensor containing the IDs of the items in the training set
# IDs_train = torch.tensor(df_train.values)
# # concatenate the compatibility labels and the IDs tensors
# labels_train = torch.cat((compatibility_train.unsqueeze(1), IDs_train), dim=1)
# # create the Tensor Dataset for the training set
# training_set = training_set.transpose(0, 1)  # transpose the tensor to have the batch dimension first
# train_dataset = torch.utils.data.TensorDataset(training_set, labels_train)
# # create the DataLoader for the training set

# # repeat the same operations for the validation set
# print('Creating the tensor dataset for the validation set...')
# validation_set = create_tensor_dataset_for_BC_from_dataframe(df_valid, embeddings, IDs, CLS)
# print('Scaling the validation set using z-score...')
# CLS_layer_valid = validation_set[0, :, :]
# validation_set = validation_set[1:, :, :]
# validation_set = (validation_set - mean) / std
# validation_set = torch.cat((CLS_layer_valid.unsqueeze(0), validation_set), dim=0)
# compatibility_valid = torch.tensor(compatibility_valid.values)
# IDs_valid = torch.tensor(df_valid.values)
# labels_valid = torch.cat((compatibility_valid.unsqueeze(1), IDs_valid), dim=1)
# validation_set = validation_set.transpose(0, 1)
# valid_dataset = torch.utils.data.TensorDataset(validation_set, labels_valid)


# repeat the same operations for the test set
print('Creating the tensor dataset for the test set...')
test_set = create_tensor_dataset_for_BC_from_dataframe(df_test, embeddings, IDs, CLS)
print('Scaling the test set using z-score...')
CLS_layer_test = test_set[0, :, :]
test_set = test_set[1:, :, :]
test_set = (test_set - mean) / std
test_set = torch.cat((CLS_layer_test.unsqueeze(0), test_set), dim=0)
compatibility_test = torch.tensor(compatibility_test)
IDs_test = torch.tensor(df_test.values)
labels_test = torch.cat((compatibility_test.unsqueeze(1), IDs_test), dim=1)
test_set = test_set.transpose(0, 1)
test_dataset = torch.utils.data.TensorDataset(test_set, labels_test)

params = {
    'lr': 1e-4,
    'batch_size': 256,
    'n_epochs': 50,
    'dropout': 0.036958272662759584,
    'd_model': dim_embeddings,
    'num_encoders': 8,
    'num_heads': 1,
    'weight_decay': 0.04037776702619792
}

# define the umBERT2 model
model = umBERT2(d_model=dim_embeddings, num_encoders=params['num_encoders'],
                num_heads=params['num_heads'], dropout=params['dropout'])

# use optimizer as suggested in the bayesian optimization
optimizer = Adam(params=model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

criterion = {'clf': CrossEntropyLoss(), 'recons': CosineEmbeddingLoss()}

print('Start pre-training the model')
trainer = umBERT2_trainer(model=model, optimizer=optimizer, criterion=criterion,
                          device=device, n_epochs=params['n_epochs'])
#
# # define dataloaders
# train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=0)
# valid_dataloader = DataLoader(valid_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=0)

# test the model
print('Start testing the pre-trained model...')
# define the dataloader
test_dataloader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=0)
#
# # create the dictionary containing the dataloaders
# dataloaders = {'train': train_dataloader, 'val': test_dataloader}
#
# loss_pre_train = trainer.pre_train(dataloaders=dataloaders, run=None)
# print(f'Pre-training completed with loss {loss_pre_train}')
checkpoint = torch.load('../models/umBERT2_pre_trained_128.pth')
model.load_state_dict(checkpoint['model_state_dict'])
# update the model
trainer.model = model.to(device)  # move the model to the device
# compute the predictions of the model
trainer.evaluate_pre_training(dataloader=test_dataloader)
