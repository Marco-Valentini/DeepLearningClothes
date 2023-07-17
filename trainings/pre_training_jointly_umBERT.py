import json
import random

import neptune
from hyperopt import hp, tpe, Trials, STATUS_OK, fmin
from lion_pytorch import Lion

from BERT_architecture.umBERT import umBERT
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
import os
from utility.create_tensor_dataset_for_BC_from_dataframe import create_tensor_dataset_for_BC_from_dataframe
from utility.masking_input import masking_input
from utility.dataset_augmentation import create_permutations_per_outfit
import numpy as np
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from utility.umBERT_trainer import umBERT_trainer
from constants import generate_special_embeddings_randomly

# set the seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)

# import the MASK and CLS tokens
dim_embeddings = 512
CLS, MASK = generate_special_embeddings_randomly(dim_embeddings)

# set the working directory to the path of the file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# define the run for monitoring the training on Neptune dashboard
run = neptune.init_run(
    project="marcopaolodeeplearning/DeepLearningClothes",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjMTY5ZDBlZC1kY2QzLTQzNDYtYjc0OS02YzkzM2M3YjIyOTAifQ==",
)  # your credentials

# use GPU if available
device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
print('Using device:', device)

catalogue = pd.read_csv('../reduced_data/reduced_catalogue.csv')  # load the catalogue

# first step: load the embeddings of the dataset obtained by the fine-tuned model finetuned_fashion_resnet18_512.pth
with open("../reduced_data/IDs_list", "r") as fp:
    IDs = json.load(fp)
print("IDs loaded")

with open('../reduced_data/embeddings_512_old.npy', 'rb') as f:
    embeddings = np.load(f)

print("Embeddings loaded")

# import the training set
train_dataframe = pd.read_csv('../reduced_data/reduced_compatibility_train.csv')
compatibility_train = train_dataframe['compatibility'].values
train_dataframe.drop(columns='compatibility', inplace=True)

# create the tensor dataset for the training set (which contains the CLS embedding)
print('Creating the tensor dataset for the training set...')
training_set = create_tensor_dataset_for_BC_from_dataframe(train_dataframe, embeddings, IDs, CLS)
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
# augment the training set with all the permutations of the outfits
print('Augmenting the training set with all the permutations of the outfits...')
training_set = create_permutations_per_outfit(training_set)
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
print('Scaling the validation set using z-score...')
CLS_layer_val = validation_set[0, :, :]  # CLS is the first element of the tensor, preserve it
validation_set = validation_set[1:, :, :]  # remove CLS from the tensor
validation_set = (validation_set - mean) / std
validation_set = torch.cat((CLS_layer_val.unsqueeze(0), validation_set), dim=0)  # concatenate CLS to the scaled tensor
# augment the validation set with all the permutations of the outfits
print('Augmenting the validation set with all the permutations of the outfits...')
validation_set = create_permutations_per_outfit(validation_set)
# scale the validation set using z-score (layer normalization) (using the mean and std of the training set)
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

# create the dictionary containing the dataloaders, the masked indices, the masked labels and the compatibility labels
# for the training and validation set
dataloaders = {'train': trainloader, 'val': validloader}
masked_indices = {'train': masked_indexes_train, 'val': masked_indexes_valid}

### hyperparameters tuning ###
print('Starting hyperparameters tuning...')
# define the maximum number of evaluations
max_evals = 10
# define the search space
possible_n_heads = [1, 2, 4, 8]
possible_n_encoders = [i for i in range(1, 12)]
possible_n_epochs = [20, 50, 100]
possible_batch_size = [16, 32, 64]
possible_optimizers = [Adam, AdamW, Lion]

space = {
    'lr': hp.uniform('lr', 1e-5, 1e-2),
    'batch_size': hp.choice('batch_size', possible_batch_size),
    'n_epochs': hp.choice('n_epochs', possible_n_epochs),
    'dropout': hp.uniform('dropout', 0, 0.5),
    'num_encoders': hp.choice('num_encoders', possible_n_encoders),
    'num_heads': hp.choice('num_heads', possible_n_heads),
    'weight_decay': hp.uniform('weight_decay', 0, 0.1),
    'optimizer': hp.choice('optimizer', possible_optimizers)
}

# define the algorithm
tpe_algorithm = tpe.suggest

# define the trials object
baeyes_trials = Trials()

# define the objective function
def objective(params):
    # define the model
    model = umBERT(catalogue_size=catalogue['ID'].size, d_model=dim_embeddings, num_encoders=params['num_encoders'],
                    num_heads=params['num_heads'], dropout=params['dropout'])
    # define the optimizer
    optimizer = params['optimizer'](params=model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    # define the criteria
    criterion = CrossEntropyLoss()
    # define the trainer
    trainer = umBERT_trainer(model=model, optimizer=optimizer, criterion=criterion, device=device, n_epochs=params['n_epochs'])
    # train the model
    loss = trainer.pre_train_BERT_like(dataloaders=dataloaders, run=None)

    # return the loss and the accuracy
    return {'loss': loss, 'params': params, 'status': STATUS_OK}

# optimize
best = fmin(fn=objective, space=space, algo=tpe_algorithm, max_evals=max_evals, trials=baeyes_trials)

# optimal model
print('hyperparameters tuning completed!')
print(f'the best hyperparameters combination is: {best}')

# define the parameters
params = {
    'lr': best['lr'],
    'batch_size': possible_batch_size[best['batch_size']],
    'n_epochs': possible_n_epochs[best['n_epochs']],
    'dropout': best['dropout'],
    'd_model': dim_embeddings,
    'num_encoders': possible_n_encoders[best['num_encoders']],
    'num_heads': possible_n_heads[best['num_heads']],
    'weight_decay': best['weight_decay']
}

run["parameters"] = params

# define the umBERT model
model = umBERT(catalogue_size=catalogue['ID'].size, d_model=dim_embeddings, num_encoders=params['num_encoders'],
               num_heads=params['num_heads'], dropout=params['dropout'])

# use Adam as optimizer as suggested in the paper
optimizer = possible_optimizers[best['optimizer']](params=model.parameters(), lr=params['lr'],
                                                   betas=(0.9, 0.999), weight_decay=params['weight_decay'])
criterion = CrossEntropyLoss()

print('Start pre-training the model')
trainer = umBERT_trainer(model=model, optimizer=optimizer, criterion=criterion,
                         device=device, n_epochs=params['n_epochs'])
trainer.pre_train_BERT_like(dataloaders=dataloaders, run=run)
run.stop()
print('Pre-training completed')
