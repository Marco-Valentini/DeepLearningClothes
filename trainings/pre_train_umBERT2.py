import neptune
from hyperopt import Trials
from hyperopt import hp
from hyperopt import fmin
from hyperopt import tpe
from hyperopt import STATUS_OK
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss, CosineEmbeddingLoss
import random
from BERT_architecture.umBERT2 import umBERT2
import pandas as pd
import torch
from torch.utils.data import DataLoader
from lion_pytorch import Lion
from torch.optim import Adam, AdamW
import os
from utility.create_tensor_dataset_for_BC_from_dataframe import create_tensor_dataset_for_BC_from_dataframe
import numpy as np
from utility.get_category_labels import get_category_labels
from utility.masking_input import masking_input
from utility.umBERT2_trainer import umBERT2_trainer
import json
from constants import get_special_embeddings

# define the run for monitoring the training on Neptune dashboard
run = neptune.init_run(
    project="marcopaolodeeplearning/DeepLearningClothes",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjMTY5ZDBlZC1kY2QzLTQzNDYtYjc0OS02YzkzM2M3YjIyOTAifQ==",
)  # your credentials

# set the seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)

# import the MASK and CLS tokens
dim_embeddings = 64
CLS, MASK = get_special_embeddings(dim_embeddings)

# set the working directory to the path of the file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# use GPU if available
device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
print('Device used: ', device)

# read the dataset
df = pd.read_csv('../../dataset/reduced_compatibility.csv')

# split the dataset in train, valid and test set (80%, 10%, 10%) in a stratified way on the compatibility column
compatibility = df['compatibility']
df = df.drop(columns=['compatibility'])
df_train, df_test, compatibility_train, compatibility_test = train_test_split(df, compatibility, test_size=0.2,
                                                                              stratify=compatibility)
df_valid, df_test, compatibility_valid, compatibility_test = train_test_split(df_test, compatibility_test,
                                                                              test_size=0.5,
                                                                              stratify=compatibility_test)

with open("../reduced_data/IDs_list", "r") as fp:
    IDs = json.load(fp)
print("IDs loaded")

with open('../reduced_data/embeddings_512_old.npy', 'rb') as f:
    embeddings = np.load(f)

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

# create the labels tensors for the task of compatibility classification and embedding reconstruction
# convert the compatibility labels to tensor
compatibility_train = torch.tensor(compatibility_train.values)
# from df_train, create a tensor containing the IDs of the items in the training set
IDs_train = torch.tensor(df_train.values)
# concatenate the compatibility labels and the IDs tensors
labels_train = torch.cat((compatibility_train.unsqueeze(1), IDs_train), dim=1)
# create the Tensor Dataset for the training set
train_dataset = torch.utils.data.TensorDataset(training_set, labels_train)
# create the DataLoader for the training set
print('Creating the DataLoader for the training set...')
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

# repeat the same operations for the validation set
print('Creating the tensor dataset for the validation set...')
validation_set = create_tensor_dataset_for_BC_from_dataframe(df_valid, embeddings, IDs, CLS)
print('Scaling the validation set using z-score...')
CLS_layer_valid = validation_set[0, :, :]
validation_set = validation_set[1:, :, :]
validation_set = (validation_set - mean) / std
validation_set = torch.cat((CLS_layer_valid.unsqueeze(0), validation_set), dim=0)
compatibility_valid = torch.tensor(compatibility_valid.values)
IDs_valid = torch.tensor(df_valid.values)
labels_valid = torch.cat((compatibility_valid.unsqueeze(1), IDs_valid), dim=1)
valid_dataset = torch.utils.data.TensorDataset(validation_set, labels_valid)
print('Creating the DataLoader for the validation set...')
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True, num_workers=0)

# create the dictionary containing the dataloaders
dataloaders = {'train': train_dataloader, 'val': valid_dataloader}

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
    model = umBERT2(d_model=dim_embeddings, num_encoders=params['num_encoders'],
                    num_heads=params['num_heads'], dropout=params['dropout'])
    # define the optimizer
    optimizer = params['optimizer'](params=model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    # define the criteria
    criterion = {'clf': CrossEntropyLoss(), 'recons': CosineEmbeddingLoss()}
    # define the trainer
    trainer = umBERT2_trainer(model=model, optimizer=optimizer, criterion=criterion,
                              device=device, n_epochs=params['n_epochs'])
    # train the model
    loss = trainer.pre_train(dataloaders=dataloaders, run=None)

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

# define the umBERT2 model
model = umBERT2(catalogue_sizes=catalogue_sizes, d_model=dim_embeddings, num_encoders=params['num_encoders'],
                num_heads=params['num_heads'], dropout=params['dropout'])

# use Adam as optimizer as suggested in the paper
optimizer = possible_optimizers[best['optimizer']](params=model.parameters(), lr=params['lr'],
                                                   betas=(0.9, 0.999), weight_decay=params['weight_decay'])

criterion = {'clf': CrossEntropyLoss(), 'recons': CosineEmbeddingLoss()}

print('Start pre-training the model')
trainer = umBERT2_trainer(model=model, optimizer=optimizer, criterion=criterion,
                          device=device, n_epochs=params['n_epochs'])

trainer.pre_train(dataloaders=dataloaders, run=run)
run.stop()  # stop the run on wandb website and save the results in the folder specified in the config file
print('Pre-training completed')
