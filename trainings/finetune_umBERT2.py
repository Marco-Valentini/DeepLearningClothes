# start the fine-tuning
# define the run for monitoring the training on Neptune dashboard
import json
import os
import random

import neptune
import numpy as np
import pandas as pd
import torch
from hyperopt import hp, fmin, STATUS_OK, tpe, Trials
from lion_pytorch import Lion
from sklearn.model_selection import train_test_split
from torch.nn import CosineEmbeddingLoss, CrossEntropyLoss
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader

from BERT_architecture.umBERT2 import umBERT2
from constants import generate_special_embeddings_randomly
from utility.create_tensor_dataset_for_BC_from_dataframe import create_tensor_dataset_for_BC_from_dataframe
from utility.dataset_augmentation import mask_one_item_per_time
from utility.umBERT2_trainer import umBERT2_trainer

# set the seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
SEED = 42

# import the MASK and CLS tokens
dim_embeddings = 128
CLS, MASK = generate_special_embeddings_randomly(dim_embeddings)

# set the working directory to the path of the file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

run = neptune.init_run(
    project="marcopaolodeeplearning/DeepLearningOutfitCompetion",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjMTY5ZDBlZC1kY2QzLTQzNDYtYjc0OS02YzkzM2M3YjIyOTAifQ==", # your credentials
    name="fine-tuning umBERT2",
    tags=["umBERT2", "fine-tuning"],
)  # your credentials

# use GPU if available
device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
print('Device used: ', device)

with open("../reduced_data/IDs_list", "r") as fp:
    IDs = json.load(fp)
print("IDs loaded")

with open(f'../reduced_data/embeddings_{dim_embeddings}.npy', 'rb') as f:
    embeddings = np.load(f)

# load the model
# define the umBERT model
checkpoint = torch.load(f'../models/umBERT2_pre_trained_{dim_embeddings}.pth')

model = umBERT2(d_model=checkpoint['d_model'],
                num_encoders=checkpoint['num_encoders'],
                num_heads=checkpoint['num_heads'],
                dropout=checkpoint['dropout'],
                dim_feedforward=checkpoint['dim_feedforward'])
# load the model weights
model.load_state_dict(checkpoint['model_state_dict'])

print('Starting the search for fine-tuning the model...')
# load the fine-tuning dataset
df_fine_tuning = pd.read_csv('../reduced_data/unified_dataset_MLM.csv')

# split the dataset into training, validation set and test set
df_fine_tuning_train, df_fine_tuning_test = train_test_split(df_fine_tuning, test_size=0.2, random_state=42, shuffle=True)
df_fine_tuning_valid, df_fine_tuning_test = train_test_split(df_fine_tuning_test, test_size=0.5, random_state=42, shuffle=True)
# reset the indexes
df_fine_tuning_train = df_fine_tuning_train.reset_index(drop=True)
df_fine_tuning_valid = df_fine_tuning_valid.reset_index(drop=True)
df_fine_tuning_test = df_fine_tuning_test.reset_index(drop=True)

# create the tensor dataset for the training set
print('Creating the tensor dataset for the training set...')
FT_training_set = create_tensor_dataset_for_BC_from_dataframe(df_fine_tuning_train, embeddings, IDs, CLS)
print('Tensor dataset for the training set created!')
# remove the CLS
CLS_layer_train_FT = FT_training_set[0, :, :]
FT_training_set = FT_training_set[1:, :, :]
print("Scaling the training set using the z-score...")
FT_mean = FT_training_set.mean(dim=0).mean(dim=0)
FT_std = FT_training_set.std(dim=0).std(dim=0)
torch.save(FT_mean, '../models/FT_mean.pth')
torch.save(FT_std, '../models/FT_std.pth')
FT_training_set = (FT_training_set - FT_mean) / FT_std
# re-attach the CLS
FT_training_set = torch.cat((CLS_layer_train_FT.unsqueeze(0), FT_training_set), dim=0)
print("masking the items...")
train_masked_outfit, train_masked_indexes, train_masked_IDs = mask_one_item_per_time(FT_training_set,
                                                                                     df_fine_tuning_train,
                                                                                     MASK,
                                                                                     input_contains_CLS=True,
                                                                                     device=device,
                                                                                     output_in_batch_first=True)
train_masked_outfit = torch.Tensor(train_masked_outfit)
train_masked_indexes = torch.Tensor(train_masked_indexes).unsqueeze(1)
train_masked_IDs = torch.Tensor(train_masked_IDs).unsqueeze(1)
print("items masked!")
# labels are the masked items IDs
# repeat each element of df_fine_tuning_train 4 times
df_fine_tuning_train = pd.DataFrame(np.repeat(df_fine_tuning_train.values, 4, axis=0), columns=df_fine_tuning_train.columns)

FT_IDs_train = torch.Tensor(df_fine_tuning_train.values)
print(f'shape FT_IDs_train: {FT_IDs_train.shape}')
print(f'shape train_masked_indexes: {train_masked_indexes.shape}')
print(f'shape train_masked_IDs: {train_masked_IDs.shape}')
FT_labels_train = torch.cat((FT_IDs_train, train_masked_indexes, train_masked_IDs), dim=1)
FT_training_dataset = torch.utils.data.TensorDataset(train_masked_outfit, FT_labels_train)
print("dataset created!")

# repeat the same operations for validation set
print('Creating the tensor dataset for the validation set...')
FT_valid_set = create_tensor_dataset_for_BC_from_dataframe(df_fine_tuning_valid, embeddings, IDs, CLS)
print('Tensor dataset for the validation set created!')
# remove the CLS
CLS_layer_valid_FT = FT_valid_set[0, :, :]
FT_valid_set = FT_valid_set[1:, :, :]
print("Scaling the validation set using the z-score...")
FT_valid_set = (FT_valid_set - FT_mean) / FT_std
# re-attach the CLS
FT_valid_set = torch.cat((CLS_layer_valid_FT.unsqueeze(0), FT_valid_set), dim=0)
print("masking the items...")
valid_masked_outfit, valid_masked_indexes, valid_masked_IDs = mask_one_item_per_time(FT_valid_set,
                                                                                     df_fine_tuning_valid,
                                                                                     MASK,
                                                                                     input_contains_CLS=True,
                                                                                     device=device,
                                                                                     output_in_batch_first=True)

valid_masked_outfit = torch.Tensor(valid_masked_outfit)
valid_masked_indexes = torch.Tensor(valid_masked_indexes).unsqueeze(1)
valid_masked_IDs = torch.Tensor(valid_masked_IDs).unsqueeze(1)
print("items masked!")
# labels are the masked items IDs
# repeat each element of df_fine_tuning_train 4 times
df_fine_tuning_valid = pd.DataFrame(np.repeat(df_fine_tuning_valid.values, 4, axis=0), columns=df_fine_tuning_valid.columns)
FT_IDs_valid = torch.Tensor(df_fine_tuning_valid.values)
FT_labels_valid = torch.cat((FT_IDs_valid, valid_masked_indexes, valid_masked_IDs), dim=1)
FT_valid_dataset = torch.utils.data.TensorDataset(valid_masked_outfit, FT_labels_valid)
print("dataset created!")

# apply baesyan search to find the best hyperparameters combination
### hyperparameters tuning ###
print('Starting hyperparameters tuning with baesyan search for fine-tuning...')
# define the maximum number of evaluations
max_evals = 10
# define the search space
possible_lr = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
possible_n_epochs = [10, 20, 50]
possible_batch_size = [32, 64, 256, 512]
possible_optimizers = [Adam, AdamW, Lion]

space = {
    'lr': hp.choice('lr', possible_lr),
    'batch_size': hp.choice('batch_size', possible_batch_size),
    'n_epochs': hp.choice('n_epochs', possible_n_epochs),
    'weight_decay': hp.uniform('weight_decay', 0, 0.1),
    'optimizer': hp.choice('optimizer', possible_optimizers)
}
# define the algorithm
tpe_algorithm_FT = tpe.suggest

# define the trials object
baeyes_trials_FT = Trials()

# call the fine-tune training
# define the objective function
def objective_fine_tuning(params):
    print(f"Fine-tuning with params: {params}")
    # define the optimizer
    optimizer = params['optimizer'](params=model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    # define the criteria
    criterion = {'clf': CrossEntropyLoss(), 'recons': CosineEmbeddingLoss()}
    # define the trainer
    trainer = umBERT2_trainer(model=model, optimizer=optimizer, criterion=criterion,
                              device=device, n_epochs=params['n_epochs'])
    # create the data loader
    FT_training_dataloader = DataLoader(FT_training_dataset, batch_size=params['batch_size'],
                                        shuffle=True, num_workers=0)
    FT_valid_dataloader = DataLoader(FT_valid_dataset, batch_size=params['batch_size'],
                                     shuffle=True, num_workers=0)
    FT_data_loaders = {'train': FT_training_dataloader, 'val': FT_valid_dataloader}

    # train the model
    accuracy = trainer.fine_tuning(dataloaders=FT_data_loaders, run=None)

    loss = 1 - accuracy

    # return the loss and the accuracy
    return {'loss': loss, 'params': params, 'status': STATUS_OK}


# optimize
best_fine_tuning = fmin(fn=objective_fine_tuning, space=space, algo=tpe_algorithm_FT, max_evals=max_evals, trials=baeyes_trials_FT, rstate=np.random.default_rng(SEED), verbose=True)

# optimal model
print('hyperparameters FINE tuning completed!')
print(f'the best hyperparameters combination in fine tuning is: {best_fine_tuning}')

# define the parameters
params = {
    'lr': possible_lr[best_fine_tuning['lr']],
    'batch_size': possible_batch_size[best_fine_tuning['batch_size']],
    'n_epochs': possible_n_epochs[best_fine_tuning['n_epochs']],
    'weight_decay': best_fine_tuning['weight_decay']
}

run["parameters"] = params

# use optimizer as suggested in the bayesian optimization
optimizer = possible_optimizers[best_fine_tuning['optimizer']](params=model.parameters(), lr=params['lr'],
                                                               weight_decay=params['weight_decay'])

criterion = {'recons': CosineEmbeddingLoss()}

print('Starting fine-tuning the model')
trainer = umBERT2_trainer(model=model, optimizer=optimizer, criterion=criterion,
                          device=device, n_epochs=params['n_epochs'])

# create the data loader
FT_training_dataloader = DataLoader(FT_training_dataset, batch_size=params['batch_size'],
                                    shuffle=True, num_workers=0)
FT_valid_dataloader = DataLoader(FT_valid_dataset, batch_size=params['batch_size'],
                                 shuffle=True, num_workers=0)
FT_data_loaders = {'train': FT_training_dataloader, 'val': FT_valid_dataloader}

trainer.fine_tuning(dataloaders=FT_data_loaders, run=run)
run.stop()  # stop the run on wandb website and save the results in the folder specified in the config file
print('Fine-tuning completed')

# test the model
print('Start testing the model...')
# repeat the same operations for test set
print('Creating the tensor dataset for the test set...')
FT_test_set = create_tensor_dataset_for_BC_from_dataframe(df_fine_tuning_test, embeddings, IDs, CLS)
print('Tensor dataset for the test set created!')
# remove the CLS
CLS_layer_test_FT = FT_test_set[0, :, :]
FT_test_set = FT_test_set[1:, :, :]
print("Scaling the test set using the z-score...")
FT_test_set = (FT_test_set - FT_mean) / FT_std
# re-attach the CLS
FT_test_set = torch.cat((CLS_layer_test_FT.unsqueeze(0), FT_test_set), dim=0)
print("masking the items...")
test_masked_outfit, test_masked_indexes, test_masked_IDs = mask_one_item_per_time(FT_test_set,
                                                                                  df_fine_tuning_test,
                                                                                  MASK,
                                                                                  input_contains_CLS=True,
                                                                                  device=device,
                                                                                  output_in_batch_first=True)
test_masked_outfit = torch.Tensor(test_masked_outfit)
test_masked_indexes = torch.Tensor(test_masked_indexes)
test_masked_IDs = torch.Tensor(test_masked_IDs)
print("items masked!")
# repeat the labels row
df_fine_tuning_test = pd.DataFrame(np.repeat(df_fine_tuning_test.values, 4, axis=0), columns=df_fine_tuning_test.columns)
# labels are the masked items IDs
FT_IDs_test = torch.Tensor(df_fine_tuning_test.values)
FT_labels_test = torch.cat((FT_IDs_test, test_masked_indexes, test_masked_IDs))  # TODO capire come gestire queste shape
FT_test_dataset = torch.utils.data.TensorDataset(FT_test_set, FT_labels_test)
print("dataset created!")
# create the data loader for test
FT_test_dataloader = DataLoader(FT_test_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=0)