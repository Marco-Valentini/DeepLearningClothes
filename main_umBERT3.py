# pretraining task 1: Binary Classification (using compatibility dataset)
# pretraining task 2: Reconstruction of the inputs with random masking (using unified dataset MLM)
# fine-tuning task: Reconstruction of the inputs with one mask per time
import random
import neptune
import torch
import os
import json
import numpy as np
import pandas as pd
from BERT_architecture.umBERT3 import umBERT3 as umBERT
from hyperopt import Trials, hp, fmin, tpe, STATUS_OK
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.optim import Adam, AdamW
from lion_pytorch import Lion
from torch.nn import CrossEntropyLoss


# utility functions
def create_tensor_dataset_from_dataframe(df_outfit: pd.DataFrame, embeddings, ids):
    """
    This function takes as input a dataframe containing the labels of the items in the outfit, the embeddings of the items and the ids of the items.
    It returns a tensor of shape (n_outfits, seq_len, embedding_size) containing the embeddings of the items in the outfit and the CLS token.
    :param df_outfit:  dataframe containing the labels of the items in the outfit
    :param embeddings:  embeddings of the items in the outfit (a tensor of shape (n_items, embedding_size))
    :param ids:  ids of the items in the outfit (a list of length n_items)
    :param CLS: the embedding token CLS (a tensor of shape (1, embedding_size))
    :return: a tensor of shape (n_outfits,seq_len, embedding_size) containing the embeddings of the items in the outfit and the CLS token
    """
    dataset = np.zeros((df_outfit.shape[0], 4, embeddings.shape[1]))
    for i in range(df_outfit.shape[0]):  # for each outfit
        for j in range(df_outfit.shape[1]):  # for each item in the outfit
            ID = df_outfit.iloc[i, j]
            index_item = ids.index(ID)
            embedding = embeddings[index_item]
            dataset[i, j, :] = embedding
            # do not transpose to not create conflicts with masking input
    return torch.Tensor(dataset)


# set the seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
SEED = 42

dim_embeddings = 64

# set the working directory to the path of the file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# use GPU if available
device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
print('Device used: ', device)

# pre-training task #1: Binary Classification (using compatibility dataset)
# load the compatibility dataset
print('Loading the compatibility dataset...')
df = pd.read_csv('./reduced_data/reduced_compatibility.csv')
print('Compatibility dataset loaded!')
# load the IDs of the images
with open("./reduced_data/IDs_list", "r") as fp:
    IDs = json.load(fp)
# load the embeddings
with open(f'./reduced_data/embeddings_{str(dim_embeddings)}.npy', 'rb') as f:
    embeddings = np.load(f)

# split the dataset in train, valid and test set (80%, 10%, 10%) in a stratified way on the compatibility column
print("creating the datasets for BC pre-training...")
compatibility = df['compatibility'].values
df = df.drop(columns=['compatibility'])
df_train, df_test, compatibility_train, compatibility_test = train_test_split(df, compatibility, test_size=0.2,
                                                                              stratify=compatibility,
                                                                              random_state=42,
                                                                              shuffle=True)
# reset the index of the dataframes
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
# from the dataframes create the tensor datasets
tensor_dataset_train = create_tensor_dataset_from_dataframe(df_train, embeddings,
                                                            IDs)  # shape (n_outfits, seq_len, embedding_size)
tensor_dataset_test = create_tensor_dataset_from_dataframe(df_test, embeddings, IDs)
# from the training set compute mean and std to normalize both the tests
mean_pre_training_BC = tensor_dataset_train.mean(dim=1).mean(dim=0)
std_pre_training_BC = tensor_dataset_train.std(dim=1).std(dim=0)
# normalize the datasets with z-score
tensor_dataset_train = (tensor_dataset_train - mean_pre_training_BC) / std_pre_training_BC
tensor_dataset_test = (tensor_dataset_test - mean_pre_training_BC) / std_pre_training_BC
# split the training into train and valid set
tensor_dataset_train, tensor_dataset_valid, compatibility_train, compatibility_valid = train_test_split(
    tensor_dataset_train, compatibility_train, test_size=0.2,
    stratify=compatibility_train,
    random_state=42,
    shuffle=True)

print("dataset for BC created")
# create the dataloaders
print("creating dataloaders...")
train_dataloader_pre_training_BC = DataLoader(TensorDataset(tensor_dataset_train, torch.Tensor(compatibility_train)),
                                              batch_size=32,
                                              shuffle=True, num_workers=0)
valid_dataloader_pre_training_BC = DataLoader(TensorDataset(tensor_dataset_valid, torch.Tensor(compatibility_valid)),
                                              batch_size=32,
                                              shuffle=True, num_workers=0)
test_dataloader_pre_training_BC = DataLoader(TensorDataset(tensor_dataset_test, torch.Tensor(compatibility_test)),
                                             batch_size=32, shuffle=True,
                                             num_workers=0)
print("dataloaders for pre-training task #1 created!")

# prepare the data for the pre-training task #2
# load the unified dataset
print('Loading the unified dataset...')
df_2 = pd.read_csv('./reduced_data/unified_dataset_MLM.csv')
print('Unified dataset loaded!')

df_2_train, df_2_test = train_test_split(df_2, test_size=0.2,
                                         random_state=42,
                                         shuffle=True)
# reset the index of the dataframes
df_2_train = df_2_train.reset_index(drop=True)
df_2_test = df_2_test.reset_index(drop=True)
# from the dataframes create the tensor datasets
tensor_dataset_train_2 = create_tensor_dataset_from_dataframe(df_2_train, embeddings,
                                                            IDs)  # shape (n_outfits, seq_len, embedding_size)
tensor_dataset_test_2 = create_tensor_dataset_from_dataframe(df_2_test, embeddings, IDs)
# from the training set compute mean and std to normalize both the tests
mean_pre_training_MLM = tensor_dataset_train_2.mean(dim=1).mean(dim=0)
std_pre_training_MLM = tensor_dataset_train_2.std(dim=1).std(dim=0)
# normalize the datasets with z-score
tensor_dataset_train_2 = (tensor_dataset_train_2 - mean_pre_training_MLM) / std_pre_training_MLM
tensor_dataset_test_2 = (tensor_dataset_test_2 - mean_pre_training_MLM) / std_pre_training_MLM
# split the training into train and valid set
tensor_dataset_train_2, tensor_dataset_valid_2 = train_test_split(
    tensor_dataset_train_2, test_size=0.2,
    random_state=42,
    shuffle=True)

print("dataset for BC created")
# create the dataloaders
print("creating dataloaders...")
train_dataloader_pre_training_MLM = DataLoader(TensorDataset(tensor_dataset_train_2),
                                              batch_size=32,
                                              shuffle=True, num_workers=0)
valid_dataloader_pre_training_MLM = DataLoader(TensorDataset(tensor_dataset_valid_2),
                                              batch_size=32,
                                              shuffle=True, num_workers=0)
test_dataloader_pre_training_MLM = DataLoader(TensorDataset(tensor_dataset_test_2),
                                             batch_size=32, shuffle=True,
                                             num_workers=0)
print("dataloaders for pre-training task #2 created!")

# create the dataset for the fill in the blank task
# load the unified dataset
print('Loading the unified dataset...')
df_3 = pd.read_csv('./reduced_data/unified_dataset_MLM.csv')

# shuffle of df_3 to make it different from df_2
df_3 = df_3.sample(frac=1, random_state=42).reset_index(drop=True)

print('Unified dataset loaded!')


df_3_train, df_3_test = train_test_split(df_3, test_size=0.2,
                                         random_state=42,
                                         shuffle=True)
# reset the index of the dataframes
df_3_train = df_3_train.reset_index(drop=True)
df_3_test = df_3_test.reset_index(drop=True)
# from the dataframes create the tensor datasets
tensor_dataset_train_3 = create_tensor_dataset_from_dataframe(df_3_train, embeddings,
                                                            IDs)  # shape (n_outfits, seq_len, embedding_size)
tensor_dataset_test_3 = create_tensor_dataset_from_dataframe(df_3_test, embeddings, IDs)
# from the training set compute mean and std to normalize both the tests
mean_fine_tuning = tensor_dataset_train_3.mean(dim=1).mean(dim=0)
std_fine_tuning = tensor_dataset_train_3.std(dim=1).std(dim=0)
# normalize the datasets with z-score
tensor_dataset_train_3 = (tensor_dataset_train_3 - mean_fine_tuning) / std_fine_tuning
tensor_dataset_test_3 = (tensor_dataset_test_3 - mean_fine_tuning) / std_fine_tuning
# split the training into train and valid set
tensor_dataset_train_3, tensor_dataset_valid_3 = train_test_split(
    tensor_dataset_train_3, test_size=0.2,
    random_state=42,
    shuffle=True)

print("dataset for BC created")
# create the dataloaders
print("creating dataloaders...")
train_dataloader_fine_tuning = DataLoader(TensorDataset(tensor_dataset_train_3),
                                              batch_size=32,
                                              shuffle=True, num_workers=0)
valid_dataloader_fine_tuning = DataLoader(TensorDataset(tensor_dataset_valid_3),
                                              batch_size=32,
                                              shuffle=True, num_workers=0)
test_dataloader_fine_tuning = DataLoader(TensorDataset(tensor_dataset_test_3),
                                             batch_size=32, shuffle=True,
                                             num_workers=0)
print("dataloaders for pre-training task #2 created!")


# define the space in which to search for the hyperparameters
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
    'lr': hp.uniform('lr', 1e-5, 1e-1),
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
    model = umBERT(embeddings=embeddings, num_encoders=params['num_encoders'],
                   num_heads=params['num_heads'], dropout=params['dropout'])
    # define the optimizer
    optimizer = params['optimizer'](params=model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    # pre train on task #1

    # pre train on task #2

    # fine tune on task #3

    # return the validation accuracy on fill in the blank task in the fine-tuning phase
    return {'loss': loss, 'params': params, 'status': STATUS_OK}


# optimize
best = fmin(fn=objective, space=space, algo=tpe_algorithm, max_evals=max_evals, trials=baeyes_trials)

# pre-training task #2: Reconstruction of the inputs with random masking (using unified dataset MLM)
# fine-tuning task: Reconstruction of the inputs with one mask per time
