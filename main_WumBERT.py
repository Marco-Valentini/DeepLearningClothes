import random
import os
import json
import pandas as pd
import numpy as np
import torch
from BERT_architecture.WumBERT import WumBERT
from hyperopt import Trials, hp, fmin, tpe, STATUS_OK
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from lion_pytorch import Lion
from torch.optim import Adam
from torch.nn import MSELoss

# set the seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
SEED = 42

# set the working directory to the path of the file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# use GPU if available
device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
# device = torch.device('cpu')
print('Device used: ', device)

# load the dataset
print('Loading the compatibility dataset...')
df = pd.read_csv('./reduced_data/reduced_compatibility.csv')

# remove all the non-compatible outfits
df = df[df['compatibility'] == 1].drop(columns=['compatibility'])
df.reset_index(drop=True, inplace=True)
with open("./nuovi_embeddings/2023_07_27AE_IDs_list", "r") as fp:
    IDs = json.load(fp)
# load the embeddings
with open(f'./nuovi_embeddings/2023_07_27AE_embeddings_128.npy', 'rb') as f:
    embeddings = np.load(f)
print('Dataset loaded')

# create the mappings and the item sets
total_mapping = {i: id for i, id in enumerate(IDs)}
total_mapping_reverse = {v: k for k,v in total_mapping.items()}

shoes_idx = {total_mapping_reverse[i] for i in df['item_1'].unique()}
tops_idx = {total_mapping_reverse[i] for i in df['item_2'].unique()}
accessories_idx = {total_mapping_reverse[i] for i in df['item_3'].unique()}
bottoms_idx = {total_mapping_reverse[i] for i in df['item_4'].unique()}

# split the outfits in train and test
df = df.applymap(lambda x: total_mapping_reverse[x]).values
df_train, df_test = train_test_split(df, test_size=0.2,
                                     random_state=42)
df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=42)

# create the dataloader
train_loader = DataLoader(df_train, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(df_val, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(df_test, batch_size=32, shuffle=True, num_workers=0)

dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

# define the space in which to search for the hyperparameters
### hyperparameters tuning ###
print('Starting hyperparameters tuning...')
# define the maximum number of evaluations
max_evals = 25
# define the search space
possible_learning_rates_pre_training = [1e-5,1e-4,1e-3]
possible_learning_rates_fine_tuning = [1e-5, 1e-4, 1e-3]
possible_n_heads = [1, 2, 4, 8]
possible_n_encoders = [3, 6,9, 12]

possible_optimizers = [Adam, Lion]


space = {
    'lr1': hp.choice('lr1', possible_learning_rates_pre_training),
    'lr2': hp.choice('lr2', possible_learning_rates_fine_tuning),
    'dropout': hp.uniform('dropout', 0, 0.2),
    'num_encoders': hp.choice('num_encoders', possible_n_encoders),
    'num_heads': hp.choice('num_heads', possible_n_heads),
    'weight_decay': hp.uniform('weight_decay', 0, 0.01),
    'optimizer1': hp.choice('optimizer1', possible_optimizers),
    'optimizer2': hp.choice('optimizer2', possible_optimizers)
}

# define the algorithm
tpe_algorithm = tpe.suggest

# define the trials object
baeyes_trials = Trials()

def objective(params):
    print("Starting new trial ")
    print(f"Trainig with params: {params}")
    # define the model
    model = WumBERT(embeddings=embeddings,num_encoders=params['num_encoders'], num_heads=params['num_heads'], dropout=params['dropout'],shoes_idx=shoes_idx, tops_idx=tops_idx, accessories_idx=accessories_idx, bottoms_idx=bottoms_idx)

    model.to(device)  # move the model to the device
    print(f"model loaded on {device}")
    # pre-train on task #1 reconstruction
    # pre-train on task #2
    # define the optimizer
    print("Starting pre-training the model on task reconstruction...")
    n_epochs = 1
    optimizer1 = params['optimizer1'](params=model.parameters(), lr=params['lr1'], weight_decay=params['weight_decay'])
    criterion1 = MSELoss()
    model, best_loss_reconstruction = model.fit_reconstruction(dataloaders=dataloaders, device=device, epochs=n_epochs, criterion=criterion1, optimizer=optimizer1)

    # fine-tune on task #3
    # define the optimizer
    print("Starting fine tuning the model...")
    optimizer2 = params['optimizer2'](params=model.parameters(), lr=params['lr2'], weight_decay=params['weight_decay'])
    criterion2 = MSELoss()
    model, best_loss_fine_tune = model.fit_fill_in_the_blank(dataloaders=dataloaders, device=device, epochs=n_epochs, criterion=criterion2, optimizer=optimizer2)
    # compute the weighted sum of the losses
    loss = best_loss_reconstruction + best_loss_fine_tune
    # return the validation accuracy on fill in the blank task in the fine-tuning phase
    return {'loss': loss, 'params': params, 'status': STATUS_OK}

# optimize
best = fmin(fn=objective, space=space, algo=tpe_algorithm, max_evals=max_evals,
            trials=baeyes_trials, rstate=np.random.default_rng(SEED), verbose=True, show_progressbar=False)

# train the model using the optimal hyperparameters found
params = {
    'lr1': possible_learning_rates_pre_training[best['lr1']],
    'lr2': possible_learning_rates_pre_training[best['lr2']],
    'dropout': best['dropout'],
    'num_encoders': possible_n_encoders[best['num_encoders']],
    'num_heads': possible_n_heads[best['num_heads']],
    'weight_decay': best['weight_decay'],
    'optimizer1': possible_optimizers[best['optimizer1']],
    'optimizer2': possible_optimizers[best['optimizer2']],
}
print(f"Best hyperparameters found: {params}")

# define the model
model = WumBERT(embeddings=embeddings, num_encoders=params['num_encoders'], num_heads=params['num_heads'], dropout=params['dropout'],shoes_idx=shoes_idx, tops_idx=tops_idx, accessories_idx=accessories_idx, bottoms_idx=bottoms_idx)
model.to(device)  # move the model to the device

# pre-train on task #1 reconstruction
# define the optimizer
n_epochs = 500
optimizer1 = params['optimizer1'](params=model.parameters(), lr=params['lr1'], weight_decay=params['weight_decay'])
criterion1 = MSELoss()
model, best_loss_reconstruction = model.fit_reconstruction(dataloaders=dataloaders, device=device, epochs=n_epochs, criterion=criterion1, optimizer=optimizer1)
# fine-tune on task fill in the  blank
# define the optimizer
optimizer2 = params['optimizer2'](params=model.parameters(), lr=params['lr2'], weight_decay=params['weight_decay'])
criterion2 = MSELoss()
model, best_loss_fine_tune = model.fit_fill_in_the_blank(dataloaders=dataloaders, device=device, epochs=n_epochs, criterion=criterion2, optimizer=optimizer2)

print(f"Best loss reconstruction: {best_loss_reconstruction}")
print(f"Best loss fine-tune: {best_loss_fine_tune}")
print("THE END")

# test on the test set

# for input in dataloaders['test']:
#     input = input.to(device)
#     output = model.forward_with_masking(input)



