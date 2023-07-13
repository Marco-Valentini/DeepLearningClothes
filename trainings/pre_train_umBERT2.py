from hyperopt import Trials
from hyperopt import hp
from hyperopt import fmin
from hyperopt import tpe
from hyperopt import STATUS_OK
from torch.nn import CrossEntropyLoss
import random
from BERT_architecture.umBERT2 import umBERT2
import pandas as pd
import torch
from torch.utils.data import DataLoader
from lion_pytorch import Lion
from torch.optim import Adam, AdamW
import os

from scripts.bayesian_optimization import objective_umBERT2
from utility.create_tensor_dataset_for_BC_from_dataframe import create_tensor_dataset_for_BC_from_dataframe
import numpy as np
from utility.get_category_labels import get_category_labels
from utility.masking_input import masking_input
from utility.umBERT2_trainer import umBERT2_trainer
import json
from constants import get_special_embeddings

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

# load the catalogue for each category (shoes, tops, accessories, bottoms)
catalogues = {}
for category in ['shoes', 'tops', 'accessories', 'bottoms']:
    catalogues[category] = pd.read_csv(f'../reduced_data/reduced_catalogue_{category}.csv')
    print(f'Catalogue {category} loaded')

# first step: load the embeddings of the dataset obtained from fine-tuned model finetuned_fashion_resnet18
with open(f'../reduced_data/IDs_list', 'r') as fp:
    IDs = json.load(fp)
print(f'IDs {category} loaded')

with open(f'../reduced_data/embeddings_{dim_embeddings}.npy', 'rb') as f:
    embeddings = np.load(f)
print(f'Embeddings {category} loaded')

# create a dict of catalogue sizes for each category
catalogue_sizes = {}
for category in ['shoes', 'tops', 'accessories', 'bottoms']:
    catalogue_sizes[category] = catalogues[category]['ID'].size

# import the training set
train_dataframe = pd.read_csv('../reduced_data/reduced_compatibility_train2.csv')
compatibility_train = train_dataframe['compatibility'].values
train_dataframe.drop(columns=['compatibility'], inplace=True)

# create the labels for each item in the catalogue with resepct to their position in their catalogue
# (e.g. the first item of the catalogue has label 0, the second item has label 1, etc.)
category_labels_train = get_category_labels(train_dataframe, catalogues)

# create the tensor dataset for the training set (which contains the CLS embedding)
print('Creating the tensor dataset for the training set...')
training_set = create_tensor_dataset_for_BC_from_dataframe(train_dataframe, embeddings, IDs, CLS)
print('Scaling the training set using z-score...')
CLS_layer_train = training_set[0, :, :]  # CLS is the first element of the tensor, preserve it
training_set = training_set[1:, :, :]  # remove CLS from the tensor
mean = training_set.mean(dim=0).mean(dim=0)
std = training_set.std(dim=0).std(dim=0)
torch.save(mean, '../reduced_data/mean.pth')
torch.save(std, '../reduced_data/std.pth')
training_set = (training_set - mean) / std
training_set = torch.cat((CLS_layer_train.unsqueeze(0), training_set), dim=0)  # concatenate CLS to the scaled tensor

# mask the input (using the MASK embedding)
print('Masking the input...')
training_set, _, _ = masking_input(training_set, train_dataframe, MASK)
CLF_train_labels = torch.Tensor(compatibility_train).unsqueeze(1)
shoes_trainings_labels = torch.Tensor(category_labels_train['shoes']).unsqueeze(1)
tops_trainings_labels = torch.Tensor(category_labels_train['tops']).unsqueeze(1)
accessories_trainings_labels = torch.Tensor(category_labels_train['accessories']).unsqueeze(1)
bottoms_trainings_labels = torch.Tensor(category_labels_train['bottoms']).unsqueeze(1)

# concatenate the labels to the tensor
training_labels = torch.cat((CLF_train_labels,
                             shoes_trainings_labels,
                             tops_trainings_labels,
                             accessories_trainings_labels,
                             bottoms_trainings_labels), dim=1)
# create a Tensor Dataset
training_set = torch.utils.data.TensorDataset(training_set, training_labels)

# create the dataloader for the training set
print('Creating the dataloader for the training set...')
trainloader = DataLoader(training_set, batch_size=32, shuffle=True, num_workers=0)

# repeat the same steps for the validation set
validation_dataframe = pd.read_csv('../reduced_data/reduced_compatibility_valid.csv')
compatibility_validation = validation_dataframe['compatibility'].values
validation_dataframe.drop(columns=['compatibility'], inplace=True)
category_labels_val = get_category_labels(validation_dataframe, catalogues)
validation_set = create_tensor_dataset_for_BC_from_dataframe(validation_dataframe, embeddings, IDs, CLS)
print('Scaling the validation set using z-score...')
CLS_layer_validation = validation_set[0, :, :]
validation_set = validation_set[1:, :, :]
validation_set = (validation_set - mean) / std
validation_set = torch.cat((CLS_layer_validation.unsqueeze(0), validation_set), dim=0)
print('Masking the input...')
validation_set, _, _ = masking_input(validation_set, validation_dataframe, MASK)
CLF_validation_labels = torch.Tensor(compatibility_validation).unsqueeze(1)
shoes_validation_labels = torch.Tensor(category_labels_val['shoes']).unsqueeze(1)
tops_validation_labels = torch.Tensor(category_labels_val['tops']).unsqueeze(1)
accessories_validation_labels = torch.Tensor(category_labels_val['accessories']).unsqueeze(1)
bottoms_validation_labels = torch.Tensor(category_labels_val['bottoms']).unsqueeze(1)

validation_labels = torch.cat((CLF_validation_labels,
                               shoes_validation_labels,
                               tops_validation_labels,
                               accessories_validation_labels,
                               bottoms_validation_labels), dim=1)
validation_set = torch.utils.data.TensorDataset(validation_set, validation_labels)
validationloader = DataLoader(validation_set, batch_size=32, shuffle=True, num_workers=0)

# create the dictionary containing the dataloaders for the training and validation set
dataloaders = {'train': trainloader, 'val': validationloader}

### hyperparameters tuning ###
print('Starting hyperparameters tuning...')
# define the maximum number of evaluations
max_evals = 10
# define the search space
space = {
    'lr': hp.uniform('lr', 1e-5, 1e-2),
    'batch_size': hp.choice('batch_size', [16, 32, 64]),
    'n_epochs': hp.choice('n_epochs', [20, 50, 100]),
    'dropout': hp.uniform('dropout', 0, 0.5),
#    'd_model': hp.choice('d_model', [32, 64, 128, 256]),
    'num_encoders': hp.choice('num_encoders', [i for i in range(1, 12)]),
    'num_heads': hp.choice('num_heads', [1, 2, 4, 8]),
    'weight_decay': hp.uniform('weight_decay', 0, 0.1),
    'optimizer': hp.choice('optimizer', [Adam, AdamW, Lion])
}

# define the algorithm
tpe_algorithm = tpe.suggest

# define the trials object
baeyes_trials = Trials()

# define the objective function
def objective(params):
    # define the model
    model = umBERT2(catalogue_sizes=catalogue_sizes, d_model=dim_embeddings, num_encoders=params['num_encoders'],
                    num_heads=params['num_heads'], dropout=params['dropout'])
    # define the optimizer
    optimizer = params['optimizer'](params=model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    # define the criteria
    criterion = CrossEntropyLoss()
    # define the trainer
    trainer = umBERT2_trainer(model=model, optimizer=optimizer, criterion=criterion, device=device, n_epochs=params['n_epochs'])
    # train the model
    acc_CLF, acc_MLM = trainer.pre_train(dataloaders=dataloaders)

    loss = 2 - (acc_CLF + acc_MLM)  # minimize the loss

    # return the loss and the accuracy
    return {'loss': loss, 'params': params, 'status': STATUS_OK}

# optimize
best = fmin(fn=objective, space=space, algo=tpe_algorithm, max_evals=max_evals, trials=baeyes_trials)

# optimal model
print(best)

# # define the umBERT2 model
# model = umBERT2(catalogue_sizes=catalogue_sizes, d_model=dim_embeddings, num_encoders=6, num_heads=1, dropout=0.3)
#
# # use Adam as optimizer as suggested in the paper
# adam = Adam(params=model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.02, eps=1e-08)
# criterion = CrossEntropyLoss()
#
# print('Start pre-training the model')
# trainer = umBERT2_trainer(model=model, optimizer=adam, criterion=criterion, device=device, n_epochs=500)
# trainer.pre_train(dataloaders=dataloaders)
# print('Pre-training completed')

