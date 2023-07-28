import random
import os
import json
from BERT_architecture.umBERT3 import umBERT3 as umBERT
from hyperopt import Trials, hp, fmin, tpe, STATUS_OK
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam, AdamW
from torch.nn import MSELoss
from nuovi_embeddings.utilities_umBERT import *

# set the seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
SEED = 42

# set the working directory to the path of the file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# use GPU if available
# device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
device = torch.device('cpu')
print('Device used: ', device)

# pre-training task #1: Binary Classification (using compatibility dataset)
# load the compatibility dataset
print('Loading the compatibility dataset...')
df = pd.read_csv('./reduced_data/reduced_compatibility.csv')
print('Compatibility dataset loaded!')
# load the IDs of the images
with open("./nuovi_embeddings/2023_07_27AE_IDs_list", "r") as fp:
    IDs = json.load(fp)
# load the embeddings
with open(f'./nuovi_embeddings/2023_07_27AE_embeddings_128.npy', 'rb') as f:
    embeddings = np.load(f)

# compute the IDs of the shoes in the outfits
shoes_mapping = {i: id for i, id in enumerate(IDs) if id in df['item_1'].unique()}
shoes_positions = np.array(
    list(shoes_mapping.keys()))  # these are the positions with respect to the ID list and so in the embeddings matrix
shoes_IDs = np.array(list(shoes_mapping.values()))  # these are the IDs of the shoes in the outfits

embeddings_shoes = embeddings[shoes_positions]
# compute the IDs of the tops in the outfits
tops_mapping = {i: id for i, id in enumerate(IDs) if id in df['item_2'].unique()}
tops_positions = np.array(
    list(tops_mapping.keys()))  # these are the positions with respect to the ID list and so in the embeddings matrix
tops_IDs = np.array(list(tops_mapping.values()))

embeddings_tops = embeddings[np.array(tops_positions)]
# compute the IDs of the accessories in the outfits
accessories_mapping = {i: id for i, id in enumerate(IDs) if id in df['item_3'].unique()}
accessories_positions = np.array(list(
    accessories_mapping.keys()))  # these are the positions with respect to the ID list and so in the embeddings matrix
accessories_IDs = np.array(list(accessories_mapping.values()))

embeddings_accessories = embeddings[accessories_positions]

# compute the IDs of the bottoms in the outfits
bottoms_mapping = {i: id for i, id in enumerate(IDs) if id in df['item_4'].unique()}
bottoms_positions = np.array(
    list(bottoms_mapping.keys()))  # these are the positions with respect to the ID list and so in the embeddings matrix
bottoms_IDs = np.array(list(bottoms_mapping.values()))

embeddings_bottoms = embeddings[bottoms_positions]

embeddings_dict = {'shoes': embeddings_shoes, 'tops': embeddings_tops, 'accessories': embeddings_accessories,
                   'bottoms': embeddings_bottoms}
# split the dataset in train, valid and test set (80%, 10%, 10%) in a stratified way on the compatibility column
df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['compatibility'], random_state=42, shuffle=True)
df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)
# only the compatible outfits from the df_train
df_train_only_compatible = df_train[df_train['compatibility'] == 1].drop(columns='compatibility')
# only the compatible outfits from the df_test
df_test_only_compatible = df_test[df_test['compatibility'] == 1].drop(columns='compatibility')

print("create the dataloader for reconstruction task")
tensor_dataset_train_2 = create_tensor_dataset_from_dataframe(df_train_only_compatible, embeddings, IDs)
tensor_dataset_test_2 = create_tensor_dataset_from_dataframe(df_test_only_compatible, embeddings, IDs)
MASK_shoes = torch.randn((1, embeddings.shape[1])) + torch.cat((tensor_dataset_train_2, tensor_dataset_test_2), dim=0)[:, 0, :].mean(dim=0)
MASK_tops = torch.randn((1, embeddings.shape[1])) + torch.cat((tensor_dataset_train_2, tensor_dataset_test_2), dim=0)[:, 1, :].mean(dim=0)
MASK_acc = torch.randn((1, embeddings.shape[1])) + torch.cat((tensor_dataset_train_2, tensor_dataset_test_2), dim=0)[:, 2, :].mean(dim=0)
MASK_bottoms = torch.randn((1, embeddings.shape[1])) + torch.cat((tensor_dataset_train_2, tensor_dataset_test_2), dim=0)[:, 3, :].mean(dim=0)
MASK_dict = {'shoes': MASK_shoes, 'tops': MASK_tops, 'accessories': MASK_acc, 'bottoms': MASK_bottoms}

tensor_dataset_train_2, tensor_dataset_valid_2, df_train_only_compatible, df_valid_only_compatible = train_test_split(
    tensor_dataset_train_2, df_train_only_compatible, test_size=0.2, random_state=42, shuffle=True)

print("dataset for reconstruction task created")
# create the dataloaders
print("Creating dataloaders for reconstruction task...")
train_dataloader_pre_training_reconstruction = DataLoader(
    TensorDataset(tensor_dataset_train_2, torch.LongTensor(df_train_only_compatible.values)),
    batch_size=128, shuffle=True, num_workers=0)
valid_dataloader_pre_training_reconstruction = DataLoader(
    TensorDataset(tensor_dataset_valid_2, torch.LongTensor(df_valid_only_compatible.values)),
    batch_size=128, shuffle=True, num_workers=0)
test_dataloader_pre_training_reconstruction = DataLoader(
    TensorDataset(tensor_dataset_test_2, torch.LongTensor(df_test_only_compatible.values)),
    batch_size=128, shuffle=True, num_workers=0)
dataloaders_reconstruction = {'train': train_dataloader_pre_training_reconstruction,
                              'val': valid_dataloader_pre_training_reconstruction,
                              'test': test_dataloader_pre_training_reconstruction}
print("dataloaders for reconstruction task created!")

# define the space in which to search for the hyperparameters
### hyperparameters tuning ###
print('Starting hyperparameters tuning...')
# define the maximum number of evaluations
max_evals = 25
# define the search space
possible_learning_rates_pre_training = [1e-5, 1e-4, 1e-3]
possible_learning_rates_fine_tuning = [1e-5, 1e-4, 1e-3]
possible_n_heads = [1, 2, 4, 8]
possible_n_encoders = [3, 6, 9, 12]
possible_optimizers = [Adam, AdamW]

space = {
    'lr1': hp.choice('lr1', possible_learning_rates_pre_training),
    'lr2': hp.choice('lr2', possible_learning_rates_fine_tuning),
    'dropout': hp.uniform('dropout', 0, 0.2),
    'num_encoders': hp.choice('num_encoders', possible_n_encoders),
    'num_heads': hp.choice('num_heads', possible_n_heads),
    'weight_decay': hp.uniform('weight_decay', 0, 0.01),
    'optimizer': hp.choice('optimizer', possible_optimizers)
}

# define the algorithm
tpe_algorithm = tpe.suggest

# define the trials object
baeyes_trials = Trials()


# define the objective function
def objective(params):
    print(f"Trainig with params: {params}")
    n_epochs = 500
    # define the model
    model = umBERT(embeddings=embeddings, embeddings_dict=embeddings_dict, num_encoders=params['num_encoders'],
                   num_heads=params['num_heads'], dropout=params['dropout'], MASK_dict=MASK_dict)
    model.to(device)  # move the model to the device
    print(f"model loaded on {device}")
    # pre-train on task #1
    # define the optimizer
    print("Starting pre-training the model...")

    # pre-train (reconstruction task)
    # define the optimizer
    optimizer = params['optimizer'](params=model.parameters(), lr=params['lr1'], weight_decay=params['weight_decay'])
    criterion = MSELoss()
    model, best_loss_reconstruction = pre_train_reconstruction(model=model, dataloaders=dataloaders_reconstruction,
                                                               optimizer=optimizer,
                                                               criterion=criterion, n_epochs=n_epochs,
                                                               shoes_IDs=shoes_IDs, tops_IDs=tops_IDs,
                                                               accessories_IDs=accessories_IDs, bottoms_IDs=bottoms_IDs,
                                                               device=device)

    # fine-tune (fill in the blank task)
    # define the optimizer
    print("Starting fine tuning the model...")
    optimizer = params['optimizer'](params=model.parameters(), lr=params['lr2'], weight_decay=params['weight_decay'])
    criterion = MSELoss()
    model, best_loss_fine_tune = fine_tune(model=model, dataloaders=dataloaders_reconstruction, optimizer=optimizer,
                                           criterion=criterion, n_epochs=n_epochs, shoes_IDs=shoes_IDs,
                                           tops_IDs=tops_IDs, accessories_IDs=accessories_IDs, bottoms_IDs=bottoms_IDs,
                                           device=device)
    # compute the weighted sum of the losses
    loss = best_loss_reconstruction + best_loss_fine_tune
    # return the validation accuracy on fill in the blank task in the fine-tuning phase
    return {'loss': loss, 'params': params, 'status': STATUS_OK}


# optimize
best = fmin(fn=objective, space=space, algo=tpe_algorithm, max_evals=max_evals, trials=baeyes_trials,
            rstate=np.random.default_rng(SEED), verbose=True, show_progressbar=False)

# train the model using the optimal hyperparameters found
params = {
    'lr1': possible_learning_rates_pre_training[best['lr1']],
    'lr2': possible_learning_rates_fine_tuning[best['lr2']],
    'dropout': best['dropout'],
    'num_encoders': possible_n_encoders[best['num_encoders']],
    'num_heads': possible_n_heads[best['num_heads']],
    'weight_decay': best['weight_decay'],
    'optimizer': possible_optimizers[best['optimizer']],
}
n_epochs = 500
print(f"Best hyperparameters found: {params}")

# define the model
model = umBERT(embeddings=embeddings, embeddings_dict=embeddings_dict, num_encoders=params['num_encoders'],
               num_heads=params['num_heads'], dropout=params['dropout'], MASK_dict=MASK_dict)
model.to(device)  # move the model to the device
# pre-train on task #2
# define the optimizer
optimizer = params['optimizer'](params=model.parameters(), lr=params['lr1'], weight_decay=params['weight_decay'])
criterion = MSELoss()
model, best_loss_reconstruction = pre_train_reconstruction(model=model, dataloaders=dataloaders_reconstruction,
                                                           optimizer=optimizer, criterion=criterion, n_epochs=n_epochs,
                                                           shoes_IDs=shoes_IDs, tops_IDs=tops_IDs,
                                                           accessories_IDs=accessories_IDs, bottoms_IDs=bottoms_IDs,
                                                           device=device,)
# fine-tune on task #3
# define the optimizer
optimizer = params['optimizer'](params=model.parameters(), lr=params['lr2'], weight_decay=params['weight_decay'])
model, best_loss_fine_tune = fine_tune(model=model, dataloaders=dataloaders_reconstruction, optimizer=optimizer,
                                       criterion=criterion, n_epochs=n_epochs, shoes_IDs=shoes_IDs, tops_IDs=tops_IDs,
                                       accessories_IDs=accessories_IDs, bottoms_IDs=bottoms_IDs, device=device)

print(f"Best loss reconstruction: {best_loss_reconstruction}")
print(f"Best loss fine-tune: {best_loss_fine_tune}")


# test the model
test_model(model, device, dataloaders_reconstruction['test'], shoes_IDs, tops_IDs, accessories_IDs, bottoms_IDs, criterion)

print("THE END")