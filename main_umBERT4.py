import random
import os
import json
from BERT_architecture.umBERT3 import umBERT3 as umBERT
from hyperopt import Trials, hp, fmin, tpe, STATUS_OK
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.nn import MSELoss
from constants import API_TOKEN
from nuovi_embeddings.utilities_umBERT import *

# set the seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
SEED = 42

# dim_embeddings = 64

# set the working directory to the path of the file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# use GPU if available
device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
# device = torch.device('cpu')
print('Device used: ', device)

# pre-training task #1: Binary Classification (using compatibility dataset)
# load the compatibility dataset
print('Loading the compatibility dataset...')
df = pd.read_csv('./reduced_data/reduced_compatibility.csv')
# balance the 2 classes of compatibility by removing some of the non-compatible outfits
# df = pd.concat([df[df['compatibility'] == 1], df[df['compatibility'] == 0].sample(n=df[df['compatibility'] == 1].shape[0], random_state=42)], axis=0)
df.reset_index(drop=True, inplace=True)
print('Compatibility dataset loaded!')
# load the IDs of the images
with open("./nuovi_embeddings/AE_IDs_list", "r") as fp:
    IDs = json.load(fp)
# load the embeddings
with open(f'./nuovi_embeddings/AE_embeddings_128.npy', 'rb') as f:
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
print("Creating the datasets for BC pre-training...")

df_train, df_test = train_test_split(df, test_size=0.2,
                                     stratify=df['compatibility'],
                                     random_state=42,
                                     shuffle=True)
df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)

df_train_only_compatible = df_train[df_train['compatibility'] == 1].drop(
    columns='compatibility')  # only the compatible outfits from the df_train
df_test_only_compatible = df_test[df_test['compatibility'] == 1].drop(
    columns='compatibility')  # only the compatible outfits from the df_test

compatibility_train = df_train['compatibility'].values  # compatibility labels for the tarin dataframe
compatibility_test = df_test['compatibility'].values  # compatiblility labels for the test dataframe
df_train.drop(columns='compatibility', inplace=True)
df_test.drop(columns='compatibility', inplace=True)

tensor_dataset_train = create_tensor_dataset_from_dataframe(df_train, embeddings, IDs)
tensor_dataset_test = create_tensor_dataset_from_dataframe(df_test, embeddings, IDs)
# compute the CLS as the average of the embeddings of the items in the outfit
# TODO controlla in debug se è giusto
compatible_mean = torch.cat(
    (tensor_dataset_train[compatibility_train == 1, :, :], tensor_dataset_test[compatibility_test == 1, :, :]),
    dim=0).mean(dim=1).mean(dim=0)
not_compatible_mean = torch.cat(
    (tensor_dataset_train[compatibility_train == 0, :, :], tensor_dataset_test[compatibility_test == 0, :, :]),
    dim=0).mean(dim=1).mean(dim=0)

# CLS = torch.mean(torch.stack((compatible_mean,not_compatible_mean)),dim=0).unsqueeze(0)
# MASK = compatible_mean.unsqueeze(0)

tensor_dataset_train, tensor_dataset_valid, compatibility_train, compatibility_valid = train_test_split(
    tensor_dataset_train, compatibility_train, test_size=0.2,
    stratify=compatibility_train, random_state=42, shuffle=True)

print("dataset for BC created")
# create the dataloaders
print("creating dataloaders for the pre-training...")
train_dataloader_pre_training_BC = DataLoader(TensorDataset(tensor_dataset_train, torch.Tensor(compatibility_train)),
                                              batch_size=16, shuffle=True, num_workers=0)
valid_dataloader_pre_training_BC = DataLoader(TensorDataset(tensor_dataset_valid, torch.Tensor(compatibility_valid)),
                                              batch_size=16, shuffle=True, num_workers=0)
test_dataloader_pre_training_BC = DataLoader(TensorDataset(tensor_dataset_test, torch.Tensor(compatibility_test)),
                                             batch_size=16, shuffle=True, num_workers=0)
dataloaders_BC = {'train': train_dataloader_pre_training_BC, 'val': valid_dataloader_pre_training_BC,
                  'test': test_dataloader_pre_training_BC}
print("dataloaders for pre-training task #1 created!")

print("create the dataloader for task #2")
tensor_dataset_train_2 = create_tensor_dataset_from_dataframe(df_train_only_compatible, embeddings, IDs)
tensor_dataset_test_2 = create_tensor_dataset_from_dataframe(df_test_only_compatible, embeddings, IDs)
MASK_shoes = torch.randn((1,embeddings.shape[1])) + torch.cat((tensor_dataset_train_2, tensor_dataset_test_2), dim=0)[:,0,:].mean(dim=0)
MASK_tops = torch.randn((1,embeddings.shape[1])) + torch.cat((tensor_dataset_train_2, tensor_dataset_test_2), dim=0)[:,1,:].mean(dim=0)
MASK_acc = torch.randn((1,embeddings.shape[1])) + torch.cat((tensor_dataset_train_2, tensor_dataset_test_2), dim=0)[:,2,:].mean(dim=0)
MASK_bottoms = torch.randn((1,embeddings.shape[1])) + torch.cat((tensor_dataset_train_2, tensor_dataset_test_2), dim=0)[:,3,:].mean(dim=0)
MASK_dict = {'shoes': MASK_shoes, 'tops': MASK_tops, 'accessories': MASK_acc, 'bottoms': MASK_bottoms}

tensor_dataset_train_2, tensor_dataset_valid_2, df_train_only_compatible, df_valid_only_compatible = train_test_split(
    tensor_dataset_train_2, df_train_only_compatible, test_size=0.2, random_state=42, shuffle=True)

print("dataset for task #2 and #3 created")
# create the dataloaders
print("Creating dataloaders for the pre-training task #2...")
train_dataloader_pre_training_reconstruction = DataLoader(
    TensorDataset(tensor_dataset_train_2, torch.LongTensor(df_train_only_compatible.values)),
    batch_size=16, shuffle=True, num_workers=0)
valid_dataloader_pre_training_reconstruction = DataLoader(
    TensorDataset(tensor_dataset_valid_2, torch.LongTensor(df_valid_only_compatible.values)),
    batch_size=16, shuffle=True, num_workers=0)
test_dataloader_pre_training_reconstruction = DataLoader(
    TensorDataset(tensor_dataset_test_2, torch.LongTensor(df_test_only_compatible.values)),
    batch_size=16, shuffle=True, num_workers=0)
dataloaders_reconstruction = {'train': train_dataloader_pre_training_reconstruction,
                              'val': valid_dataloader_pre_training_reconstruction,
                              'test': test_dataloader_pre_training_reconstruction}
print("dataloaders for pre-training task #2 created!")

# define the space in which to search for the hyperparameters
### hyperparameters tuning ###
print('Starting hyperparameters tuning...')
# define the maximum number of evaluations
max_evals = 10
# define the search space
possible_learning_rates_pre_training = [1e-5, 1e-4, 1e-3]
possible_learning_rates_fine_tuning = [1e-5, 1e-4, 1e-3]
possible_n_heads = [1, 2, 4, 8]
possible_n_encoders = [3, 6, 9, 12]
possible_n_epochs_pretrainig = [500]
possible_n_epochs_finetuning = [500]
possible_optimizers = [Adam]#, AdamW, Lion]

space = {
    #'lr1': hp.choice('lr1', possible_learning_rates_pre_training),
    'lr2': hp.choice('lr2', possible_learning_rates_pre_training),
    'lr3': hp.choice('lr3', possible_learning_rates_fine_tuning),
    #'n_epochs_1': hp.choice('n_epochs_1', possible_n_epochs_pretrainig),
    'n_epochs_2': hp.choice('n_epochs_2', possible_n_epochs_pretrainig),
    'n_epochs_3': hp.choice('n_epochs_3', possible_n_epochs_finetuning),
    'dropout': hp.uniform('dropout', 0, 0.2),
    'num_encoders': hp.choice('num_encoders', possible_n_encoders),
    'num_heads': hp.choice('num_heads', possible_n_heads),
    'weight_decay': hp.uniform('weight_decay', 0, 0.01),
    #'optimizer1': hp.choice('optimizer1', possible_optimizers),
    'optimizer2': hp.choice('optimizer2', possible_optimizers),
    'optimizer3': hp.choice('optimizer3', possible_optimizers)
}

# define the algorithm
tpe_algorithm = tpe.suggest

# define the trials object
baeyes_trials = Trials()


# define the objective function
def objective(params):
    print(f"Trainig with params: {params}")
    # define the model
    model = umBERT(embeddings=embeddings, embeddings_dict=embeddings_dict, num_encoders=params['num_encoders'],
                   num_heads=params['num_heads'], dropout=params['dropout'], MASK_dict=MASK_dict)
    model.to(device)  # move the model to the device
    print(f"model loaded on {device}")
    # pre-train on task #1
    # define the optimizer
    print("Starting pre-training the model on task #1...")

    # checkpoint = torch.load('./models/2023_07_23_umBERT4_pre_trained_reconstruction_64.pth')
    # model.load_state_dict(checkpoint['model_state_dict'])
    print("model parameters loaded")
    #
    # # optimizer1 = params['optimizer1'](params=model.parameters(), lr=params['lr1'], weight_decay=params['weight_decay'])
    # optimizer1 = torch.optim.SGD(params=model.parameters(), lr=params['lr1'], momentum=0.9, weight_decay=0.01)
    # criterion1 = CrossEntropyLoss()
    # model, best_loss_BC = pre_train_BC(model=model, dataloaders=dataloaders_BC, optimizer=optimizer1,
    #                                    criterion=criterion1, n_epochs=params['n_epochs_1'], device=device, run=None)

    # pre-train on task #2
    # define the optimizer
    print("Starting pre-training the model on task #2...")
    optimizer2 = params['optimizer2'](params=model.parameters(), lr=params['lr2'], weight_decay=params['weight_decay'])
    criterion2 = MSELoss()
    model, best_loss_reconstruction = pre_train_reconstruction(model=model, dataloaders=dataloaders_reconstruction,
                                                               optimizer=optimizer2,
                                                               criterion=criterion2, n_epochs=params['n_epochs_2'],
                                                               shoes_IDs=shoes_IDs, tops_IDs=tops_IDs,
                                                               accessories_IDs=accessories_IDs, bottoms_IDs=bottoms_IDs,
                                                               device=device, run=None)

    # fine-tune on task #3
    # define the optimizer
    print("Starting fine tuning the model...")
    optimizer3 = params['optimizer3'](params=model.parameters(), lr=params['lr3'], weight_decay=params['weight_decay'])
    criterion3 = MSELoss()
    model, best_loss_fine_tune = fine_tune(model=model, dataloaders=dataloaders_reconstruction, optimizer=optimizer3,
                                           criterion=criterion3, n_epochs=params['n_epochs_3'], shoes_IDs=shoes_IDs, tops_IDs=tops_IDs,
                                                               accessories_IDs=accessories_IDs, bottoms_IDs=bottoms_IDs,device=device, run=None)
    best_loss_BC = 0
    # compute the weighted sum of the losses
    loss = best_loss_BC + best_loss_reconstruction + best_loss_fine_tune
    # return the validation accuracy on fill in the blank task in the fine-tuning phase
    return {'loss': loss, 'params': params, 'status': STATUS_OK}


# optimize
best = fmin(fn=objective, space=space, algo=tpe_algorithm, max_evals=max_evals,
            trials=baeyes_trials, rstate=np.random.default_rng(SEED))

# train the model using the optimal hyperparameters found
params = {
#    'lr1': possible_learning_rates_pre_training[best['lr1']],
    'lr2': possible_learning_rates_pre_training[best['lr2']],
    'lr3': possible_learning_rates_fine_tuning[best['lr3']],
#    'n_epochs_1': possible_n_epochs_pretrainig[best['n_epochs_1']],
    'n_epochs_2': possible_n_epochs_pretrainig[best['n_epochs_2']],
    'n_epochs_3': possible_n_epochs_finetuning[best['n_epochs_3']],
    'dropout': best['dropout'],
    'num_encoders': possible_n_encoders[best['num_encoders']],
    'num_heads': possible_n_heads[best['num_heads']],
    'weight_decay': best['weight_decay'],
#    'optimizer1': possible_optimizers[best['optimizer1']],
    'optimizer2': possible_optimizers[best['optimizer2']],
    'optimizer3': possible_optimizers[best['optimizer3']]
}
print(f"Best hyperparameters found: {params}")

# define the model
model = umBERT(embeddings=embeddings, embeddings_dict=embeddings_dict, num_encoders=params['num_encoders'],
               num_heads=params['num_heads'], dropout=params['dropout'], MASK_dict=MASK_dict)
model.to(device)  # move the model to the device
# pre-train on task #1
# define the run for monitoring the training on Neptune dashboard
# define the optimizer
# optimizer1 = params['optimizer1'](params=model.parameters(), lr=params['lr1'], weight_decay=params['weight_decay'])
# criterion1 = CrossEntropyLoss()
# model, best_loss_BC = pre_train_BC(model=model, dataloaders=dataloaders_BC, optimizer=optimizer1,
#                                    criterion=criterion1, n_epochs=params['n_epochs_1'], device = device, run=None)
# pre-train on task #2
# define the optimizer
optimizer2 = params['optimizer2'](params=model.parameters(), lr=params['lr2'], weight_decay=params['weight_decay'])
criterion2 = MSELoss()
model, best_loss_reconstruction = pre_train_reconstruction(model=model, dataloaders=dataloaders_reconstruction,
                                                           optimizer=optimizer2,
                                                           criterion=criterion2, n_epochs=params['n_epochs_2'],shoes_IDs=shoes_IDs, tops_IDs=tops_IDs,
                                                               accessories_IDs=accessories_IDs, bottoms_IDs=bottoms_IDs, device=device,
                                                           run=None)
# fine-tune on task #3
# define the optimizer
optimizer3 = params['optimizer3'](params=model.parameters(), lr=params['lr3'], weight_decay=params['weight_decay'])
criterion3 = MSELoss()
model, best_loss_fine_tune = fine_tune(model=model, dataloaders=dataloaders_reconstruction, optimizer=optimizer3,
                                       criterion=criterion3, n_epochs=params['n_epochs_3'], shoes_IDs=shoes_IDs, tops_IDs=tops_IDs,
                                                               accessories_IDs=accessories_IDs, bottoms_IDs=bottoms_IDs, device=device, run=None)

#print(f"Best loss BC: {best_loss_BC}")
print(f"Best loss reconstruction: {best_loss_reconstruction}")
print(f"Best loss fine-tune: {best_loss_fine_tune}")
print("THE END")
