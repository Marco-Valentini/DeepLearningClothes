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
from matplotlib import pyplot as plt

from BERT_architecture.umBERT3 import umBERT3 as umBERT
from hyperopt import Trials, hp, fmin, tpe, STATUS_OK
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.optim import Adam, AdamW
from lion_pytorch import Lion
from torch.nn import CrossEntropyLoss, CosineEmbeddingLoss

from constants import API_TOKEN


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


def pre_train_BC(model, dataloaders, optimizer, criterion, n_epochs, run):
    """
            This function performs the pre-training of the umBERT model in a BERT-like fashion (MLM + classification tasks)
            :param dataloaders: the dataloaders used to load the data (train and validation)
            :param masked_positions: the positions of the masked elements in the outfit(train and validation)
            :param run: the run of the experiment (used to save the model and the plots of the loss and accuracy on neptune.ai)
            :return: None
            """
    train_loss = []  # keep track of the loss of the training phase
    val_loss = []  # keep track of the loss of the validation phase
    train_acc_CLF = []  # keep track of the accuracy of the training phase on the BC task
    val_acc_CLF = []  # keep track of the accuracy of the validation phase on the MLM task
    train_acc_decoding = []  # keep track of the accuracy of the training phase on the MLM classification task
    val_acc_decoding = []  # keep track of the accuracy of the validation phase on the MLM classification task

    valid_loss_min = np.Inf  # track change in validation loss
    early_stopping = 0  # counter to keep track of the number of epochs without improvements in the validation loss

    for epoch in range(n_epochs):
        for phase in ['train', 'val']:
            print(f'Epoch: {epoch + 1}/{n_epochs} | Phase: {phase}')
            if phase == 'train':
                model.train()  # set model to training mode
                print("Training...")
            else:
                model.eval()  # set model to evaluate mode
                print("Validation...")

            running_loss = 0.0  # keep track of the loss
            accuracy_CLF = 0.0  # keep track of the accuracy of the classification task

            for inputs, labels in dataloaders[phase]:  # for each batch
                inputs = inputs.to(device)  # move the data to the device
                labels_CLF = labels[:, 0].type(torch.LongTensor).to(device)  # move the labels_CLF to the device
                # do a one-hot encoding of the labels of the classification task and move them to the device
                labels_CLF_one_hot = torch.nn.functional.one_hot(labels_CLF, num_classes=2).to(device)

                optimizer.zero_grad()  # zero the gradients

                # set the gradient computation only if in training phase
                with torch.set_grad_enabled(phase == 'train'):
                    # compute the predictions of the model
                    logits_CLF = model.forward_BC(inputs)

                    # compute the total loss (sum of the average values of the two losses)
                    loss = criterion(logits_CLF, labels_CLF_one_hot)

                    if phase == 'train':
                        loss.backward()  # compute the gradients of the loss
                        optimizer.step()  # update the parameters

                # update the loss value (multiply by the batch size)
                running_loss += loss.item() * inputs.size(0)

                # update the accuracy of the classification task
                pred_labels_CLF = torch.max((model.softmax(logits_CLF, dim=1)), dim=1).indices

                # update the accuracy of the classification task
                accuracy_CLF += torch.sum(pred_labels_CLF == labels_CLF)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)  # compute the average loss of the epoch
            # compute the average accuracy of the classification task of the epoch
            epoch_accuracy_CLF = accuracy_CLF / len(dataloaders[phase].dataset)

            if run is not None:
                run[f"{phase}/epoch/loss"].append(epoch_loss)
                run[f"{phase}/epoch/acc_clf"].append(epoch_accuracy_CLF)

            print(f'{phase} Loss: {epoch_loss}')
            print(f'{phase} Accuracy (Classification): {epoch_accuracy_CLF}')
            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc_CLF.append(epoch_accuracy_CLF.item())
            else:
                val_loss.append(epoch_loss)
                val_acc_CLF.append(epoch_accuracy_CLF.item())

                # save model if validation loss has decreased
                if epoch_loss <= valid_loss_min:
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                        valid_loss_min,
                        epoch_loss))
                    print('Validation accuracy BC of the saved model: {:.6f}'.format(epoch_accuracy_CLF))
                    # save a checkpoint dictionary containing the model state_dict
                    checkpoint = {'d_model': model.d_model,
                                  'num_encoders': model.num_encoders,
                                  'num_heads': model.num_heads,
                                  'dropout': model.dropout,
                                  'dim_feedforward': model.dim_feedforward,
                                  'model_state_dict': model.state_dict()}
                    # save the checkpoint dictionary to a file
                    torch.save(checkpoint, f'../models/umBERT2_pre_trained_{model.d_model}.pth')
                    valid_loss_min = epoch_loss  # update the minimum validation loss
                    early_stopping = 0  # reset early stopping counter
                else:
                    early_stopping += 1  # increment early stopping counter
        if early_stopping == 10:
            print('Early stopping the training')
            break
    plt.plot(train_loss, label='train')
    plt.plot(val_loss, label='val')
    plt.legend()
    plt.title('Loss pre-training')
    plt.show()
    plt.plot(train_acc_CLF, label='train')
    plt.plot(val_acc_CLF, label='val')
    plt.legend()
    plt.title('Accuracy (Classification) pre-training')
    plt.show()
    plt.plot(train_acc_decoding, label='train')
    plt.plot(val_acc_decoding, label='val')
    plt.legend()
    plt.title('Accuracy (decoding) pre-training')
    plt.show()
    return model, valid_loss_min

def pre_train_MLM(model, dataloaders, optimizer, num_epochs):
    return model, best_loss

def fine_tune(model, dataloaders, optimizer, num_epochs):
    return model, best_loss


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
    stratify=compatibility_train, random_state=42, shuffle=True)

print("dataset for BC created")
# create the dataloaders
print("creating dataloaders for the pre-training...")
train_dataloader_pre_training_BC = DataLoader(TensorDataset(tensor_dataset_train, torch.Tensor(compatibility_train)),
                                              batch_size=64, shuffle=True, num_workers=0)
valid_dataloader_pre_training_BC = DataLoader(TensorDataset(tensor_dataset_valid, torch.Tensor(compatibility_valid)),
                                              batch_size=64, shuffle=True, num_workers=0)
test_dataloader_pre_training_BC = DataLoader(TensorDataset(tensor_dataset_test, torch.Tensor(compatibility_test)),
                                             batch_size=64, shuffle=True, num_workers=0)
dataloaders_BC = {'train': train_dataloader_pre_training_BC, 'val': valid_dataloader_pre_training_BC, 'test': test_dataloader_pre_training_BC}

print("dataloaders for pre-training task #1 created!")

# prepare the data for the pre-training task #2
# load the unified dataset
print('Loading the unified dataset...')
df_2 = pd.read_csv('./reduced_data/unified_dataset_MLM.csv')
print('Unified dataset loaded!')

df_2_train, df_2_test = train_test_split(df_2, test_size=0.2, random_state=42, shuffle=True)
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
tensor_dataset_train_2, tensor_dataset_valid_2 = train_test_split(tensor_dataset_train_2, test_size=0.2,
                                                                  random_state=42, shuffle=True)

print("dataset for BC created")
# create the dataloaders
print("creating dataloaders...")
train_dataloader_pre_training_MLM = DataLoader(TensorDataset(tensor_dataset_train_2),
                                               batch_size=64, shuffle=True, num_workers=0)
valid_dataloader_pre_training_MLM = DataLoader(TensorDataset(tensor_dataset_valid_2),
                                               batch_size=64, shuffle=True, num_workers=0)
test_dataloader_pre_training_MLM = DataLoader(TensorDataset(tensor_dataset_test_2),
                                              batch_size=64, shuffle=True, num_workers=0)

dataloaders_MLM = {'train': train_dataloader_pre_training_MLM, 'val': valid_dataloader_pre_training_MLM, 'test': test_dataloader_pre_training_MLM}
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
                                              batch_size=64,
                                              shuffle=True, num_workers=0)
valid_dataloader_fine_tuning = DataLoader(TensorDataset(tensor_dataset_valid_3),
                                              batch_size=64,
                                              shuffle=True, num_workers=0)
test_dataloader_fine_tuning = DataLoader(TensorDataset(tensor_dataset_test_3),
                                             batch_size=64, shuffle=True,
                                             num_workers=0)
dataloaders_fine_tuning = {'train': train_dataloader_fine_tuning, 'val': valid_dataloader_fine_tuning, 'test': test_dataloader_fine_tuning}
print("dataloaders for pre-training task #2 created!")


# define the space in which to search for the hyperparameters
### hyperparameters tuning ###
print('Starting hyperparameters tuning...')
# define the maximum number of evaluations
max_evals = 10
# define the search space
possible_learning_rates = [1e-5,1e-4,1e-3,1e-2]
possible_n_heads = [1, 2, 4, 8]
possible_n_encoders = [i for i in range(1, 12)]
possible_n_epochs_pretrainig = [50, 100, 200]
possible_n_epochs_finetuning = [10, 20, 50]
possible_optimizers = [Adam, AdamW, Lion]

space = {
    'lr1': hp.choice('lr1', possible_learning_rates),
    'lr2': hp.choice('lr2', possible_learning_rates),
    'lr3': hp.choice('lr3', possible_learning_rates),
    'n_epochs_1': hp.choice('n_epochs_1', possible_n_epochs_pretrainig),
    'n_epochs_2': hp.choice('n_epochs_2', possible_n_epochs_pretrainig),
    'n_epochs_3': hp.choice('n_epochs_3', possible_n_epochs_finetuning),
    'dropout': hp.uniform('dropout', 0, 0.5),
    'num_encoders': hp.choice('num_encoders', possible_n_encoders),
    'num_heads': hp.choice('num_heads', possible_n_heads),
    'weight_decay': hp.uniform('weight_decay', 0, 0.1),
    'optimizer1': hp.choice('optimizer1', possible_optimizers),
    'optimizer2': hp.choice('optimizer2', possible_optimizers),
    'optimizer3': hp.choice('optimizer3', possible_optimizers)
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
    model.to(device)  # move the model to the device
    # pre train on task #1
    # define the optimizer
    optimizer1 = params['optimizer1'](params=model.parameters(), lr=params['lr1'], weight_decay=params['weight_decay'])
    model, best_loss_BC = pre_train_BC(model, dataloaders_BC, optimizer1)

    # pre train on task #2
    # define the optimizer
    optimizer2 = params['optimizer2'](params=model.parameters(), lr=params['lr2'], weight_decay=params['weight_decay'])
    model, best_loss_MLM = pre_train_MLM(model, dataloaders_MLM, optimizer2)

    # fine tune on task #3
    # define the optimizer
    optimizer3 = params['optimizer3'](params=model.parameters(), lr=params['lr3'], weight_decay=params['weight_decay'])
    model, best_loss_fine_tune = fine_tune(model, dataloaders_fine_tuning, optimizer3)

    loss = 0.2*best_loss_BC + 0.3*best_loss_MLM + 0.5*best_loss_fine_tune
    # return the validation accuracy on fill in the blank task in the fine-tuning phase
    return {'loss': loss, 'params': params, 'status': STATUS_OK}


# optimize
best = fmin(fn=objective, space=space, algo=tpe_algorithm, max_evals=max_evals, trials=baeyes_trials)

# train the model using the optimal hyperparameters found
params = {
    'lr1': possible_learning_rates[best['lr1']],
    'lr2': possible_learning_rates[best['lr2']],
    'lr3': possible_learning_rates[best['lr3']],
    'n_epochs_1': possible_n_epochs_pretrainig[best['n_epochs_1']],
    'n_epochs_2': possible_n_epochs_pretrainig[best['n_epochs_2']],
    'n_epochs_3': possible_n_epochs_finetuning[best['n_epochs_3']],
    'dropout': best['dropout'],
    'num_encoders': possible_n_encoders[best['num_encoders']],
    'num_heads': possible_n_heads[best['num_heads']],
    'weight_decay': best['weight_decay'],
    'optimizer1': possible_optimizers[best['optimizer1']],
    'optimizer2': possible_optimizers[best['optimizer2']],
    'optimizer3': possible_optimizers[best['optimizer3']]
}

# define the model
model = umBERT(embeddings=embeddings, num_encoders=params['num_encoders'],
               num_heads=params['num_heads'], dropout=params['dropout'])
model.to(device)  # move the model to the device

# pre train on task #1
# define the run for monitoring the training on Neptune dashboard
run = neptune.init_run(
    project="marcopaolodeeplearning/DeepLearningOutfitCompetion",
    api_token=API_TOKEN,  # your credentials
    name="pre training umBERT2",
    tags=["umBERT2", "pre-training", "binary classification"],
)  # your credentials
run["parameters"] = {
    'lr1': params['lr1'],
    'n_epochs_1': params['n_epochs_1'],
    'dropout': params['dropout'],
    'num_encoders': params['num_encoders'],
    'num_heads': params['num_heads'],
    'weight_decay': params['weight_decay'],
    'optimizer1': params['optimizer1']
}
# define the optimizer
optimizer1 = params['optimizer1'](params=model.parameters(), lr=params['lr1'], weight_decay=params['weight_decay'])
criterion1 = CrossEntropyLoss()
model, best_loss_BC = pre_train_BC(model=model, dataloaders=dataloaders_BC, optimizer=optimizer1,
                                   criterion=criterion1, n_epochs=params['n_epochs_1'], run=run)

# pre train on task #2
# define the run for monitoring the training on Neptune dashboard
run = neptune.init_run(
    project="marcopaolodeeplearning/DeepLearningOutfitCompetion",
    api_token=API_TOKEN,  # your credentials
    name="pre training umBERT2",
    tags=["umBERT2", "pre-training", "MLM task"],
)  # your credentials
run["parameters"] = {
    'lr2': params['lr2'],
    'n_epochs_2': params['n_epochs_2'],
    'dropout': params['dropout'],
    'num_encoders': params['num_encoders'],
    'num_heads': params['num_heads'],
    'weight_decay': params['weight_decay'],
    'optimizer2': params['optimizer2']
}
# define the optimizer
optimizer2 = params['optimizer2'](params=model.parameters(), lr=params['lr2'], weight_decay=params['weight_decay'])
criterion2 = CosineEmbeddingLoss()
model, best_loss_MLM = pre_train_MLM(model, dataloaders_MLM, optimizer2)

# fine tune on task #3
run = neptune.init_run(
    project="marcopaolodeeplearning/DeepLearningOutfitCompetion",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjMTY5ZDBlZC1kY2QzLTQzNDYtYjc0OS02YzkzM2M3YjIyOTAifQ==", # your credentials
    name="fine-tuning umBERT2",
    tags=["umBERT2", "fine-tuning", "fill in the blank"],
)  # your credentials
run["parameters"] = {
    'lr3': params['lr3'],
    'n_epochs_3': params['n_epochs_3'],
    'dropout': params['dropout'],
    'num_encoders': params['num_encoders'],
    'num_heads': params['num_heads'],
    'weight_decay': params['weight_decay'],
    'optimizer3': params['optimizer3']
}
# define the optimizer
optimizer3 = params['optimizer3'](params=model.parameters(), lr=params['lr3'], weight_decay=params['weight_decay'])
criterion3 = CosineEmbeddingLoss()
model, best_loss_fine_tune = fine_tune(model, dataloaders_fine_tuning, optimizer3)

