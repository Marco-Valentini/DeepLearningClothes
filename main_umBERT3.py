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
from torch.nn import CrossEntropyLoss, CosineEmbeddingLoss, CosineSimilarity
from datetime import datetime

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
    This function performs the pre-training of the umBERT model on the Binary Classification task.
    :param model: the umBERT model
    :param dataloaders: the dataloaders used to load the data (train and validation)
    :param optimizer: the optimizer used to update the parameters of the model
    :param criterion: the loss function used to compute the loss
    :param n_epochs: the number of epochs
    :param run: the run of the experiment (used to save the model and the plots of the loss and accuracy on neptune.ai)
    :return: the model and the minimum validation loss
    """
    train_loss = []  # keep track of the loss of the training phase
    val_loss = []  # keep track of the loss of the validation phase
    train_acc_CLF = []  # keep track of the accuracy of the training phase on the BC task
    val_acc_CLF = []  # keep track of the accuracy of the validation phase on the MLM task

    valid_loss_min = np.Inf  # track change in validation loss
    early_stopping = 0  # counter to keep track of the number of epochs without improvements in the validation loss

    for epoch in range(n_epochs):
        for phase in ['train', 'val']:
            print(f'Pre-train BC epoch: {epoch + 1}/{n_epochs} | Phase: {phase}')
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
                labels_CLF = labels.type(torch.LongTensor).to(device)  # move the labels_CLF to the device
                # do a one-hot encoding of the labels of the classification task and move them to the device
                labels_CLF_one_hot = torch.nn.functional.one_hot(labels_CLF, num_classes=2)

                optimizer.zero_grad()  # zero the gradients

                # set the gradient computation only if in training phase
                with torch.set_grad_enabled(phase == 'train'):
                    # compute the predictions of the model
                    logits_CLF = model.forward_BC(inputs)

                    # compute the total loss (sum of the average values of the two losses)
                    loss = criterion(logits_CLF, labels_CLF_one_hot.float())

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
                    now = datetime.now()
                    dt_string = now.strftime("%d_%m_%Y")
                    # TODO se va male prova salvataggio su cpu
                    torch.save(checkpoint, f'./models/umBERT2_pre_trained_BC_{model.d_model}_{dt_string}.pth')
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
    plt.title('Loss pre-training (Classification)')
    plt.show()
    plt.plot(train_acc_CLF, label='train')
    plt.plot(val_acc_CLF, label='val')
    plt.legend()
    plt.title('ccuracy (Classification) pre-training')
    plt.show()
    return model, valid_loss_min


def pre_train_MLM(model, dataloaders, optimizer, criterion, n_epochs, run):
    """
    This function performs the pre-training of the umBERT model on the MLM task.
    :param model: the umBERT model
    :param dataloaders: the dataloaders used to load the data (train and validation)
    :param optimizer: the optimizer used to update the parameters of the model
    :param criterion: the loss function used to compute the loss
    :param n_epochs: the number of epochs
    :param run: the run of the experiment (used to save the model and the plots of the loss and accuracy on neptune.ai)
    :return: the model and the minimum validation loss
    """
    # the model given from te main is already on the GPU
    train_loss = []  # keep track of the loss of the training phase
    val_loss = []  # keep track of the loss of the validation phase
    train_acc_decoding = []  # keep track of the accuracy of the training phase on the MLM classification task
    val_acc_decoding = []  # keep track of the accuracy of the validation phase on the MLM classification task

    valid_loss_min = np.Inf  # track change in validation loss
    early_stopping = 0  # counter to keep track of the number of epochs without improvements in the validation loss
    for epoch in range(n_epochs):
        for phase in ['train', 'val']:
            print(f'Pre-training MLM Epoch: {epoch + 1}/{n_epochs} | Phase: {phase}')
            if phase == 'train':
                model.train()  # set model to training mode
                print("Training...")
            else:
                model.eval()  # set model to evaluate mode
                print("Validation...")
            running_loss = 0.0  # keep track of the loss
            accuracy_shoes = 0.0  # keep track of the accuracy of shoes classification task
            accuracy_tops = 0.0  # keep track of the accuracy of tops classification task
            accuracy_acc = 0.0  # keep track of the accuracy of accessories classification task
            accuracy_bottoms = 0.0  # keep track of the accuracy of bottoms classification task
            for inputs in dataloaders[phase]:  # for each batch
                inputs = inputs.to(device)  # move the input tensors to the GPU

                optimizer.zero_grad()  # zero the parameter gradients
                with torch.set_grad_enabled(phase == 'train'):  # forward + backward + optimize only if in training phase
                    logits_shoes, logits_tops, logits_acc, logits_bottoms = model.forward_MLM(inputs)
                    # compute the loss
                    target = torch.ones(logits_shoes.shape[0]).to(device)  # target is a tensor of ones
                    loss_shoes = criterion(logits_shoes, inputs[:, 0, :], target)  # compute the loss for shoes
                    loss_tops = criterion(logits_tops, inputs[:, 1, :], target)  # compute the loss for tops
                    loss_acc = criterion(logits_acc, inputs[:, 2, :], target)  # compute the loss for accessories
                    loss_bottoms = criterion(logits_bottoms, inputs[:, 3, :], target)  # compute the loss for bottoms
                    loss = loss_shoes + loss_tops + loss_acc + loss_bottoms  # compute the total loss

                    if phase == 'train':
                        loss.backward()  # compute the gradients of the loss
                        optimizer.step()  # update the parameters

                # update the loss value (multiply by the batch size)
                running_loss += loss.item() * inputs.size(0)

                # compute the closest embeddings to the reconstructed embeddings
                pred_shoes = find_closest_embeddings(logits_shoes, model.catalogue)
                pred_tops = find_closest_embeddings(logits_tops, model.catalogue)
                pred_acc = find_closest_embeddings(logits_acc, model.catalogue)
                pred_bottoms = find_closest_embeddings(logits_bottoms, model.catalogue)

                # compute the accuracy of the MLM task
                accuracy_shoes = 0.0
                accuracy_tops = 0.0
                accuracy_acc = 0.0
                accuracy_bottoms = 0.0
                for i in range(inputs.shape[0]):
                    if torch.equal(pred_shoes[i, :], inputs[i, 0, :]):
                        accuracy_shoes += 1
                    if torch.equal(pred_tops[i, :], inputs[i, 1, :]):
                        accuracy_tops += 1
                    if torch.equal(pred_acc[i, :], inputs[i, 2, :]):
                        accuracy_acc += 1
                    if torch.equal(pred_bottoms[i, :], inputs[i, 3, :]):
                        accuracy_bottoms += 1

            # compute the average loss of the epoch
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            # compute the average accuracy of the MLM task of the epoch
            epoch_accuracy_shoes = accuracy_shoes / len(dataloaders[phase].dataset)
            epoch_accuracy_tops = accuracy_tops / len(dataloaders[phase].dataset)
            epoch_accuracy_acc = accuracy_acc / len(dataloaders[phase].dataset)
            epoch_accuracy_bottoms = accuracy_bottoms / len(dataloaders[phase].dataset)
            epoch_accuracy_MLM = (epoch_accuracy_shoes + epoch_accuracy_tops + epoch_accuracy_acc + epoch_accuracy_bottoms) / 4

            if run is not None:
                run[f"{phase}/epoch/loss"].append(epoch_loss)
                run[f'{phase}/epoch/acc_shoes'].append(epoch_accuracy_shoes)
                run[f'{phase}/epoch/acc_tops'].append(epoch_accuracy_tops)
                run[f'{phase}/epoch/acc_acc'].append(epoch_accuracy_acc)
                run[f'{phase}/epoch/acc_bottoms'].append(epoch_accuracy_bottoms)
                run[f"{phase}/epoch/acc_MLM"].append(epoch_accuracy_MLM)
            print(f'{phase} Loss: {epoch_loss}')
            print(f'{phase} Accuracy (shoes): {epoch_accuracy_shoes}')
            print(f'{phase} Accuracy (tops): {epoch_accuracy_tops}')
            print(f'{phase} Accuracy (acc): {epoch_accuracy_acc}')
            print(f'{phase} Accuracy (bottoms): {epoch_accuracy_bottoms}')
            print(f'{phase} Accuracy (MLM): {epoch_accuracy_MLM}')

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc_decoding.append(epoch_accuracy_MLM)
            else:
                val_loss.append(epoch_loss)
                val_acc_decoding.append(epoch_accuracy_MLM)

                # save model if validation loss has decreased
                if epoch_loss <= valid_loss_min:
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                        valid_loss_min,
                        epoch_loss))
                    print('Validation accuracy MLM of the saved model: {:.6f}'.format(epoch_accuracy_MLM))
                    # save a checkpoint dictionary containing the model state_dict
                    checkpoint = {'d_model': model.d_model,
                                  'num_encoders': model.num_encoders,
                                  'num_heads': model.num_heads,
                                  'dropout': model.dropout,
                                  'dim_feedforward': model.dim_feedforward,
                                  'model_state_dict': model.state_dict()}
                    # save the checkpoint dictionary to a file
                    now = datetime.now()
                    dt_string = now.strftime("%d_%m_%Y")
                    # TODO se va male prova salvataggio su cpu
                    torch.save(checkpoint, f'./models/umBERT2_pre_trained_MLM_{model.d_model}_{dt_string}.pth')
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
    plt.title('Loss pre-training (MLM task)')
    plt.show()
    plt.plot(train_acc_decoding, label='train')
    plt.plot(val_acc_decoding, label='val')
    plt.legend()
    plt.title('Accuracy (MLM) pre-training')
    plt.show()
    return model, valid_loss_min


def fine_tune(model, dataloaders, optimizer, criterion, n_epochs, run):
    """
    This function performs the pre-training of the umBERT model on the fill in the blank task.
    :param model: the umBERT model
    :param dataloaders: the dataloaders used to load the data (train and validation)
    :param optimizer: the optimizer used to update the parameters of the model
    :param criterion: the loss function used to compute the loss
    :param n_epochs: the number of epochs
    :param run: the run of the experiment (used to save the model and the plots of the loss and accuracy on neptune.ai)
    :return: the model and the minimum validation loss
    """
    # the model given from te main is already on the GPU
    train_loss = []  # keep track of the loss of the training phase
    val_loss = []  # keep track of the loss of the validation phase
    train_acc = []  # keep track of the accuracy of the training phase on the fill in the blank task
    val_acc = []  # keep track of the accuracy of the validation phase on the fill in the blank task

    valid_loss_min = np.Inf  # track change in validation loss
    early_stopping = 0  # counter to keep track of the number of epochs without improvements in the validation loss

    for epoch in range(n_epochs):
        for phase in ['train', 'val']:
            print(f'Fine-tune epoch: {epoch + 1}/{n_epochs} | Phase: {phase}')
            if phase == 'train':
                model.train()  # set model to training mode
                print("Training...")
            else:
                model.eval()  # set model to evaluate mode
                print("Validation...")

            running_loss = 0.0  # keep track of the loss
            accuracy = 0.0  # keep track of the accuracy of the fill in the blank task

            for inputs in dataloaders[phase]:  # for each batch
                inputs = inputs.to(device)  # move the input tensors to the GPU

                optimizer.zero_grad()  # zero the parameter gradients

                with torch.set_grad_enabled(phase == 'train'):  # forward + backward + optimize only if in training phase
                    # compute the forward pass
                    outputs, masked_positions, masked_items = model.forward_fill_in_the_blank(inputs)

                    # compute the loss
                    loss = torch.zeros(1)
                    for i in range(outputs.shape[0]):  # for each outfit in the batch
                        j = masked_positions[i]  # the position of the masked item in the outfit
                        # compute the loss
                        loss += criterion(outputs[i, j, :], masked_items[i, :])  # compute the loss for the masked item

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # update the loss value (multiply by the batch size)
                running_loss += loss.item() * inputs.size(0)

                # from the outputs of the model, retrieve only the predictions of the masked items
                # (the ones that are in the positions of the masked items)
                masked_reconstructions = outputs[torch.arange(outputs.shape[0]), masked_positions, :]

                # compute the closest embeddings to the reconstructed embeddings
                predictions = find_closest_embeddings(masked_reconstructions, model.catalogue)

                # compute the accuracy of the fill in the blank task
                accuracy = 0
                for i in range(inputs.shape[0]):
                    if torch.equal(predictions[i, :], masked_items[i, :]):
                        accuracy += 1

            # compute the average loss of the epoch
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            # compute the average accuracy of the fill in the blank task of the epoch
            epoch_accuracy = accuracy / len(dataloaders[phase].dataset)

            if run is not None:
                run[f"{phase}/epoch/loss"].append(epoch_loss)
                run[f"{phase}/epoch/acc"].append(epoch_accuracy)

            print(f'{phase} Loss: {epoch_loss}')
            print(f'{phase} Accuracy (fill in the blank): {epoch_accuracy}')

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_accuracy)
            else:
                val_loss.append(epoch_loss)
                val_acc.append(epoch_accuracy)

                # save model if validation loss has decreased
                if epoch_loss <= valid_loss_min:
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                        valid_loss_min,
                        epoch_loss))
                    print('Validation accuracy fill in the blank of the saved model: {:.6f}'.format(epoch_accuracy))
                    # save a checkpoint dictionary containing the model state_dict
                    checkpoint = {'d_model': model.d_model,
                                  'num_encoders': model.num_encoders,
                                  'num_heads': model.num_heads,
                                  'dropout': model.dropout,
                                  'dim_feedforward': model.dim_feedforward,
                                  'model_state_dict': model.state_dict()}
                    # save the checkpoint dictionary to a file
                    now = datetime.now()
                    dt_string = now.strftime("%d_%m_%Y")
                    # TODO se va male prova salvataggio su cpu
                    torch.save(checkpoint, f'./models/umBERT2_fine_tuned_{model.d_model}_{dt_string}.pth')
                    valid_loss_min = epoch_loss
                    early_stopping = 0  # reset early stopping counter
                else:
                    early_stopping += 1
        if early_stopping == 10:
            print('Early stopping the training')
            break
    plt.plot(train_loss, label='train')
    plt.plot(val_loss, label='val')
    plt.legend()
    plt.title('Loss pre-training (fill in the blank task)')
    plt.show()
    plt.plot(train_acc, label='train')
    plt.plot(val_acc, label='val')
    plt.legend()
    plt.title('Accuracy (fill in the blank) pre-training')
    plt.show()
    return model, valid_loss_min


def find_closest_embeddings(recons_embeddings, embeddings):
    """
    Find the closest embeddings in the catalogue to the reconstructed embeddings
    :param recons_embeddings: the reconstructed embeddings (tensor) (shape: (batch_size, embedding_size))
    :return: the closest embeddings (tensor)
    """
    cosine_similarity = CosineSimilarity(dim=1, eps=1e-6)  # define the cosine similarity function
    embeddings = torch.from_numpy(embeddings).to(device)  # convert to tensor
    closest_embeddings = []
    for i in range(recons_embeddings.shape[0]):  # for each reconstructed embedding in the batch
        # compute the cosine similarity between the reconstructed embedding and the embeddings of the catalogue
        distances = cosine_similarity(recons_embeddings[i, :], embeddings)
        # find the index of the closest embedding
        idx = torch.max(distances, dim=0).indices
        # retrieve the closest embedding
        closest_embedding = embeddings[idx, :]
        # append the closest embedding to the list
        closest_embeddings.append(closest_embedding)
    return torch.stack(closest_embeddings).to(device)

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
#device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
device = torch.device('cpu')
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
tensor_dataset_train = create_tensor_dataset_from_dataframe(df_train, embeddings, IDs)  # shape (n_outfits, seq_len, embedding_size)
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
tensor_dataset_train_2 = create_tensor_dataset_from_dataframe(df_2_train, embeddings, IDs)  # shape (n_outfits, seq_len, embedding_size)
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

print("dataset for MLM created")
# create the dataloaders
print("creating dataloaders...")
train_dataloader_pre_training_MLM = DataLoader(tensor_dataset_train_2, batch_size=64, shuffle=True, num_workers=0)
valid_dataloader_pre_training_MLM = DataLoader(tensor_dataset_valid_2, batch_size=64, shuffle=True, num_workers=0)
test_dataloader_pre_training_MLM = DataLoader(tensor_dataset_test_2, batch_size=64, shuffle=True, num_workers=0)

dataloaders_MLM = {'train': train_dataloader_pre_training_MLM, 'val': valid_dataloader_pre_training_MLM, 'test': test_dataloader_pre_training_MLM}
print("dataloaders for pre-training task #2 created!")

# create the dataset for the fill in the blank task
# load the unified dataset
print('Loading the unified dataset...')
df_3 = pd.read_csv('./reduced_data/unified_dataset_MLM.csv')

# shuffle of df_3 to make it different from df_2
df_3 = df_3.sample(frac=1, random_state=42).reset_index(drop=True)

print('Unified dataset loaded!')


df_3_train, df_3_test = train_test_split(df_3, test_size=0.2, random_state=42, shuffle=True)
# reset the index of the dataframes
df_3_train = df_3_train.reset_index(drop=True)
df_3_test = df_3_test.reset_index(drop=True)
# from the dataframes create the tensor datasets
tensor_dataset_train_3 = create_tensor_dataset_from_dataframe(df_3_train, embeddings, IDs) # shape (n_outfits, seq_len, embedding_size)
tensor_dataset_test_3 = create_tensor_dataset_from_dataframe(df_3_test, embeddings, IDs)
# from the training set compute mean and std to normalize both the tests
mean_fine_tuning = tensor_dataset_train_3.mean(dim=1).mean(dim=0)
std_fine_tuning = tensor_dataset_train_3.std(dim=1).std(dim=0)
# normalize the datasets with z-score
tensor_dataset_train_3 = (tensor_dataset_train_3 - mean_fine_tuning) / std_fine_tuning
tensor_dataset_test_3 = (tensor_dataset_test_3 - mean_fine_tuning) / std_fine_tuning
# split the training into train and valid set
tensor_dataset_train_3, tensor_dataset_valid_3 = train_test_split(tensor_dataset_train_3, test_size=0.2,
                                                                  random_state=42, shuffle=True)

print("dataset for BC created")
# create the dataloaders
print("creating dataloaders...")
train_dataloader_fine_tuning = DataLoader(tensor_dataset_train_3, batch_size=64, shuffle=True, num_workers=0)
valid_dataloader_fine_tuning = DataLoader(tensor_dataset_valid_3, batch_size=64, shuffle=True, num_workers=0)
test_dataloader_fine_tuning = DataLoader(tensor_dataset_test_3, batch_size=64, shuffle=True, num_workers=0)
dataloaders_fine_tuning = {'train': train_dataloader_fine_tuning, 'val': valid_dataloader_fine_tuning,
                           'test': test_dataloader_fine_tuning}
print("dataloaders for pre-training task #2 created!")

# define the space in which to search for the hyperparameters
### hyperparameters tuning ###
print('Starting hyperparameters tuning...')
# define the maximum number of evaluations
max_evals = 1
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
    print(f"model loaded on {device}")
    # pre-train on task #1
    # define the optimizer
    optimizer1 = params['optimizer1'](params=model.parameters(), lr=params['lr1'], weight_decay=params['weight_decay'])
    criterion1 = CrossEntropyLoss()
    model, best_loss_BC = pre_train_BC(model=model, dataloaders=dataloaders_BC, optimizer=optimizer1,
                                       criterion=criterion1, n_epochs=params['n_epochs_1'], run=None)

    # pre-train on task #2
    # define the optimizer
    optimizer2 = params['optimizer2'](params=model.parameters(), lr=params['lr2'], weight_decay=params['weight_decay'])
    criterion2 = CosineEmbeddingLoss()
    model, best_loss_MLM = pre_train_MLM(model=model, dataloaders=dataloaders_MLM, optimizer=optimizer2,
                                         criterion=criterion2, n_epochs=params['n_epochs_2'], run=None)

    # fine-tune on task #3
    # define the optimizer
    optimizer3 = params['optimizer3'](params=model.parameters(), lr=params['lr3'], weight_decay=params['weight_decay'])
    criterion3 = CosineEmbeddingLoss()
    model, best_loss_fine_tune = fine_tune(model=model, dataloaders=dataloaders_fine_tuning, optimizer=optimizer3,
                                           criterion=criterion3, n_epochs=params['n_epochs_3'], run=None)

    loss = 0.2*best_loss_BC + 0.3*best_loss_MLM + 0.5*best_loss_fine_tune
    # return the validation accuracy on fill in the blank task in the fine-tuning phase
    return {'loss': loss, 'params': params, 'status': STATUS_OK}


# optimize
best = fmin(fn=objective, space=space, algo=tpe_algorithm, max_evals=max_evals,
            trials=baeyes_trials, rstate=np.random.default_rng(SEED))

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

# pre-train on task #1
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

# pre-train on task #2
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
model, best_loss_MLM = pre_train_MLM(model=model, dataloaders=dataloaders_MLM, optimizer=optimizer2,
                                     criterion=criterion2, n_epochs=params['n_epochs_2'], run=run)

# fine-tune on task #3
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
model, best_loss_fine_tune = fine_tune(model=model, dataloaders=dataloaders_fine_tuning, optimizer=optimizer3,
                                       criterion=criterion3, n_epochs=params['n_epochs_3'], run=run)
