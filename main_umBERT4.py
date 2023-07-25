import random
import torch
import os
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from BERT_architecture.umBERT3 import umBERT3 as umBERT
from hyperopt import Trials, hp, fmin, tpe, STATUS_OK
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
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
    return torch.Tensor(dataset)


def find_closest_embeddings(recons_embeddings, embeddings, IDs_list):
    """
    Find the closest embeddings in the catalogue to the reconstructed embeddings
    :param recons_embeddings: the reconstructed embeddings (tensor) (shape: (batch_size, embedding_size))
    :return: the closest embeddings (tensor)
    """
    # TODO valutare se invece si usa una euclidean distance cosa succede
    embeddings = torch.from_numpy(embeddings).to(device)  # convert to tensor
    # with open('./reduced_data/IDs_list') as f:
    #     IDs_list = json.load(f)
    closest_embeddings = []
    cosine_similarity = CosineSimilarity(dim=1, eps=1e-6)
    for i in range(recons_embeddings.shape[0]):  # for each reconstructed embedding in the batch
        # compute the cosine similarity between the reconstructed embedding and the embeddings of the catalogue
        similarities = cosine_similarity(recons_embeddings[i, :], embeddings)
        # find the index of the closest embedding
        idx = torch.max(similarities, dim=0).indices
        # retrieve the closest embedding
        closest_embedding = embeddings[idx, :]
        # append the closest embedding to the list
        closest_embeddings.append(IDs_list[idx])
    return torch.LongTensor(closest_embeddings).to(device)


def find_top_k_closest_embeddings(recons_embeddings, embeddings_dict, masked_positions, topk=10):
    embeddings_shoes = embeddings_dict['shoes']
    embeddings_tops = embeddings_dict['tops']
    embeddings_accessories = embeddings_dict['accessories']
    embeddings_bottoms = embeddings_dict['bottoms']
    closest_embeddings = []
    cosine_similarity = CosineSimilarity(dim=1)
    for i, pos in enumerate(masked_positions):
        if pos == 0:  # shoes
            # compute the cosine similarity between the reconstructed embedding and the embeddings of the catalogue
            similarities = cosine_similarity(recons_embeddings[i, :], torch.Tensor(embeddings_shoes).to(device))
            idx = torch.topk(similarities, k=topk).indices
            idx = idx.tolist()
            closest = [shoes_IDs[j] for j in idx]
        elif pos == 1:  # tops
            # compute the cosine similarity between the reconstructed embedding and the embeddings of the catalogue
            similarities = cosine_similarity(recons_embeddings[i, :], torch.Tensor(embeddings_tops).to(device))
            idx = torch.topk(similarities, k=topk).indices
            idx = idx.tolist()
            closest = [tops_IDs[j] for j in idx]
        elif pos == 2:  # accessories
            # compute the cosine similarity between the reconstructed embedding and the embeddings of the catalogue
            similarities = cosine_similarity(recons_embeddings[i, :], torch.Tensor(embeddings_accessories).to(device))
            idx = torch.topk(similarities, k=topk).indices
            idx = idx.tolist()
            closest = [accessories_IDs[j] for j in idx]
        elif pos == 3:  # bottoms
            # compute the cosine similarity between the reconstructed embedding and the embeddings of the catalogue
            similarities = cosine_similarity(recons_embeddings[i, :], torch.Tensor(embeddings_bottoms).to(device))
            idx = torch.topk(similarities, k=topk).indices
            idx = idx.tolist()
            closest = [bottoms_IDs[j] for j in idx]
        # append the closest embedding to the list
        closest_embeddings.append(closest)
    return closest_embeddings


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
    best_model = model
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
            # batch_number = 0
            for inputs, labels in dataloaders[phase]:  # for each batch
                # print(f'Batch: {batch_number}/{len(dataloaders[phase])}')
                # batch_number += 1
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
                # phase is validation
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
                    dt_string = now.strftime("%Y_%m_%d")
                    torch.save(checkpoint, f'./models/{dt_string}_umBERT4_pre_trained_BC_{model.d_model}.pth')
                    valid_loss_min = epoch_loss  # update the minimum validation loss
                    best_model = model
                    early_stopping = 0  # reset early stopping counter
                elif epoch > 50:
                    early_stopping += 1  # increment early stopping counter
        if early_stopping == 10 and epoch > 50:
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
    plt.title('Accuracy (Classification) pre-training')
    plt.show()
    return best_model, valid_loss_min


def pre_train_reconstruction(model, dataloaders, optimizer, criterion, n_epochs, run, shoes_IDs, tops_IDs,
                             accessories_IDs, bottoms_IDs):
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
    best_model = model
    for epoch in range(n_epochs):
        for phase in ['train', 'val']:
            print(f'Pre-training Reconstruction Epoch: {epoch + 1}/{n_epochs} | Phase: {phase}')
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
            # batch_number = 0
            for inputs, labels in dataloaders[phase]:  # for each batch
                # print(f'Batch: {batch_number}/{len(dataloaders[phase])}')
                # batch_number += 1
                inputs = inputs.to(device)  # move the input tensors to the GPU
                # labels are the IDs of the items in the outfit
                labels_shoes = labels[:, 0].to(device)  # move the labels_shoes to the device
                labels_tops = labels[:, 1].to(device)  # move the labels_tops to the device
                labels_acc = labels[:, 2].to(device)  # move the labels_acc to the device
                labels_bottoms = labels[:, 3].to(device)  # move the labels_bottoms to the device

                optimizer.zero_grad()  # zero the parameter gradients
                with torch.set_grad_enabled(
                        phase == 'train'):  # forward + backward + optimize only if in training phase
                    logits_shoes, logits_tops, logits_acc, logits_bottoms = model.forward_reconstruction(inputs)
                    # compute the loss
                    target = torch.ones(logits_shoes.shape[0]).to(device)  # target is a tensor of ones
                    # TODO problema che dia valori maggiori di 1?
                    loss_shoes = criterion(logits_shoes, inputs[:, 0, :], target)  # compute the loss for shoes
                    loss_tops = criterion(logits_tops, inputs[:, 1, :], target)  # compute the loss for tops
                    loss_acc = criterion(logits_acc, inputs[:, 2, :], target)  # compute the loss for accessories
                    loss_bottoms = criterion(logits_bottoms, inputs[:, 3, :], target)  # compute the loss for bottoms
                    # TODO chiedi se dividere o no per 4
                    loss = (
                                       loss_shoes + loss_tops + loss_acc + loss_bottoms) / 4  # compute the total loss and normalize it

                    if phase == 'train':
                        # loss_shoes.backward()  # compute the gradients of the loss
                        # loss_tops.backward()  # compute the gradients of the loss
                        # loss_acc.backward()  # compute the gradients of the loss
                        # loss_bottoms.backward()  # compute the gradients of the loss
                        loss.backward()  # compute the gradients of the loss
                        optimizer.step()  # update the parameters

                # update the loss value (multiply by the batch size)
                running_loss += loss.item() * inputs.size(0)

                # compute the closest embeddings to the reconstructed embeddings
                pred_shoes = find_closest_embeddings(logits_shoes, model.catalogue_dict['shoes'], shoes_IDs)
                pred_tops = find_closest_embeddings(logits_tops, model.catalogue_dict['tops'], tops_IDs)
                pred_acc = find_closest_embeddings(logits_acc, model.catalogue_dict['accessories'], accessories_IDs)
                pred_bottoms = find_closest_embeddings(logits_bottoms, model.catalogue_dict['bottoms'], bottoms_IDs)

                # update the accuracy of the reconstruction task
                accuracy_shoes += np.sum(pred_shoes.int().cpu().numpy() == labels_shoes.cpu().numpy())
                accuracy_tops += np.sum(pred_tops.int().cpu().numpy() == labels_tops.cpu().numpy())
                accuracy_acc += np.sum(pred_acc.int().cpu().numpy() == labels_acc.cpu().numpy())
                accuracy_bottoms += np.sum(pred_bottoms.int().cpu().numpy() == labels_bottoms.cpu().numpy())

            # compute the average loss of the epoch
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            # compute the average accuracy of the MLM task of the epoch
            epoch_accuracy_shoes = accuracy_shoes / len(dataloaders[phase].dataset)
            epoch_accuracy_tops = accuracy_tops / len(dataloaders[phase].dataset)
            epoch_accuracy_acc = accuracy_acc / len(dataloaders[phase].dataset)
            epoch_accuracy_bottoms = accuracy_bottoms / len(dataloaders[phase].dataset)
            epoch_accuracy_reconstruction = (
                                                    epoch_accuracy_shoes + epoch_accuracy_tops + epoch_accuracy_acc + epoch_accuracy_bottoms) / 4

            if run is not None:
                run[f"{phase}/epoch/loss"].append(epoch_loss)
                run[f'{phase}/epoch/acc_shoes'].append(epoch_accuracy_shoes)
                run[f'{phase}/epoch/acc_tops'].append(epoch_accuracy_tops)
                run[f'{phase}/epoch/acc_acc'].append(epoch_accuracy_acc)
                run[f'{phase}/epoch/acc_bottoms'].append(epoch_accuracy_bottoms)
                run[f"{phase}/epoch/acc_MLM"].append(epoch_accuracy_reconstruction)
            print(f'{phase} Loss: {epoch_loss}')
            print(f'{phase} Accuracy (shoes): {epoch_accuracy_shoes}')
            print(f'{phase} Accuracy (tops): {epoch_accuracy_tops}')
            print(f'{phase} Accuracy (acc): {epoch_accuracy_acc}')
            print(f'{phase} Accuracy (bottoms): {epoch_accuracy_bottoms}')
            print(f'{phase} Accuracy (Reconstruction): {epoch_accuracy_reconstruction}')

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc_decoding.append(epoch_accuracy_reconstruction)
            else:
                val_loss.append(epoch_loss)
                val_acc_decoding.append(epoch_accuracy_reconstruction)

                # save model if validation loss has decreased
                if epoch_loss <= valid_loss_min:
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                        valid_loss_min,
                        epoch_loss))
                    print('Validation accuracy in reconstruction of the saved model: {:.6f}'.format(
                        epoch_accuracy_reconstruction))
                    # save a checkpoint dictionary containing the model state_dict
                    checkpoint = {'d_model': model.d_model,
                                  'num_encoders': model.num_encoders,
                                  'num_heads': model.num_heads,
                                  'dropout': model.dropout,
                                  'dim_feedforward': model.dim_feedforward,
                                  'model_state_dict': model.state_dict()}
                    # save the checkpoint dictionary to a file
                    now = datetime.now()
                    dt_string = now.strftime("%Y_%m_%d")
                    torch.save(checkpoint,
                               f'./models/{dt_string}_umBERT4_pre_trained_reconstruction_{model.d_model}.pth')
                    valid_loss_min = epoch_loss  # update the minimum validation loss
                    early_stopping = 0  # reset early stopping counter
                    best_model = model
                else:
                    early_stopping += 1  # increment early stopping counter
        if early_stopping == 10:
            print('Early stopping the training')
            break
    plt.plot(train_loss, label='train')
    plt.plot(val_loss, label='val')
    plt.legend()
    plt.title('Loss pre-training (reconstruction task)')
    plt.show()
    plt.plot(train_acc_decoding, label='train')
    plt.plot(val_acc_decoding, label='val')
    plt.legend()
    plt.title('Accuracy (reconstruction) pre-training')
    plt.show()
    return best_model, valid_loss_min


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
    train_hit_ratio = []  # keep track of the accuracy of the training phase on the fill in the blank task
    val_hit_ratio = []  # keep track of the accuracy of the validation phase on the fill in the blank task

    valid_loss_min = np.Inf  # track change in validation loss
    early_stopping = 0  # counter to keep track of the number of epochs without improvements in the validation loss
    best_model = model

    for epoch in range(n_epochs):
        for phase in ['train', 'val']:
            print(f'Fine-tuning epoch: {epoch + 1}/{n_epochs} | Phase: {phase}')
            if phase == 'train':
                model.train()  # set model to training mode
                print("Training...")
            else:
                model.eval()  # set model to evaluate mode
                print("Validation...")

            running_loss = 0.0  # keep track of the loss
            hit_ratio = 0.0  # keep track of the accuracy of the fill in the blank task
            # batch_number = 0
            for inputs, labels in dataloaders[phase]:  # for each batch
                # print(f'Batch: {batch_number}/{len(dataloaders[phase])}')
                # batch_number += 1
                inputs = inputs.to(device)  # move the input tensors to the GPU
                # labels are the IDs of the items in the outfit
                labels_shoes = labels[:, 0].to(device)  # move the labels_shoes to the device
                labels_tops = labels[:, 1].to(device)  # move the labels_tops to the device
                labels_acc = labels[:, 2].to(device)  # move the labels_acc to the device
                labels_bottoms = labels[:, 3].to(device)  # move the labels_bottoms to the device

                optimizer.zero_grad()  # zero the parameter gradients

                with torch.set_grad_enabled(
                        phase == 'train'):  # forward + backward + optimize only if in training phase
                    # compute the forward pass
                    masked_logits, masked_items, masked_positions = model.forward_fill_in_the_blank(inputs)

                    # compute the loss
                    target = torch.ones(masked_logits.shape[0]).to(device)
                    # compute the loss for each masked item
                    loss = criterion(masked_logits, masked_items, target)  # compute the loss for the masked item
                    # normalize the loss
                    # TODO controlla potrebbe essere troppo piccola come loss
                    # loss = loss / masked_logits.shape[0]

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # update the loss value (multiply by the batch size)
                running_loss += loss.item() * inputs.size(0)

                # from the outputs of the model, retrieve only the predictions of the masked items
                # (the ones that are in the positions of the masked items)

                # compute the closest embeddings to the reconstructed embeddings
                # predictions = find_closest_embeddings(masked_logits, model.catalogue)

                #  implement top-k accuracy
                top_k_predictions = find_top_k_closest_embeddings(masked_logits, model.catalogue_dict, masked_positions,
                                                                  topk=10)

                masked_IDs = []

                for i in range(len(labels_shoes)):
                    masked_IDs.append(labels_shoes[i].item())
                    masked_IDs.append(labels_tops[i].item())
                    masked_IDs.append(labels_acc[i].item())
                    masked_IDs.append(labels_bottoms[i].item())
                # TODO capire come calcolare accuracy
                for i, id in enumerate(masked_IDs):
                    if id in top_k_predictions[i]:
                        hit_ratio += 1

                # # compute the accuracy of the fill in the blank task
                # accuracy += torch.sum(predictions == masked_IDs)

            # compute the average loss of the epoch
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            # compute the average accuracy of the fill in the blank task of the epoch
            epoch_hit_ratio = hit_ratio  # / len(dataloaders[phase].dataset) # TODO valuta per cosa dividi qui
            print(f"length of masked IDs {len(masked_IDs)}")
            print(f"Length of dataset {len(dataloaders[phase].dataset)}")

            if run is not None:
                run[f"{phase}/epoch/loss"].append(epoch_loss)
                run[f"{phase}/epoch/hit_ratio"].append(epoch_hit_ratio)

            print(f'{phase} Loss: {epoch_loss}')
            print(f'{phase} Hit ratio (fill in the blank): {epoch_hit_ratio}')

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_hit_ratio.append(epoch_hit_ratio)
            else:
                val_loss.append(epoch_loss)
                val_hit_ratio.append(epoch_hit_ratio)

                # save model if validation loss has decreased
                if epoch_loss <= valid_loss_min:
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                        valid_loss_min,
                        epoch_loss))
                    print('Validation hit ratio fill in the blank of the saved model: {:.6f}'.format(epoch_hit_ratio))
                    # save a checkpoint dictionary containing the model state_dict
                    checkpoint = {'d_model': model.d_model,
                                  'num_encoders': model.num_encoders,
                                  'num_heads': model.num_heads,
                                  'dropout': model.dropout,
                                  'dim_feedforward': model.dim_feedforward,
                                  'model_state_dict': model.state_dict()}
                    # save the checkpoint dictionary to a file
                    now = datetime.now()
                    dt_string = now.strftime("%Y_%m_%d")
                    # TODO se va male prova salvataggio su cpu
                    torch.save(checkpoint, f'./models/{dt_string}_umBERT4_fine_tuned_{model.d_model}.pth')
                    valid_loss_min = epoch_loss
                    early_stopping = 0  # reset early stopping counter
                    best_model = model
                else:
                    early_stopping += 1
        if early_stopping == 10:
            print('Early stopping the training')
            break
    plt.plot(train_loss, label='train')
    plt.plot(val_loss, label='val')
    plt.legend()
    plt.title('Loss fine-tuning (fill in the blank task)')
    plt.show()
    plt.plot(train_hit_ratio, label='train')
    plt.plot(val_hit_ratio, label='val')
    plt.legend()
    plt.title('Hit ratio (fill in the blank) fine-tuning')
    plt.show()
    return best_model, valid_loss_min


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
with open("./nuovi_embeddings/AESSIMVAL_with_linear_layers_new_IDs_list", "r") as fp:
    IDs = json.load(fp)
# load the embeddings
with open(f'./nuovi_embeddings/SCIUE_with_linear_layers_new_embeddings_1024.npy', 'rb') as f:
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

tensor_dataset_train_2, tensor_dataset_valid_2, df_train_only_compatible, df_valid_only_compatible = train_test_split(
    tensor_dataset_train_2, df_train_only_compatible, test_size=0.2, random_state=42, shuffle=True)

print("dataset for task #2 and #3 created")
# create the dataloaders
print("Creating dataloaders for the pre-training task #2...")
list_ID_train = [list(df_train_only_compatible['item_1'].values), list(df_train_only_compatible['item_2'].values),
                 list(df_train_only_compatible['item_3'].values), list(df_train_only_compatible['item_4'].values)]
list_ID_valid = [list(df_valid_only_compatible['item_1'].values), list(df_valid_only_compatible['item_2'].values),
                 list(df_valid_only_compatible['item_3'].values), list(df_valid_only_compatible['item_4'].values)]
list_ID_test = [list(df_test_only_compatible['item_1'].values), list(df_test_only_compatible['item_2'].values),
                list(df_test_only_compatible['item_3'].values), list(df_test_only_compatible['item_4'].values)]
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
possible_learning_rates_pre_training = [1e-3, 1e-2, 1e-1]
possible_learning_rates_fine_tuning = [1e-5, 1e-4, 1e-3, 1e-2]
possible_n_heads = [1, 2, 4,8]
possible_n_encoders = [1, 3, 6,9,12]
possible_n_epochs_pretrainig = [50, 100, 200]
possible_n_epochs_finetuning = [20, 50, 100]
possible_optimizers = [Adam, AdamW, Lion]

space = {
    'lr1': hp.choice('lr1', possible_learning_rates_pre_training),
    'lr2': hp.choice('lr2', possible_learning_rates_pre_training),
    'lr3': hp.choice('lr3', possible_learning_rates_fine_tuning),
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
    print(f"Trainig with params: {params}")
    # define the model
    model = umBERT(embeddings=embeddings, embeddings_dict=embeddings_dict, num_encoders=params['num_encoders'],
                   num_heads=params['num_heads'], dropout=params['dropout'])
    model.to(device)  # move the model to the device
    print(f"model loaded on {device}")
    # pre-train on task #1
    # define the optimizer
    print("Starting pre-training the model on task #1...")

    # checkpoint = torch.load('./models/2023_07_23_umBERT4_pre_trained_reconstruction_64.pth')
    # model.load_state_dict(checkpoint['model_state_dict'])
    print("model parameters loaded")

    # optimizer1 = params['optimizer1'](params=model.parameters(), lr=params['lr1'], weight_decay=params['weight_decay'])
    optimizer1 = torch.optim.SGD(params=model.parameters(), lr=params['lr1'], momentum=0.9, weight_decay=0.01)
    criterion1 = CrossEntropyLoss()
    model, best_loss_BC = pre_train_BC(model=model, dataloaders=dataloaders_BC, optimizer=optimizer1,
                                       criterion=criterion1, n_epochs=params['n_epochs_1'], run=None)

    # pre-train on task #2
    # define the optimizer
    print("Starting pre-training the model on task #2...")
    optimizer2 = params['optimizer2'](params=model.parameters(), lr=params['lr2'], weight_decay=params['weight_decay'])
    criterion2 = CosineEmbeddingLoss()
    model, best_loss_reconstruction = pre_train_reconstruction(model=model, dataloaders=dataloaders_reconstruction,
                                                               optimizer=optimizer2,
                                                               criterion=criterion2, n_epochs=params['n_epochs_2'],
                                                               shoes_IDs=shoes_IDs, tops_IDs=tops_IDs,
                                                               accessories_IDs=accessories_IDs, bottoms_IDs=bottoms_IDs,
                                                               run=None)

    # fine-tune on task #3
    # define the optimizer
    print("Starting fine tuning the model...")
    optimizer3 = params['optimizer3'](params=model.parameters(), lr=params['lr3'], weight_decay=params['weight_decay'])
    criterion3 = CosineEmbeddingLoss()
    model, best_loss_fine_tune = fine_tune(model=model, dataloaders=dataloaders_reconstruction, optimizer=optimizer3,
                                           criterion=criterion3, n_epochs=params['n_epochs_3'], run=None)
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
    'lr1': possible_learning_rates_pre_training[best['lr1']],
    'lr2': possible_learning_rates_pre_training[best['lr2']],
    'lr3': possible_learning_rates_fine_tuning[best['lr3']],
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
print(f"Best hyperparameters found: {params}")

# define the model
model = umBERT(embeddings=embeddings, embeddings_dict=embeddings_dict, num_encoders=params['num_encoders'],
               num_heads=params['num_heads'], dropout=params['dropout'])
model.to(device)  # move the model to the device
# pre-train on task #1
# define the run for monitoring the training on Neptune dashboard
# define the optimizer
optimizer1 = params['optimizer1'](params=model.parameters(), lr=params['lr1'], weight_decay=params['weight_decay'])
criterion1 = CrossEntropyLoss()
model, best_loss_BC = pre_train_BC(model=model, dataloaders=dataloaders_BC, optimizer=optimizer1,
                                   criterion=criterion1, n_epochs=params['n_epochs_1'], run=None)
# pre-train on task #2
# define the optimizer
optimizer2 = params['optimizer2'](params=model.parameters(), lr=params['lr2'], weight_decay=params['weight_decay'])
criterion2 = CosineEmbeddingLoss()
model, best_loss_reconstruction = pre_train_reconstruction(model=model, dataloaders=dataloaders_reconstruction,
                                                           optimizer=optimizer2,
                                                           criterion=criterion2, n_epochs=params['n_epochs_2'],
                                                           run=None)
# fine-tune on task #3
# define the optimizer
optimizer3 = params['optimizer3'](params=model.parameters(), lr=params['lr3'], weight_decay=params['weight_decay'])
criterion3 = CosineEmbeddingLoss()
model, best_loss_fine_tune = fine_tune(model=model, dataloaders=dataloaders_reconstruction, optimizer=optimizer3,
                                       criterion=criterion3, n_epochs=params['n_epochs_3'], run=None)

print(f"Best loss BC: {best_loss_BC}")
print(f"Best loss reconstruction: {best_loss_reconstruction}")
print(f"Best loss fine-tune: {best_loss_fine_tune}")
print("THE END")
