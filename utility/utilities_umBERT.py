import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
from copy import deepcopy
import tqdm


def create_tensor_dataset_from_dataframe(df_outfit: pd.DataFrame, embeddings, ids):
    """
    This function takes as input a dataframe containing the labels of the items in the outfit,
    the embeddings of the items and the ids of the items.
    It returns a tensor of shape (n_outfits, seq_len, embedding_size) containing the embeddings
    of the items in the outfit and the CLS token.
    :param df_outfit:  dataframe containing the labels of the items in the outfit
    :param embeddings:  embeddings of the items in the outfit (a tensor of shape (n_items, embedding_size))
    :param ids:  ids of the items in the outfit (a list of length n_items)
    :return: a tensor of shape (n_outfits,seq_len, embedding_size) containing the embeddings of the items in the outfit
    and the CLS token
    """
    dataset = np.zeros((df_outfit.shape[0], 4, embeddings.shape[1]))
    for i in range(df_outfit.shape[0]):  # for each outfit
        for j in range(df_outfit.shape[1]):  # for each item in the outfit
            ID = df_outfit.iloc[i, j]
            index_item = ids.index(ID)
            embedding = embeddings[index_item]
            dataset[i, j, :] = embedding
    return torch.Tensor(dataset)


def find_closest_embeddings(recons_embeddings, embeddings, IDs_list, device):
    """
    Find the closest embeddings in the catalogue to the reconstructed embeddings
    :param recons_embeddings: the reconstructed embeddings (tensor) (shape: (batch_size, embedding_size))
    :param embeddings: the embeddings of the catalogue (tensor) (shape: (n_items, embedding_size))
    :param IDs_list: the IDs of the items in the catalogue (list)
    :param device: the device used to train the model
    :return: the closest embeddings (tensor)
    """
    embeddings = torch.from_numpy(embeddings).to(device)  # convert to tensor
    closest_embeddings = []
    for i in range(recons_embeddings.shape[0]):  # for each reconstructed embedding in the batch
        # compute the euclidean distances between the reconstructed embedding and the embeddings of the catalogue
        distances = torch.cdist(recons_embeddings[i, :].unsqueeze(0), embeddings)
        # find the index of the closest embedding
        idx = torch.min(distances, dim=1).indices
        # append the closest embedding to the list
        closest_embeddings.append(IDs_list[idx])
    closest_embeddings = np.array(closest_embeddings)
    return torch.LongTensor(closest_embeddings).to(device)


def find_top_k_closest_embeddings(recons_embeddings, embeddings_dict, masked_positions, shoes_IDs, tops_IDs,
                                  accessories_IDs, bottoms_IDs, device, topk=10):
    """
    Find the top k closest embeddings in the catalogue to the reconstructed embeddings for each masked position in the outfit
    :param recons_embeddings: the reconstructed embeddings (tensor) (shape: (batch_size, embedding_size))
    :param embeddings_dict: dictionary containing the embeddings of the catalogue for each category
    :param masked_positions: list containing the position of the masked items in the outfit
    :param shoes_IDs: the IDs of the shoes in the catalogue
    :param tops_IDs: the IDs of the tops in the catalogue
    :param accessories_IDs: the IDs of the accessories in the catalogue
    :param bottoms_IDs: the IDs of the bottoms in the catalogue
    :param device: the device used to train the model
    :param topk: the number of closest embeddings to return
    :return: the top k closest embeddings (tensor)
    """
    embeddings_shoes = embeddings_dict['shoes']
    embeddings_tops = embeddings_dict['tops']
    embeddings_accessories = embeddings_dict['accessories']
    embeddings_bottoms = embeddings_dict['bottoms']
    closest_embeddings = []
    for i, pos in enumerate(masked_positions):
        if pos == 0:  # shoes
            distances = torch.cdist(recons_embeddings[i, :].unsqueeze(0), torch.Tensor(embeddings_shoes).to(device))
            idx = torch.topk(distances, k=topk, largest=False).indices
            idx = idx.tolist()
            closest_embeddings.append([shoes_IDs[j] for j in idx])
        elif pos == 1:  # tops
            distances = torch.cdist(recons_embeddings[i, :].unsqueeze(0), torch.Tensor(embeddings_tops).to(device))
            idx = torch.topk(distances, k=topk, largest=False).indices
            idx = idx.tolist()
            closest_embeddings.append([tops_IDs[j] for j in idx])
        elif pos == 2:  # accessories
            distances = torch.cdist(recons_embeddings[i, :].unsqueeze(0),
                                    torch.Tensor(embeddings_accessories).to(device))
            idx = torch.topk(distances, k=topk, largest=False).indices
            idx = idx.tolist()
            closest_embeddings.append([accessories_IDs[j] for j in idx])
        elif pos == 3:  # bottoms
            distances = torch.cdist(recons_embeddings[i, :].unsqueeze(0), torch.Tensor(embeddings_bottoms).to(device))
            idx = torch.topk(distances, k=topk, largest=False).indices
            idx = idx.tolist()
            closest_embeddings.append([bottoms_IDs[j] for j in idx])
    return closest_embeddings


def pre_train_reconstruction(model, dataloaders, optimizer, criterion, n_epochs, shoes_IDs, tops_IDs,
                             accessories_IDs, bottoms_IDs, device):
    """
    This function performs the pre-training of the umBERT model on the MLM task.
    :param model: the umBERT model
    :param dataloaders: the dataloaders used to load the data (train and validation)
    :param optimizer: the optimizer used to update the parameters of the model
    :param criterion: the loss function used to compute the loss
    :param n_epochs: the number of epochs
    :return: the model and the minimum validation loss
    """
    # the model given from te main is already on the device
    train_loss = []  # keep track of the loss of the training phase
    val_loss = []  # keep track of the loss of the validation phase
    train_acc_decoding = []  # keep track of the accuracy of the training phase
    val_acc_decoding = []  # keep track of the accuracy of the validation phase

    valid_acc_max = 0  # keep track of the maximum validation accuracy
    early_stopping = 0  # counter to keep track of the number of epochs without improvements in the validation loss
    best_model = deepcopy(model)
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
            for inputs, labels in tqdm.tqdm(dataloaders[phase], colour='blue'):  # for each batch
                inputs = inputs.to(device)  # move the input tensors to the device
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
                    loss_shoes = criterion(logits_shoes, inputs[:, 0, :])  # compute the loss for shoes
                    loss_tops = criterion(logits_tops, inputs[:, 1, :])  # compute the loss for tops
                    loss_acc = criterion(logits_acc, inputs[:, 2, :])  # compute the loss for accessories
                    loss_bottoms = criterion(logits_bottoms, inputs[:, 3, :])  # compute the loss for bottoms
                    loss = loss_shoes + loss_tops + loss_acc + loss_bottoms  # compute the total loss and normalize it

                    if phase == 'train':
                        loss.backward()  # compute the gradients of the loss
                        optimizer.step()  # update the parameters

                # update the loss value (multiply by the batch size)
                running_loss += loss.item() * inputs.size(0)

                # compute the closest embeddings to the reconstructed embeddings
                pred_shoes = find_closest_embeddings(logits_shoes, model.catalogue_dict['shoes'], shoes_IDs, device)
                pred_tops = find_closest_embeddings(logits_tops, model.catalogue_dict['tops'], tops_IDs, device)
                pred_acc = find_closest_embeddings(logits_acc, model.catalogue_dict['accessories'], accessories_IDs,
                                                   device)
                pred_bottoms = find_closest_embeddings(logits_bottoms, model.catalogue_dict['bottoms'], bottoms_IDs,
                                                       device)

                # update the accuracy of the reconstruction task
                accuracy_shoes += np.sum(pred_shoes.cpu().numpy() == labels_shoes.cpu().numpy())
                accuracy_tops += np.sum(pred_tops.cpu().numpy() == labels_tops.cpu().numpy())
                accuracy_acc += np.sum(pred_acc.cpu().numpy() == labels_acc.cpu().numpy())
                accuracy_bottoms += np.sum(pred_bottoms.cpu().numpy() == labels_bottoms.cpu().numpy())

            # compute the average loss of the epoch
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            # compute the average accuracy of the reconstruction of the epoch
            epoch_accuracy_shoes = accuracy_shoes / len(dataloaders[phase].dataset)
            epoch_accuracy_tops = accuracy_tops / len(dataloaders[phase].dataset)
            epoch_accuracy_acc = accuracy_acc / len(dataloaders[phase].dataset)
            epoch_accuracy_bottoms = accuracy_bottoms / len(dataloaders[phase].dataset)
            epoch_accuracy_reconstruction = (epoch_accuracy_shoes + epoch_accuracy_tops +
                                             epoch_accuracy_acc + epoch_accuracy_bottoms) / 4

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
                if epoch_accuracy_reconstruction >= valid_acc_max:
                    print('Validation accuracy increased ({:.6f} --> {:.6f}).'.format(valid_acc_max, epoch_accuracy_reconstruction))
                    print('Validation accuracy in reconstruction of the saved model: {:.6f}'.format(
                        epoch_accuracy_reconstruction))
                    valid_acc_max = epoch_accuracy_reconstruction
                    early_stopping = 0  # reset early stopping counter
                    best_model = deepcopy(model)
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
    return best_model, valid_acc_max


def fine_tune(model, dataloaders, optimizer, criterion, n_epochs, shoes_IDs, tops_IDs,
              accessories_IDs, bottoms_IDs, device):
    """
    This function performs the pre-training of the umBERT model on the fill in the blank task.
    :param model: the umBERT model
    :param dataloaders: the dataloaders used to load the data (train and validation)
    :param optimizer: the optimizer used to update the parameters of the model
    :param criterion: the loss function used to compute the loss
    :param n_epochs: the number of epochs
    :return: the model and the minimum validation loss
    """
    # the model given from te main is already moved to the device
    train_loss = []  # keep track of the loss of the training phase
    val_loss = []  # keep track of the loss of the validation phase
    train_hit_ratio = []  # keep track of the accuracy of the training phase on the fill in the blank task
    val_hit_ratio = []  # keep track of the accuracy of the validation phase on the fill in the blank task

    valid_hit_max = 0  # keep track of the maximum validation accuracy
    early_stopping = 0  # counter to keep track of the number of epochs without improvements in the validation loss
    best_model = deepcopy(model)

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
            accuracy_shoes = 0.0  # keep track of the accuracy of the fill in the blank task
            accuracy_tops = 0.0  # keep track of the accuracy of the fill in the blank task
            accuracy_acc = 0.0  # keep track of the accuracy of the fill in the blank task
            accuracy_bottoms = 0.0  # keep track of the accuracy of the fill in the blank task
            for inputs, labels in tqdm.tqdm(dataloaders[phase], colour='green'):  # for each batch
                inputs = inputs.to(device)  # move the inputs to the device
                # labels are the IDs of the items in the outfit
                labels_shoes = labels[:, 0].to(device)  # move the labels_shoes to the device
                labels_tops = labels[:, 1].to(device)  # move the labels_tops to the device
                labels_acc = labels[:, 2].to(device)  # move the labels_acc to the device
                labels_bottoms = labels[:, 3].to(device)  # move the labels_bottoms to the device
                # repeat each label 4 times
                labels_shoes = labels_shoes.repeat(4)
                labels_tops = labels_tops.repeat(4)
                labels_acc = labels_acc.repeat(4)
                labels_bottoms = labels_bottoms.repeat(4)

                optimizer.zero_grad()  # zero the parameter gradients

                with torch.set_grad_enabled(phase == 'train'):
                    # compute the forward pass
                    logits_shoes, logits_tops, logits_acc, logits_bottoms, masked_logits, \
                        masked_items, masked_positions = model.forward_fill_in_the_blank(inputs)

                    # compute the loss
                    # compute the loss for each masked item
                    loss_shoes = criterion(logits_shoes, inputs[:, 0].repeat(4, 1))
                    loss_tops = criterion(logits_tops, inputs[:, 1].repeat(4, 1))
                    loss_acc = criterion(logits_acc, inputs[:, 2].repeat(4, 1))
                    loss_bottoms = criterion(logits_bottoms, inputs[:, 3].repeat(4, 1))

                    loss = loss_shoes + loss_tops + loss_acc + loss_bottoms

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # update the loss value (multiply by the batch size)
                running_loss += loss.item() * inputs.size(0)

                pred_shoes = find_closest_embeddings(logits_shoes, model.catalogue_dict['shoes'], shoes_IDs, device)
                pred_tops = find_closest_embeddings(logits_tops, model.catalogue_dict['tops'], tops_IDs, device)
                pred_acc = find_closest_embeddings(logits_acc, model.catalogue_dict['accessories'], accessories_IDs,
                                                   device)
                pred_bottoms = find_closest_embeddings(logits_bottoms, model.catalogue_dict['bottoms'], bottoms_IDs,
                                                       device)

                # update the accuracy of the reconstruction task
                accuracy_shoes += np.sum(pred_shoes.cpu().numpy() == labels_shoes.cpu().numpy())
                accuracy_tops += np.sum(pred_tops.cpu().numpy() == labels_tops.cpu().numpy())
                accuracy_acc += np.sum(pred_acc.cpu().numpy() == labels_acc.cpu().numpy())
                accuracy_bottoms += np.sum(pred_bottoms.cpu().numpy() == labels_bottoms.cpu().numpy())

                # from the outputs of the model, retrieve only the predictions of the masked items
                # (the ones that are in the positions of the masked items)
                # compute the closest embeddings to the reconstructed embeddings
                top_k_predictions = find_top_k_closest_embeddings(masked_logits, model.catalogue_dict, masked_positions,
                                                                  shoes_IDs, tops_IDs, accessories_IDs, bottoms_IDs,
                                                                  device, topk=10)

                masked_IDs = []
                for i in range(len(labels_shoes)):
                    if masked_positions[i] == 0:
                        masked_IDs.append(labels_shoes[i].item())
                    if masked_positions[i] == 1:
                        masked_IDs.append(labels_tops[i].item())
                    if masked_positions[i] == 2:
                        masked_IDs.append(labels_acc[i].item())
                    if masked_positions[i] == 3:
                        masked_IDs.append(labels_bottoms[i].item())

                for i, id in enumerate(masked_IDs):
                    if id in top_k_predictions[i][0]:
                        hit_ratio += 1

            # compute the average loss of the epoch
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            # compute the average accuracy of the fill in the blank task of the epoch
            epoch_hit_ratio = hit_ratio
            epoch_accuracy_shoes = accuracy_shoes / (len(dataloaders[phase].dataset) * 4)
            epoch_accuracy_tops = accuracy_tops / (len(dataloaders[phase].dataset) * 4)
            epoch_accuracy_acc = accuracy_acc / (len(dataloaders[phase].dataset) * 4)
            epoch_accuracy_bottoms = accuracy_bottoms / (len(dataloaders[phase].dataset) * 4)

            print(f'{phase} Loss: {epoch_loss}')
            print(f'{phase} Accuracy shoes: {epoch_accuracy_shoes}')
            print(f'{phase} Accuracy tops: {epoch_accuracy_tops}')
            print(f'{phase} Accuracy accessories: {epoch_accuracy_acc}')
            print(f'{phase} Accuracy bottoms: {epoch_accuracy_bottoms}')
            print(f'{phase} Hit ratio (fill in the blank): {epoch_hit_ratio}')
            print(f'{phase} Hit ratio (fill in the blank) normalized: '
                  f'{epoch_hit_ratio / (len(dataloaders[phase].dataset) * 4)}')

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_hit_ratio.append(epoch_hit_ratio)
            else:
                val_loss.append(epoch_loss)
                val_hit_ratio.append(epoch_hit_ratio)

                # save model if validation loss has decreased
                if epoch_hit_ratio >= valid_hit_max:
                    print('Validation hit ratio increased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                        valid_hit_max, epoch_hit_ratio))
                    print('Validation hit ratio in reconstruction of the saved model: {:.6f}'.format(epoch_hit_ratio))
                    # save a checkpoint dictionary containing the model state_dict
                    checkpoint = {'d_model': model.d_model,
                                  'num_encoders': model.num_encoders,
                                  'num_heads': model.num_heads,
                                  'dropout': model.dropout,
                                  'dim_feedforward': model.dim_feedforward,
                                  'model_state_dict': model.state_dict()}
                    # save the checkpoint dictionary to a file
                    torch.save(checkpoint,
                               f"../checkpoints/umBERT_FT_NE_{model.num_encoders}_NH_{model.num_heads}_D_{model.dropout:.5f}"
                               f"_LR_{optimizer.param_groups[0]['lr']}_OPT_{type(optimizer).__name__}.pth")
                    valid_hit_max = epoch_hit_ratio
                    early_stopping = 0  # reset early stopping counter
                    best_model = deepcopy(model)
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
    return best_model, valid_hit_max


def test_model(model, device, dataloader, shoes_IDs, tops_IDs, accessories_IDs, bottoms_IDs, criterion):
    """
    Test the model on the test set of the fill in the blank task
    :param model: the model to test
    :param device: the device to use
    :param dataloader: the dataloader of the test set
    :param shoes_IDs: the IDs of the shoes
    :param tops_IDs: the IDs of the tops
    :param accessories_IDs: the IDs of the accessories
    :param bottoms_IDs: the IDs of the bottoms
    :param criterion: the loss function
    :return: None
    """
    running_loss = 0.0  # keep track of the loss
    hit_ratio = 0.0  # keep track of the accuracy of the fill in the blank task
    for inputs, labels in dataloader:  # for each batch
        inputs = inputs.to(device)  # move the input tensors to the GPU
        # labels are the IDs of the items in the outfit
        labels_shoes = labels[:, 0].to(device)  # move the labels_shoes to the device
        labels_tops = labels[:, 1].to(device)  # move the labels_tops to the device
        labels_acc = labels[:, 2].to(device)  # move the labels_acc to the device
        labels_bottoms = labels[:, 3].to(device)  # move the labels_bottoms to the device

        with torch.set_grad_enabled(False):
            # compute the forward pass
            logits_shoes, logits_tops, logits_acc, logits_bottoms, \
                masked_logits, masked_items, masked_positions = model.forward_fill_in_the_blank(inputs)

            # compute the loss
            # compute the loss for each masked item
            loss_shoes = criterion(logits_shoes, inputs[:, 0].repeat(4, 1))  # compute the loss for the masked item
            loss_tops = criterion(logits_tops, inputs[:, 1].repeat(4, 1))  # compute the loss for the masked item
            loss_acc = criterion(logits_acc, inputs[:, 2].repeat(4, 1))  # compute the loss for the masked item
            loss_bottoms = criterion(logits_bottoms, inputs[:, 3].repeat(4, 1))  # compute the loss for the masked item
            # normalize the loss
            loss = loss_shoes + loss_tops + loss_acc + loss_bottoms

        # update the loss value (multiply by the batch size)
        running_loss += loss.item() * inputs.size(0)

        # from the outputs of the model, retrieve only the predictions of the masked items

        #  implement top-k accuracy
        top_k_predictions = find_top_k_closest_embeddings(masked_logits, model.catalogue_dict, masked_positions,
                                                          shoes_IDs, tops_IDs, accessories_IDs, bottoms_IDs,
                                                          device, topk=10)

        masked_IDs = []

        for i in range(len(labels_shoes)):
            masked_IDs.append(labels_shoes[i].item())
            masked_IDs.append(labels_tops[i].item())
            masked_IDs.append(labels_acc[i].item())
            masked_IDs.append(labels_bottoms[i].item())
        for i, id in enumerate(masked_IDs):
            if id in top_k_predictions[i][0]:
                hit_ratio += 1

    # compute the average loss of the epoch
    loss = running_loss / (len(dataloader.dataset) * 4)
    print(f'Test Loss: {loss}')
    print(f'Test Hit ratio (fill in the blank): {hit_ratio}')
    print(f'Test Hit ratio normalized (fill in the blank): {hit_ratio / (len(dataloader.dataset) * 4)}')
