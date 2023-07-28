import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torch.nn import CosineSimilarity
from datetime import datetime
from copy import deepcopy

# utility functions
# device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
# device = torch.device("cpu")

def create_tensor_dataset_from_dataframe(df_outfit: pd.DataFrame, embeddings, ids):
    """
    This function takes as input a dataframe containing the labels of the items in the outfit, the embeddings of the items and the ids of the items.
    It returns a tensor of shape (n_outfits, seq_len, embedding_size) containing the embeddings of the items in the outfit and the CLS token.
    :param df_outfit:  dataframe containing the labels of the items in the outfit
    :param embeddings:  embeddings of the items in the outfit (a tensor of shape (n_items, embedding_size))
    :param ids:  ids of the items in the outfit (a list of length n_items)
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


def find_closest_embeddings(recons_embeddings, embeddings, IDs_list, device):
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
    for i in range(recons_embeddings.shape[0]):  # for each reconstructed embedding in the batch
        # compute the cosine similarity between the reconstructed embedding and the embeddings of the catalogue
        distances = torch.cdist(recons_embeddings[i, :].unsqueeze(0), embeddings)
        # find the index of the closest embedding
        idx = torch.min(distances, dim=1).indices
        # append the closest embedding to the list
        closest_embeddings.append(IDs_list[idx])
    closest_embeddings = np.array(closest_embeddings)
    return torch.LongTensor(closest_embeddings).to(device)


def find_top_k_closest_embeddings(recons_embeddings, embeddings_dict, masked_positions, shoes_IDs, tops_IDs, accessories_IDs, bottoms_IDs, device, topk=10):
    embeddings_shoes = embeddings_dict['shoes']
    embeddings_tops = embeddings_dict['tops']
    embeddings_accessories = embeddings_dict['accessories']
    embeddings_bottoms = embeddings_dict['bottoms']
    closest_embeddings = []
    for i, pos in enumerate(masked_positions):
        if pos == 0:  # shoes
            # compute the cosine similarity between the reconstructed embedding and the embeddings of the catalogue
            distances = torch.cdist(recons_embeddings[i, :].unsqueeze(0), torch.Tensor(embeddings_shoes).to(device))
            idx = torch.topk(distances, k=topk, largest=False).indices
            idx = idx.tolist()
            closest_embeddings.append([shoes_IDs[j] for j in idx])
        elif pos == 1:  # tops
            # compute the cosine similarity between the reconstructed embedding and the embeddings of the catalogue
            distances = torch.cdist(recons_embeddings[i, :].unsqueeze(0), torch.Tensor(embeddings_tops).to(device))
            idx = torch.topk(distances, k=topk, largest=False).indices
            idx = idx.tolist()
            closest_embeddings.append([tops_IDs[j] for j in idx])
        elif pos == 2:  # accessories
            # compute the cosine similarity between the reconstructed embedding and the embeddings of the catalogue
            distances = torch.cdist(recons_embeddings[i, :].unsqueeze(0), torch.Tensor(embeddings_accessories).to(device))
            idx = torch.topk(distances, k=topk, largest=False).indices
            idx = idx.tolist()
            closest_embeddings.append([accessories_IDs[j] for j in idx])
        elif pos == 3:  # bottoms
            # compute the cosine similarity between the reconstructed embedding and the embeddings of the catalogue
            distances = torch.cdist(recons_embeddings[i, :].unsqueeze(0), torch.Tensor(embeddings_bottoms).to(device))
            idx = torch.topk(distances, k=topk, largest=False).indices
            idx = idx.tolist()
            closest_embeddings.append([bottoms_IDs[j] for j in idx])
    return closest_embeddings


def pre_train_BC(model, dataloaders, optimizer, criterion, n_epochs, device, run):
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
    best_model = deepcopy(model)
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
                    best_model = deepcopy(model)
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
    plt.title('Accuracy (Classification) pre-training')
    plt.show()
    return best_model, valid_loss_min


def pre_train_reconstruction(model, dataloaders, optimizer, criterion, n_epochs, run, shoes_IDs, tops_IDs,
                             accessories_IDs, bottoms_IDs, device):
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
                pred_acc = find_closest_embeddings(logits_acc, model.catalogue_dict['accessories'], accessories_IDs, device)
                pred_bottoms = find_closest_embeddings(logits_bottoms, model.catalogue_dict['bottoms'], bottoms_IDs, device)

                # update the accuracy of the reconstruction task
                accuracy_shoes += np.sum(pred_shoes.cpu().numpy() == labels_shoes.cpu().numpy())
                accuracy_tops += np.sum(pred_tops.cpu().numpy() == labels_tops.cpu().numpy())
                accuracy_acc += np.sum(pred_acc.cpu().numpy() == labels_acc.cpu().numpy())
                accuracy_bottoms += np.sum(pred_bottoms.cpu().numpy() == labels_bottoms.cpu().numpy())

            # compute the average loss of the epoch
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            # compute the average accuracy of the MLM task of the epoch
            epoch_accuracy_shoes = accuracy_shoes / len(dataloaders[phase].dataset)
            epoch_accuracy_tops = accuracy_tops / len(dataloaders[phase].dataset)
            epoch_accuracy_acc = accuracy_acc / len(dataloaders[phase].dataset)
            epoch_accuracy_bottoms = accuracy_bottoms / len(dataloaders[phase].dataset)
            epoch_accuracy_reconstruction = (epoch_accuracy_shoes + epoch_accuracy_tops + epoch_accuracy_acc + epoch_accuracy_bottoms) / 4

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
                               f'./models/{dt_string}_umBERT4_pre_trained_reconstruction_with_MSE_{model.d_model}.pth')
                    valid_loss_min = epoch_loss  # update the minimum validation loss
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
    return best_model, valid_loss_min


def fine_tune(model, dataloaders, optimizer, criterion, n_epochs,shoes_IDs, tops_IDs,
                             accessories_IDs, bottoms_IDs,device, run):
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
                    logits_shoes, logits_tops, logits_acc, logits_bottoms,masked_logits, masked_items, masked_positions = model.forward_fill_in_the_blank(inputs)

                    # compute the loss
                    # compute the loss for each masked item
                    loss_shoes = criterion(logits_shoes, inputs[:,0].repeat(4,1))  # compute the loss for the masked item
                    loss_tops = criterion(logits_tops, inputs[:,1].repeat(4,1))  # compute the loss for the masked item
                    loss_acc = criterion(logits_acc, inputs[:,2].repeat(4,1))  # compute the loss for the masked item
                    loss_bottoms = criterion(logits_bottoms, inputs[:,3].repeat(4,1))  # compute the loss for the masked item
                    # normalize the loss
                    loss = loss_shoes + loss_tops + loss_acc + loss_bottoms

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
                top_k_predictions = find_top_k_closest_embeddings(masked_logits, model.catalogue_dict, masked_positions, shoes_IDs, tops_IDs,
                                accessories_IDs, bottoms_IDs, device, topk=10)

                masked_IDs = []

                for i in range(len(labels_shoes)):
                    masked_IDs.append(labels_shoes[i].item())
                    masked_IDs.append(labels_tops[i].item())
                    masked_IDs.append(labels_acc[i].item())
                    masked_IDs.append(labels_bottoms[i].item())
                # TODO capire come calcolare accuracy
                for i, id in enumerate(masked_IDs):
                    if id in top_k_predictions[i][0]:
                        hit_ratio += 1

                # # compute the accuracy of the fill in the blank task
                # accuracy += torch.sum(predictions == masked_IDs)

            # compute the average loss of the epoch
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            # compute the average accuracy of the fill in the blank task of the epoch
            epoch_hit_ratio = hit_ratio  # / len(dataloaders[phase].dataset) # TODO valuta per cosa dividi qui

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
    return best_model, valid_loss_min

def test_model(model, device, dataloader, shoes_IDs, tops_IDs, accessories_IDs, bottoms_IDs, criterion):
    running_loss = 0.0  # keep track of the loss
    hit_ratio = 0.0  # keep track of the accuracy of the fill in the blank task
    # batch_number = 0
    for inputs, labels in dataloader:  # for each batch
        # print(f'Batch: {batch_number}/{len(dataloaders[phase])}')
        # batch_number += 1
        inputs = inputs.to(device)  # move the input tensors to the GPU
        # labels are the IDs of the items in the outfit
        labels_shoes = labels[:, 0].to(device)  # move the labels_shoes to the device
        labels_tops = labels[:, 1].to(device)  # move the labels_tops to the device
        labels_acc = labels[:, 2].to(device)  # move the labels_acc to the device
        labels_bottoms = labels[:, 3].to(device)  # move the labels_bottoms to the device

        with torch.set_grad_enabled(False):  # forward + backward + optimize only if in training phase
            # compute the forward pass
            logits_shoes, logits_tops, logits_acc, logits_bottoms, masked_logits, masked_items, masked_positions = model.forward_fill_in_the_blank(
                inputs)

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
                                                          shoes_IDs, tops_IDs,
                                                          accessories_IDs, bottoms_IDs, device, topk=10)

        masked_IDs = []

        for i in range(len(labels_shoes)):
            masked_IDs.append(labels_shoes[i].item())
            masked_IDs.append(labels_tops[i].item())
            masked_IDs.append(labels_acc[i].item())
            masked_IDs.append(labels_bottoms[i].item())
        # TODO capire come calcolare accuracy
        for i, id in enumerate(masked_IDs):
            if id in top_k_predictions[i][0]:
                hit_ratio += 1

    # compute the average loss of the epoch
    epoch_loss = running_loss / (len(dataloader.dataset)*4)
    # compute the average accuracy of the fill in the blank task of the epoch
    epoch_hit_ratio = hit_ratio  # / len(dataloaders[phase].dataset) # TODO valuta per cosa dividi qui
    print(f'Test Loss: {epoch_loss}')
    print(f'Test Hit ratio (fill in the blank): {epoch_hit_ratio}')
    print(f'Test Hit ratio normalized (fill in the blank): {epoch_hit_ratio/(len(dataloader.dataset)*4)}')

