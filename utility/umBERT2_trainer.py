import json

import torch
from torch.nn import CosineSimilarity

from BERT_architecture.umBERT2 import umBERT2
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

class umBERT2_trainer():
    """
    This class is used to train the umBERT2 model on the two tasks (MLM and classification)
    """

    def __init__(self, model: umBERT2, optimizer, criterion, device, n_epochs=500):
        """
        This function initializes the umBERT2_trainer class with the following parameters:
        :param model: the umBERT2 model to train
        :param optimizer: the optimizer used to update the parameters of the model
        :param criterion: the loss function used to compute the loss
        :param device: the device used to perform the computations (CPU or GPU)
        :param n_epochs: the number of epochs used to train the model
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.n_epochs = n_epochs
        self.cosine_similarity = CosineSimilarity(dim=1, eps=1e-6)

    def pre_train(self, dataloaders, run):
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

        for epoch in range(self.n_epochs):
            for phase in ['train', 'val']:
                print(f'Epoch: {epoch + 1}/{self.n_epochs} | Phase: {phase}')
                if phase == 'train':
                    self.model.train()  # set model to training mode
                    print("Training...")
                else:
                    self.model.eval()  # set model to evaluate mode
                    print("Validation...")

                running_loss = 0.0  # keep track of the loss
                accuracy_CLF = 0.0  # keep track of the accuracy of the classification task
                accuracy_shoes = 0.0  # keep track of the accuracy of shoes classification task
                accuracy_tops = 0.0  # keep track of the accuracy of tops classification task
                accuracy_acc = 0.0  # keep track of the accuracy of accessories classification task
                accuracy_bottoms = 0.0  # keep track of the accuracy of bottoms classification task

                for inputs, labels in dataloaders[phase]:  # for each batch
                    inputs = inputs.to(self.device)  # move the data to the device
                    labels_CLF = labels[:, 0].type(torch.LongTensor).to(self.device)  # move the labels_CLF to the device
                    # the labels of each item are the IDs in the catalogue
                    labels_shoes = labels[:, 1].type(torch.LongTensor).to(self.device)  # move the labels_shoes to the device
                    labels_tops = labels[:, 2].type(torch.LongTensor).to(self.device)  # move the labels_tops to the device
                    labels_acc = labels[:, 3].type(torch.LongTensor).to(self.device)  # move the labels_acc to the device
                    labels_bottoms = labels[:, 4].type(torch.LongTensor).to(self.device)  # move the labels_bottoms to the device

                    # do a one-hot encoding of the labels of the classification task and move them to the device
                    labels_CLF_one_hot = torch.nn.functional.one_hot(labels_CLF, num_classes=2).to(self.device)

                    # create a dictionary with the inputs of the model (reconstruction + classification tasks)
                    # will be used to compute the loss
                    dict_inputs = {
                        'clf': labels_CLF_one_hot,
                        'shoes': inputs[:, 1, :],
                        'tops': inputs[:, 2, :],
                        'accessories': inputs[:, 3, :],
                        'bottoms': inputs[:, 4, :]
                    }

                    self.optimizer.zero_grad()  # zero the gradients

                    # set the gradient computation only if in training phase
                    with torch.set_grad_enabled(phase == 'train'):
                        # compute the predictions of the model
                        dict_outputs = self.model.forward(inputs)

                        # compute the total loss (sum of the average values of the two losses)
                        loss = self.compute_loss(dict_outputs, dict_inputs)

                        if phase == 'train':
                            loss.backward()  # compute the gradients of the loss
                            self.optimizer.step()  # update the parameters

                    # update the loss value (multiply by the batch size)
                    running_loss += loss.item() * inputs.size(0)

                    # update the accuracy of the classification task
                    pred_labels_CLF = torch.max((self.model.softmax(dict_outputs['clf'], dim=1)), dim=1).indices
                    pred_labels_shoes = self.find_closest_embeddings(dict_outputs['shoes'])
                    pred_labels_tops = self.find_closest_embeddings(dict_outputs['tops'])
                    pred_labels_acc = self.find_closest_embeddings(dict_outputs['accessories'])
                    pred_labels_bottoms = self.find_closest_embeddings(dict_outputs['bottoms'])

                    # update the accuracy of the classification task
                    accuracy_CLF += torch.sum(pred_labels_CLF == labels_CLF)
                    # update the accuracy of the reconstruction task
                    accuracy_shoes += torch.sum(pred_labels_shoes == labels_shoes)
                    accuracy_tops += torch.sum(pred_labels_tops == labels_tops)
                    accuracy_acc += torch.sum(pred_labels_acc == labels_acc)
                    accuracy_bottoms += torch.sum(pred_labels_bottoms == labels_bottoms)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)  # compute the average loss of the epoch
                # compute the average accuracy of the classification task of the epoch
                epoch_accuracy_CLF = accuracy_CLF / len(dataloaders[phase].dataset)
                # compute the average accuracy of the items classification task of the epoch
                epoch_accuracy_shoes = accuracy_shoes / len(dataloaders[phase].dataset)
                epoch_accuracy_tops = accuracy_tops / len(dataloaders[phase].dataset)
                epoch_accuracy_acc = accuracy_acc / len(dataloaders[phase].dataset)
                epoch_accuracy_bottoms = accuracy_bottoms / len(dataloaders[phase].dataset)
                # compute the average accuracy of the MLM task of the epoch
                epoch_accuracy_decoding = (epoch_accuracy_shoes + epoch_accuracy_tops + epoch_accuracy_acc + epoch_accuracy_bottoms) / 4

                if run is not None:
                    run[f"{phase}/epoch/loss"].append(epoch_loss)
                    run[f"{phase}/epoch/acc_clf"].append(epoch_accuracy_CLF)
                    run[f"{phase}/epoch/acc_shoes"].append(epoch_accuracy_shoes)
                    run[f"{phase}/epoch/acc_tops"].append(epoch_accuracy_tops)
                    run[f"{phase}/epoch/acc_acc"].append(epoch_accuracy_acc)
                    run[f"{phase}/epoch/acc_bottoms"].append(epoch_accuracy_bottoms)
                    run[f"{phase}/epoch/acc_decoding"].append(epoch_accuracy_decoding)

                print(f'{phase} Loss: {epoch_loss}')
                print(f'{phase} Accuracy (Classification): {epoch_accuracy_CLF}')
                print(f'{phase} Accuracy (Shoes): {epoch_accuracy_shoes}')
                print(f'{phase} Accuracy (Tops): {epoch_accuracy_tops}')
                print(f'{phase} Accuracy (Accessories): {epoch_accuracy_acc}')
                print(f'{phase} Accuracy (Bottoms): {epoch_accuracy_bottoms}')
                print(f'{phase} Accuracy (decoding): {epoch_accuracy_decoding}')
                if phase == 'train':
                    train_loss.append(epoch_loss)
                    train_acc_CLF.append(epoch_accuracy_CLF.item())
                    train_acc_decoding.append(epoch_accuracy_decoding.item())
                else:
                    val_loss.append(epoch_loss)
                    val_acc_CLF.append(epoch_accuracy_CLF.item())
                    val_acc_decoding.append(epoch_accuracy_decoding.item())

                    # save model if validation loss has decreased
                    if epoch_loss <= valid_loss_min:
                        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                            valid_loss_min,
                            epoch_loss))
                        print('Validation accuracy BC of the saved model: {:.6f}'.format(epoch_accuracy_CLF))
                        print('Validation accuracy MLM of the saved model: {:.6f}'.format(epoch_accuracy_decoding))
                        # save a checkpoint dictionary containing the model state_dict
                        checkpoint = {'d_model': self.model.d_model,
                                      'num_encoders': self.model.num_encoders,
                                      'num_heads': self.model.num_heads,
                                      'dropout': self.model.dropout,
                                      'dim_feedforward': self.model.dim_feedforward,
                                      'model_state_dict': self.model.state_dict()}

                        torch.save(checkpoint,
                                   f'../models/umBERT2_pre_trained_{self.model.d_model}.pth')  # save the checkpoint dictionary to a file
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
        return valid_loss_min

    def compute_loss(self, dict_outputs, dict_input):
        """
        Compute the loss of the model by summing the loss of the classification task and the reconstruction task
        :param dict_outputs: the outputs of the model (dict)
        :param dict_input: the inputs of the model (dict)
        :return: the overall loss of the model (tensor)
        """
        target = torch.ones(dict_input['shoes'].shape[0]).to(self.device)  # target is a tensor of ones
        loss_shoes = self.criterion['recons'](dict_outputs['shoes'], dict_input['shoes'], target)
        loss_tops = self.criterion['recons'](dict_outputs['tops'], dict_input['tops'], target)
        loss_acc = self.criterion['recons'](dict_outputs['accessories'], dict_input['accessories'], target)
        loss_bottoms = self.criterion['recons'](dict_outputs['bottoms'], dict_input['bottoms'], target)
        if 'clf' in dict_input.keys():
            loss_clf = self.criterion['clf'](dict_outputs['clf'], dict_input['clf'].float())
            loss = loss_shoes + loss_tops + loss_acc + loss_bottoms + loss_clf
        else:
            loss = loss_shoes + loss_tops + loss_acc + loss_bottoms
        return loss

    def find_closest_embeddings(self, recons_embeddings):
        """
        Find the closest embeddings in the catalogue to the reconstructed embeddings
        :param recons_embeddings: the reconstructed embeddings (tensor)
        :return: the IDs of the closest embeddings (tensor)
        """
        embeddings = np.load(f'../reduced_data/embeddings_{self.model.d_model}.npy')
        embeddings = torch.from_numpy(embeddings).to(self.device)  # convert to tensor
        with open('../reduced_data/IDs_list') as f:
            IDs_list = json.load(f)
        closest_embeddings = []
        for i in range(recons_embeddings.shape[0]):  # for each reconstructed embedding in the batch
            # compute the cosine similarity between the reconstructed embedding and the embeddings of the catalogue
            distances = self.cosine_similarity(recons_embeddings[i, :], embeddings)
            # find the index of the closest embedding
            idx = torch.max(distances, dim=0).indices
            # retrieve the ID of the closest embedding
            closest_embeddings.append(IDs_list[idx])
        return torch.Tensor(closest_embeddings).to(self.device)

    def fine_tuning(self, dataloaders, run=None):
        """
        Fine-tuning of the model on the fill in the blank task
        :param dataloaders: the dataloaders of the training and validation sets (dict)
        :param run: the run of the experiment, to monitor the training on neptune.ai
        :return: the minimum validation loss
        """
        # fine-tuning on the fill in the blank task, input 4 tokens and one is masked
        # the model has to predict the masked token # there isn't CLS token in the input
        train_loss = []  # keep track of the loss of the training phase
        val_loss = []  # keep track of the loss of the validation phase
        train_acc_MLM = []  # keep track of the accuracy of the training phase on the MLM classification task
        val_acc_MLM = []  # keep track of the accuracy of the validation phase on the MLM classification task

        valid_loss_min = np.Inf  # track change in validation loss
        early_stopping = 0  # early stopping counter
        best_valid_acc_MLM = 0.0  # track change in validation accuracy

        for epoch in range(self.n_epochs):
            for phase in ['train', 'val']:
                print(f'Epoch: {epoch + 1}/{self.n_epochs} | Phase: {phase}')
                if phase == 'train':
                    self.model.train()  # set model to training mode
                    print("Training...")
                else:
                    self.model.eval()  # set model to evaluate mode
                    print("Validation...")

                running_loss = 0.0  # keep track of the loss
                accuracy_shoes = 0.0  # keep track of the accuracy of shoes classification task
                accuracy_tops = 0.0  # keep track of the accuracy of tops classification task
                accuracy_acc = 0.0  # keep track of the accuracy of accessories classification task
                accuracy_bottoms = 0.0  # keep track of the accuracy of bottoms classification task
                accuracy_MLM = 0.0  # keep track of the accuracy of the MLM classification task

                for inputs, labels in dataloaders[phase]:  # for each batch
                    # labels has to contain 4 tensors of the 4 items categories, labels are the item IDs
                    inputs = inputs.to(self.device)  # move the data to the device
                    labels_shoes = labels[:, 0].type(torch.LongTensor).to(
                        self.device)  # move the labels_shoes to the device
                    labels_tops = labels[:, 1].type(torch.LongTensor).to(
                        self.device)  # move the labels_tops to the device
                    labels_acc = labels[:, 2].type(torch.LongTensor).to(
                        self.device)  # move the labels_acc to the device
                    labels_bottoms = labels[:, 3].type(torch.LongTensor).to(
                        self.device)  # move the labels_bottoms to the device
                    masked_indexes = labels[:, 4].type(torch.LongTensor).to(
                        self.device)  # move the masked_indexes to the device
                    masked_IDs = labels[:, 5].type(torch.LongTensor).to(
                        self.device)  # move the masked_IDs to the device

                    # these are the embeddings of the items in the catalogue
                    dict_inputs = {
                        'shoes': inputs[:, 1, :],
                        'tops': inputs[:, 2, :],
                        'accessories': inputs[:, 3, :],
                        'bottoms': inputs[:, 4, :]
                    }

                    self.optimizer.zero_grad()  # zero the gradients

                    # set the gradient computation only if in training phase
                    with torch.set_grad_enabled(phase == 'train'):
                        # compute the predictions of the model
                        dict_outputs = self.model.forward_fine_tune(inputs) # forward pass computes the

                        # compute the total loss (sum of the average values of the two losses)
                        loss = self.compute_loss(dict_outputs, dict_inputs)

                        if phase == 'train':
                            loss.backward()  # compute the gradients of the loss
                            self.optimizer.step()  # update the parameters

                    # update the loss value (multiply by the batch size)
                    running_loss += loss.item() * inputs.size(0)

                    # update the accuracy of the classification task
                    pred_labels_shoes = self.find_closest_embeddings(dict_outputs['shoes'])
                    pred_labels_tops = self.find_closest_embeddings(dict_outputs['tops'])
                    pred_labels_acc = self.find_closest_embeddings(dict_outputs['accessories'])
                    pred_labels_bottoms = self.find_closest_embeddings(dict_outputs['bottoms'])  # this is an ID list

                    pred_masked = []
                    for i in range(len(masked_indexes)):
                        if masked_indexes[i] == 1:  # shoes
                            pred_masked.append(pred_labels_shoes[i])
                        elif masked_indexes[i] == 2:  # tops
                            pred_masked.append(pred_labels_tops[i])
                        elif masked_indexes[i] == 3:  # accessories
                            pred_masked.append(pred_labels_acc[i])
                        elif masked_indexes[i] == 4:  # bottoms
                            pred_masked.append(pred_labels_bottoms[i])
                    pred_masked = torch.Tensor(pred_masked).to(self.device)

                    # update the accuracies
                    accuracy_shoes += torch.sum(pred_labels_shoes == labels_shoes)
                    accuracy_tops += torch.sum(pred_labels_tops == labels_tops)
                    accuracy_acc += torch.sum(pred_labels_acc == labels_acc)
                    accuracy_bottoms += torch.sum(pred_labels_bottoms == labels_bottoms)
                    accuracy_MLM += torch.sum(pred_masked == masked_IDs)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)  # compute the average loss of the epoch
                # compute the average accuracy of the items classification task of the epoch
                epoch_accuracy_shoes = accuracy_shoes / len(dataloaders[phase].dataset)
                epoch_accuracy_tops = accuracy_tops / len(dataloaders[phase].dataset)
                epoch_accuracy_acc = accuracy_acc / len(dataloaders[phase].dataset)
                epoch_accuracy_bottoms = accuracy_bottoms / len(dataloaders[phase].dataset)

                # compute the average accuracy of the MLM task of the epoch
                epoch_accuracy_MLM = accuracy_MLM / len(dataloaders[phase].dataset)

                if run is not None:
                    run[f"{phase}/epoch/loss"].append(epoch_loss)
                    run[f"{phase}/epoch/acc_shoes"].append(epoch_accuracy_shoes)
                    run[f"{phase}/epoch/acc_tops"].append(epoch_accuracy_tops)
                    run[f"{phase}/epoch/acc_acc"].append(epoch_accuracy_acc)
                    run[f"{phase}/epoch/acc_bottoms"].append(epoch_accuracy_bottoms)
                    run[f"{phase}/epoch/acc_decoding"].append(epoch_accuracy_MLM)

                print(f'{phase} Loss: {epoch_loss}')
                print(f'{phase} Accuracy (Shoes): {epoch_accuracy_shoes}')
                print(f'{phase} Accuracy (Tops): {epoch_accuracy_tops}')
                print(f'{phase} Accuracy (Accessories): {epoch_accuracy_acc}')
                print(f'{phase} Accuracy (Bottoms): {epoch_accuracy_bottoms}')
                print(f'{phase} Accuracy (MLM): {epoch_accuracy_MLM}')
                if phase == 'train':
                    train_loss.append(epoch_loss)
                    train_acc_MLM.append(epoch_accuracy_MLM.item())
                else:
                    val_loss.append(epoch_loss)
                    val_acc_MLM.append(epoch_accuracy_MLM.item())

                    # save model if validation loss has decreased
                    if epoch_loss <= valid_loss_min:
                        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                            valid_loss_min,
                            epoch_loss))
                        print('Validation accuracy MLM of the saved model: {:.6f}'.format(epoch_accuracy_MLM))
                        # save a checkpoint dictionary containing the model state_dict
                        checkpoint = {'d_model': self.model.d_model,
                                      'num_encoders': self.model.num_encoders,
                                      'num_heads': self.model.num_heads,
                                      'dropout': self.model.dropout,
                                      'dim_feedforward': self.model.dim_feedforward,
                                      'model_state_dict': self.model.state_dict()}

                        torch.save(checkpoint,
                                   f'../models/umBERT2_finetuned_{self.model.d_model}.pth')  # save the checkpoint dictionary to a file
                        valid_loss_min = epoch_loss
                        best_valid_acc_MLM = epoch_accuracy_MLM
                        early_stopping = 0  # reset early stopping counter
                    else:
                        early_stopping += 1  # increment early stopping counter
            if early_stopping == 7:
                print('Early stopping the training')
                break
        plt.plot(train_loss, label='train')
        plt.plot(val_loss, label='val')
        plt.legend()
        plt.title('Loss fine-tuning')
        plt.show()
        plt.plot(train_acc_MLM, label='train')
        plt.plot(val_acc_MLM, label='val')
        plt.legend()
        plt.title('Accuracy (MLM) fine-tuning')
        plt.show()
        return best_valid_acc_MLM

    def evaluate_fine_tuning(self, dataloaders):
        """
        Evaluate the model on the test set
        :param dataloaders: dictionary containing the dataloaders for the train, validation and test sets
        :return: the accuracy of the MLM task on the test set
        """
        self.model.eval()
        with torch.no_grad():
            accuracy_shoes = 0.0  # keep track of the accuracy of shoes classification task
            accuracy_tops = 0.0  # keep track of the accuracy of tops classification task
            accuracy_acc = 0.0  # keep track of the accuracy of accessories classification task
            accuracy_bottoms = 0.0  # keep track of the accuracy of bottoms classification task

            for inputs, labels in dataloaders:  # for each batch
                # labels has to contain 4 tensors of the 4 items categories
                inputs = inputs.to(self.device)  # move the data to the device
                labels_shoes = labels[:, 0].type(torch.LongTensor).to(
                    self.device)  # move the labels_shoes to the device
                labels_tops = labels[:, 1].type(torch.LongTensor).to(
                    self.device)  # move the labels_tops to the device
                labels_acc = labels[:, 2].type(torch.LongTensor).to(
                    self.device)  # move the labels_acc to the device
                labels_bottoms = labels[:, 3].type(torch.LongTensor).to(
                    self.device)  # move the labels_bottoms to the device
                # compute the predictions of the model
                dict_outputs = self.model.forward_fine_tune(inputs)  # forward pass computes the


                # update the accuracy of the classification task
                pred_labels_shoes = self.find_closest_embeddings(dict_outputs['shoes'])
                pred_labels_tops = self.find_closest_embeddings(dict_outputs['tops'])
                pred_labels_acc = self.find_closest_embeddings(dict_outputs['accessories'])
                pred_labels_bottoms = self.find_closest_embeddings(dict_outputs['bottoms'])  # this is an ID list

                # update the accuracy of the MLM task
                accuracy_shoes += torch.sum(pred_labels_shoes == labels_shoes)
                accuracy_tops += torch.sum(pred_labels_tops == labels_tops)
                accuracy_acc += torch.sum(pred_labels_acc == labels_acc)
                accuracy_bottoms += torch.sum(pred_labels_bottoms == labels_bottoms)

            # compute the average accuracy of the items classification task of the epoch
            accuracy_shoes = accuracy_shoes / len(dataloaders.dataset)
            accuracy_tops = accuracy_tops / len(dataloaders.dataset)
            accuracy_acc = accuracy_acc / len(dataloaders.dataset)
            accuracy_bottoms = accuracy_bottoms / len(dataloaders.dataset)

        return accuracy_shoes, accuracy_tops, accuracy_acc, accuracy_bottoms

    def evaluate_pre_training(self, dataloader):
        """
        Evaluate the model on the test set
        :param dataloader: dictionary containing the dataloader test set
        :return: the accuracy of the MLM task on the test set
        """
        self.model.eval()
        with torch.no_grad():
            accuracy_shoes = 0.0  # keep track of the accuracy of shoes classification task
            accuracy_tops = 0.0  # keep track of the accuracy of tops classification task
            accuracy_acc = 0.0  # keep track of the accuracy of accessories classification task
            accuracy_bottoms = 0.0  # keep track of the accuracy of bottoms classification task

            for inputs, labels in dataloader:  # for each batch
                # labels has to contain 4 tensors of the 4 items categories
                inputs = inputs.to(self.device)  # move the data to the device
                labels_shoes = labels[:, 1].type(torch.LongTensor).to(
                    self.device)  # move the labels_shoes to the device
                labels_tops = labels[:, 2].type(torch.LongTensor).to(
                    self.device)  # move the labels_tops to the device
                labels_acc = labels[:, 3].type(torch.LongTensor).to(
                    self.device)  # move the labels_acc to the device
                labels_bottoms = labels[:, 4].type(torch.LongTensor).to(
                    self.device)  # move the labels_bottoms to the device
                # compute the predictions of the model
                dict_outputs = self.model.forward_fine_tune(inputs)  # forward pass computes the

                # update the accuracy of the classification task
                pred_labels_shoes = self.find_closest_embeddings(dict_outputs['shoes'])
                pred_labels_tops = self.find_closest_embeddings(dict_outputs['tops'])
                pred_labels_acc = self.find_closest_embeddings(dict_outputs['accessories'])
                pred_labels_bottoms = self.find_closest_embeddings(dict_outputs['bottoms'])  # this is an ID list

                # update the accuracy of the MLM task
                accuracy_shoes += torch.sum(pred_labels_shoes == labels_shoes)
                accuracy_tops += torch.sum(pred_labels_tops == labels_tops)
                accuracy_acc += torch.sum(pred_labels_acc == labels_acc)
                accuracy_bottoms += torch.sum(pred_labels_bottoms == labels_bottoms)

            # compute the average accuracy of the items classification task of the epoch
            accuracy_shoes = accuracy_shoes / len(dataloader.dataset)
            accuracy_tops = accuracy_tops / len(dataloader.dataset)
            accuracy_acc = accuracy_acc / len(dataloader.dataset)
            accuracy_bottoms = accuracy_bottoms / len(dataloader.dataset)

            print(f'Accuracy (Shoes): {accuracy_shoes}')
            print(f'Accuracy (Tops): {accuracy_tops}')
            print(f'Accuracy (Accessories): {accuracy_acc}')
            print(f'Accuracy (Bottoms): {accuracy_bottoms}')

