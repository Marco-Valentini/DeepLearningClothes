import torch

from BERT_architecture.umBERT import umBERT
from utility.custom_gather import custom_gather
import matplotlib.pyplot as plt
import numpy as np


class umBERT_trainer():
    """
    This class is used to train the umBERT model in three different ways:
    - BERT-like: MLM + classification tasks
    - MLM only
    - classification only (BC)
    """

    def __init__(self, model: umBERT, optimizer, criterion, device, n_epochs=500):
        """
        This function initializes the umBERT_trainer class with the following parameters:
        :param model: the umBERT model to train
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

    def pre_train_BERT_like(self, dataloaders):
        """
        This function performs the pre-training of the umBERT model in a BERT-like fashion (MLM + classification tasks)
        :param dataloaders: the dataloaders used to load the data (train and validation)
        :param masked_positions: the positions of the masked elements in the outfit(train and validation)
        :return: None
        """
        train_loss = []  # keep track of the loss of the training phase
        val_loss = []  # keep track of the loss of the validation phase
        train_acc_BC = []  # keep track of the accuracy of the training phase on the BC task
        val_acc_BC = []  # keep track of the accuracy of the validation phase on the MLM task
        train_acc_MLM = []  # keep track of the accuracy of the training phase on the MLM task
        val_acc_MLM = []  # keep track of the accuracy of the validation phase on the MLM task
        valid_loss_min = np.Inf  # track change in validation loss
        self.model = self.model.to(self.device)  # set the model to run on the device

        for epoch in range(self.n_epochs):
            for phase in ['train', 'val']:
                print(f'Epoch: {epoch + 1}/{self.n_epochs} | Phase: {phase}')
                if phase == 'train':
                    self.model.train()  # set model to training mode
                else:
                    self.model.eval()  # set model to evaluate mode

                running_loss = 0.0  # keep track of the loss
                accuracy_classification = 0.0  # keep track of the accuracy of the classification task
                accuracy_MLM = 0.0  # keep track of the accuracy of the MLM task

                for inputs, labels in dataloaders[phase]:  # for each batch
                    inputs = inputs.to(self.device)  # move the data to the device

                    # take the labels of the classification task
                    labels_BC = labels[:, 0]
                    # convert the tensor labels_BC to LongTensor
                    labels_BC = labels_BC.type(torch.LongTensor)
                    labels_BC = labels_BC.to(self.device)  # move labels_BC to the device
                    # do a one-hot encoding of the labels of the classification task and move them to the device
                    labels_BC_one_hot = torch.nn.functional.one_hot(labels_BC, num_classes=2).to(self.device)

                    # take the labels of the MLM task
                    labels_MLM = labels[:, 1]
                    # convert the tensor labels_MLM to LongTensor
                    labels_MLM = labels_MLM.type(torch.LongTensor)
                    labels_MLM = labels_MLM.to(self.device)  # move them to the device
                    # do a one-hot encoding of the labels of the MLM task and move them to the device
                    labels_MLM_one_hot = torch.nn.functional.one_hot(labels_MLM,
                                                                     num_classes=self.model.catalogue_size).to(
                        self.device)

                    # take the positions of the masked elements
                    masked_positions = labels[:, 2]
                    # move masked_positions to the device
                    masked_positions = masked_positions.to(self.device)

                    self.optimizer.zero_grad()  # zero the gradients

                    with torch.set_grad_enabled(
                            phase == 'train'):  # set the gradient computation only if in training phase
                        # compute the output of the model (forward pass) [batch_size, seq_len, d_model]
                        output = self.model.forward(inputs)

                        # compute the loss of the MLM task
                        masked_elements = custom_gather(outputs=output, masked_positions=masked_positions,
                                                        device=self.device)  # select the masked elements
                        logits = self.model.ffnn(masked_elements)  # compute the logits of the MLM task
                        logits = logits.view(-1, self.model.catalogue_size)  # reshape the logits
                        loss_MLM = self.criterion['MLM'](logits,
                                                         labels_MLM_one_hot.float())  # compute the loss of the MLM task

                        clf = self.model.Binary_Classifier(
                            output[:, 0, :])  # compute the logits of the classification task
                        # compute the loss of the classification task
                        loss_BC = self.criterion['BC'](clf, labels_BC_one_hot.float())
                        # compute the total loss (sum of the average values of the two losses)
                        loss = loss_MLM.mean() + loss_BC.mean()

                        if phase == 'train':
                            loss.backward()  # compute the gradients of the loss
                            self.optimizer.step()  # update the parameters

                    # update the loss value (multiply by the batch size)
                    running_loss += loss.item() * inputs.size(0)

                    # update the accuracy of the classification task
                    pred_labels_BC = torch.max((self.model.softmax(clf, dim=1)), dim=1).indices
                    pred_labels_BC = pred_labels_BC.to(self.device)  # move the predicted labels_train to the device
                    pred_labels_MLM = torch.max(self.model.softmax(logits, dim=1), dim=1).indices
                    pred_labels_MLM = pred_labels_MLM.to(self.device)  # move the predicted labels_train to the device
                    accuracy_classification += torch.sum(pred_labels_BC == labels_BC)
                    accuracy_MLM += torch.sum(pred_labels_MLM == labels_MLM)  # update the accuracy of the MLM task

                epoch_loss = running_loss / len(dataloaders[phase].dataset)  # compute the average loss of the epoch
                # compute the average accuracy of the classification task of the epoch
                epoch_accuracy_classification = accuracy_classification / len(dataloaders[phase].dataset)
                # compute the average accuracy of the MLM task of the epoch
                epoch_accuracy_MLM = accuracy_MLM / len(dataloaders[phase].dataset)

                print(f'{phase} Loss: {epoch_loss}')
                print(f'{phase} Accuracy (Classification): {epoch_accuracy_classification}')
                print(f'{phase} Accuracy (MLM): {epoch_accuracy_MLM}')
                if phase == 'train':
                    train_loss.append(epoch_loss)
                    train_acc_BC.append(epoch_accuracy_classification.item())
                    train_acc_MLM.append(epoch_accuracy_MLM.item())
                else:
                    val_loss.append(epoch_loss)
                    val_acc_BC.append(epoch_accuracy_classification.item())
                    val_acc_MLM.append(epoch_accuracy_MLM.item())
                    # save model if validation loss has decreased
                    if epoch_loss <= valid_loss_min:
                        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                            valid_loss_min,
                            epoch_loss))
                        print('Validation accuracy BC of the saved model: {:.6f}'.format(epoch_accuracy_classification))
                        print('Validation accuracy MLM of the saved model: {:.6f}'.format(epoch_accuracy_MLM))
                        # save a checkpoint dictionary containing the model state_dict
                        checkpoint = {'d_model': self.model.d_model, 'catalogue_size': self.model.catalogue_size,
                                      'num_encoders': self.model.num_encodes,
                                      'num_heads': self.model.num_heads, 'dropout': self.model.dropout,
                                      'dim_feedforward': self.model.dim_feedforward,
                                      'model_state_dict': self.model.state_dict()}
                        torch.save(checkpoint,
                                   'umBERT_pretrained_BC.pth')  # save the checkpoint dictionary to a file
                        valid_loss_min = epoch_loss
        plt.plot(train_loss, label='train')
        plt.plot(val_loss, label='val')
        plt.legend()
        plt.title('Loss for joint training')
        plt.show()
        plt.plot(train_acc_BC, label='train')
        plt.plot(val_acc_BC, label='val')
        plt.legend()
        plt.title('Accuracy (Classification) in joint training')
        plt.show()
        plt.plot(train_acc_MLM, label='train')
        plt.plot(val_acc_MLM, label='val')
        plt.legend()
        plt.title('Accuracy (MLM) in joint training')
        plt.show()

    def pre_train_BC(self, dataloaders):
        """
        This function performs the pre-training of the umBERT model only on the classification tasks
        :param dataloaders: the dataloaders used to load the data (train and validation)
        :return: None
        """
        train_loss = []
        val_loss = []
        train_acc = []
        val_acc = []
        valid_loss_min = np.Inf  # track change in validation loss
        for epoch in range(self.n_epochs):
            for phase in ['train', 'val']:
                print(f'Epoch: {epoch + 1}/{self.n_epochs} | Phase: {phase}')
                if phase == 'train':
                    self.model.train()  # set model to training mode
                else:
                    self.model.eval()  # set model to evaluate mode

                running_loss = 0.0  # keep track of the loss
                accuracy = 0.0  # keep track of the accuracy of the classification task

                for inputs, labels in dataloaders[phase]:  # for each batch
                    inputs = inputs.to(self.device)  # move the data to the device
                    labels = labels.to(self.device)  # move the data to the device

                    self.optimizer.zero_grad()  # zero the gradients

                    with torch.set_grad_enabled(
                            phase == 'train'):  # set the gradient computation only if in training phase
                        output = self.model.forward(inputs)  # compute the output of the model (forward pass)
                        clf = self.model.Binary_Classifier(output[:, 0, :])  # compute the logits
                        # clf will be the max value between the two final logits
                        pred_labels = torch.max(self.model.softmax(clf, dim=1),
                                                dim=1).indices  # compute the predicted labels
                        pred_labels = pred_labels.unsqueeze(-1)
                        labels_one_hot = torch.nn.functional.one_hot(labels.long(), num_classes=2).to(self.device)
                        loss = self.criterion(clf, labels_one_hot.float())  # compute the loss

                        if phase == 'train':
                            loss.backward()  # compute the gradients of the loss
                            self.optimizer.step()  # update the parameters

                    running_loss += loss.item() * inputs.size(0)  # update the loss value (multiply by the batch size)
                    # the accuracy must be computed on the predicted labels
                    labels = labels.unsqueeze(-1)
                    accuracy += torch.sum(pred_labels == labels)  # update the accuracy

                print(f'length of dataset is : {len(dataloaders[phase].dataset)}')
                epoch_loss = running_loss / len(dataloaders[phase].dataset)  # compute the average loss of the epoch
                epoch_accuracy = accuracy / len(dataloaders[phase].dataset)  # compute the average accuracy of the epoch
                if phase == 'train':
                    train_loss.append(epoch_loss)
                    train_acc.append(epoch_accuracy.item())
                else:
                    val_loss.append(epoch_loss)
                    val_acc.append(epoch_accuracy.item())
                    # save model if validation loss has decreased
                    if epoch_loss <= valid_loss_min:
                        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                            valid_loss_min,
                            epoch_loss))
                        print('Validation accuracy on BC of the saved model: {:.6f}'.format(epoch_accuracy))
                        # save a checkpoint dictionary containing the model state_dict
                        checkpoint = {'d_model': self.model.d_model, 'catalogue_size': self.model.catalogue_size,
                                      'num_encoders': self.model.num_encodes,
                                      'num_heads': self.model.num_heads, 'dropout': self.model.dropout,
                                      'dim_feedforward': self.model.dim_feedforward,
                                      'model_state_dict': self.model.state_dict()}
                        torch.save(checkpoint,
                                   'umBERT_pretrained_BC.pth')  # save the checkpoint dictionary to a file
                        valid_loss_min = epoch_loss

                print(f'{phase} Loss: {epoch_loss:.10f}')
                print(f'{phase} Accuracy : {epoch_accuracy:.10f}')
        plt.plot(train_loss, label='train loss')
        plt.plot(val_loss, label='val loss')
        plt.title('Loss in BC sequential training')
        plt.legend()
        plt.show()
        plt.plot(train_acc, label='train acc')
        plt.plot(val_acc, label='val acc')
        plt.title('Accuracy in BC sequential training')
        plt.legend()
        plt.show()

    def pre_train_MLM(self, dataloaders):
        """
        This function performs the pre-training of the umBERT model only on the MLM task
        :param dataloaders: the dataloaders used to load the data (train and validation)
        :param masked_positions: the positions of the masked elements (train and validation)
        :return: None
        """
        train_loss = []
        val_loss = []
        train_acc = []
        val_acc = []
        valid_loss_min = np.Inf  # track change in validation loss
        for epoch in range(self.n_epochs):
            for phase in ['train', 'val']:
                print(f'Epoch: {epoch + 1}/{self.n_epochs} | Phase: {phase}')
                if phase == 'train':
                    self.model.train()  # set model to training mode
                else:
                    self.model.eval()  # set model to evaluate mode

                running_loss = 0.0  # keep track of the loss
                accuracy = 0.0  # keep track of the accuracy of the MLM task

                for inputs, labels in dataloaders[phase]:  # for each batch
                    inputs = inputs.to(self.device)  # move the data to the device
                    labels_MLM = labels[:, 0]
                    masked_positions = labels[:, 1]
                    labels_MLM = labels_MLM.to(self.device)  # move the data to the device
                    masked_positions = masked_positions.to(self.device)  # move the data to the device

                    self.optimizer.zero_grad()  # zero the gradients

                    with torch.set_grad_enabled(
                            phase == 'train'):  # set the gradient computation only if in training phase
                        output = self.model.forward(inputs)  # compute the output of the model (forward pass)
                        masked_elements = custom_gather(output, masked_positions,
                                                        self.device)  # select the masked elements
                        logits = self.model.ffnn(masked_elements)  # compute the logits
                        logits = logits.view(-1, self.model.catalogue_size)
                        pred_labels = torch.max(self.model.softmax(logits, dim=1), dim=1).indices
                        labels_MLM_one_hot = torch.nn.functional.one_hot(labels_MLM.long(),
                                                                         num_classes=self.model.catalogue_size).to(
                            self.device)
                        loss = self.criterion(logits, labels_MLM_one_hot.float())  # compute the loss

                        if phase == 'train':
                            loss.backward()  # compute the gradients of the loss
                            self.optimizer.step()  # update the parameters

                    running_loss += loss.item() * inputs.size(0)  # update the loss value (multiply by the batch size)
                    labels_MLM = labels_MLM.unsqueeze(-1)
                    pred_labels = pred_labels.unsqueeze(-1)
                    accuracy += torch.sum(pred_labels == labels_MLM)  # update the accuracy of the MLM task

                epoch_loss = running_loss / len(dataloaders[phase].dataset)  # compute the average loss of the epoch
                epoch_accuracy = accuracy / len(
                    dataloaders[phase].dataset)  # compute the average accuracy of the MLM task of the epoch

                print(f'{phase} Loss: {epoch_loss}')
                print(f'{phase} Accuracy: {epoch_accuracy}')
                if phase == 'train':
                    train_loss.append(epoch_loss)
                    train_acc.append(epoch_accuracy.item())
                else:
                    val_loss.append(epoch_loss)
                    val_acc.append(epoch_accuracy.item())
                    # save model if validation loss has decreased
                    if epoch_loss <= valid_loss_min:
                        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                            valid_loss_min,
                            epoch_loss))
                        print('Validation accuracy on MLM of the saved model: {:.6f}'.format(epoch_accuracy))
                        # save a checkpoint dictionary containing the model state_dict
                        checkpoint = {'d_model': self.model.d_model, 'catalogue_size': self.model.catalogue_size,
                                      'num_encoders': self.model.num_encodes,
                                      'num_heads': self.model.num_heads, 'dropout': self.model.dropout,
                                      'dim_feedforward': self.model.dim_feedforward,
                                      'model_state_dict': self.model.state_dict()}
                        torch.save(checkpoint,
                                   'umBERT_pretrained_MLM.pth')  # save the checkpoint dictionary to a file
                        valid_loss_min = epoch_loss
        plt.plot(train_loss, label='train loss')
        plt.plot(val_loss, label='val loss')
        plt.title('Loss in MLM sequential training')
        plt.legend()
        plt.show()
        plt.plot(train_acc, label='train acc')
        plt.plot(val_acc, label='val acc')
        plt.title('Accuracy in MLM sequential training')
        plt.legend()
        plt.show()

    def fine_tuning(self, dataloaders, freeze=0, fine_tuning_epochs=10):
        if freeze > 0:
            count = 0
            for param in self.model.parameters():
                if count < freeze:
                    param.requires_grad = False
                count += 1
        train_loss = []
        val_loss = []
        train_acc = []
        val_acc = []
        valid_loss_min = np.Inf  # track change in validation loss

        self.model.to(self.device)

        for epoch in range(fine_tuning_epochs):
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0
                accuracy = 0.0

                for input, labels in dataloaders[phase]:
                    input = input.to(self.device)  # move the data to the device
                    labels_MLM = labels[:, 0]  # take the labels of the MLM task
                    masked_positions = labels[:, 1]  # take the masked positions
                    labels_MLM = labels_MLM.to(self.device)  # move the data to the device
                    masked_positions = masked_positions.to(self.device)  # move the data to the device

                    # optimizer is taken from the trainer (if necessary it is changed in the main)
                    self.optimizer.zero_grad()  # zero the gradients

                    with torch.set_grad_enabled(
                            phase == 'train'):  # set the gradient computation only if in training phase
                        output = self.model.forward(input)  # compute the output of the model (forward pass)
                        masked_elements = custom_gather(output, masked_positions,
                                                        self.device)  # select the masked elements
                        logits = self.model.ffnn(masked_elements)  # compute the logits
                        logits = logits.view(-1, self.model.catalogue_size)
                        pred_labels = torch.max(self.model.softmax(logits, dim=1), dim=1).indices  # predicted labels
                        labels_MLM_one_hot = torch.nn.functional.one_hot(labels_MLM.long(),
                                                                         num_classes=self.model.catalogue_size).to(
                            self.device)
                        loss = self.criterion(logits,
                                              labels_MLM_one_hot.float())  # compute the loss # define the loss from the main
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    running_loss += loss.item() * input.size(0)  # update the loss value (multiply by the batch size)
                    labels_MLM = labels_MLM.unsqueeze(-1)
                    pred_labels = pred_labels.unsqueeze(-1)
                    accuracy += torch.sum(pred_labels == labels_MLM)  # update the accuracy of the MLM task
                epoch_loss = running_loss / len(dataloaders[phase].dataset)  # compute the average loss of the epoch
                epoch_accuracy = accuracy / len(
                    dataloaders[phase].dataset)  # compute the average accuracy of the MLM task of the epoch

                print(f'{phase} Loss: {epoch_loss}')
                print(f'{phase} Accuracy: {epoch_accuracy}')
                if phase == 'train':
                    train_loss.append(epoch_loss)
                    train_acc.append(epoch_accuracy.item())
                else:
                    val_loss.append(epoch_loss)
                    val_acc.append(epoch_accuracy.item())
                    # save model if validation loss has decreased
                    if epoch_loss <= valid_loss_min:
                        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                            valid_loss_min,
                            epoch_loss))
                        print('Validation accuracy on MLM fine tuning of the saved model: {:.6f}'.format(epoch_accuracy))
                        # save a checkpoint dictionary containing the model state_dict
                        checkpoint = {'d_model': self.model.d_model, 'catalogue_size': self.model.catalogue_size,
                                      'num_encoders': self.model.num_encodes,
                                      'num_heads': self.model.num_heads, 'dropout': self.model.dropout,
                                      'dim_feedforward': self.model.dim_feedforward,
                                      'model_state_dict': self.model.state_dict()}
                        torch.save(checkpoint,
                                   'umBERT_fine_tuned.pth')
        plt.plot(train_loss, label='train loss for fine tuning')
        plt.plot(val_loss, label='val loss for fine tuning')
        plt.title('Loss in MLM fine tuning')
        plt.legend()
        plt.show()
        plt.plot(train_acc, label='train acc for fine tuning')
        plt.plot(val_acc, label='val acc for fine tuning')
        plt.title('Accuracy in MLM fine tuning')
        plt.legend()
        plt.show()


class umBERT_evaluator():
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def evaluate_BERT_like(self, dataloader):
        accuracy_MLM = self.test_MLM(dataloader)
        accuracy_BC = self.test_BC(dataloader)
        return accuracy_MLM, accuracy_BC

    def test_BC(self, dataloader):
        self.model.eval()
        accuracy = 0.0
        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)  # move the data to the device
            # take the labels of the classification task
            labels_BC = labels[:, 0]
            # convert the tensor labels_BC to LongTensor
            labels_BC = labels_BC.type(torch.LongTensor)
            labels_BC = labels_BC.to(self.device)  # move labels_BC to the device

            with torch.no_grad():
                preds = self.model.predict_BC(inputs)
                # compute the accuracy
                accuracy += torch.sum(preds == labels_BC)

        accuracy = accuracy / len(dataloader.dataset)
        return accuracy

    def test_MLM(self, dataloader):
        self.model.eval()
        accuracy = 0.0
        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)  # move the data to the device
            # take the labels of the MLM task
            labels_MLM = labels[:, 1]
            # convert the tensor labels_MLM to LongTensor
            labels_MLM = labels_MLM.type(torch.LongTensor)
            labels_MLM = labels_MLM.to(self.device)  # move them to the device

            # take the positions of the masked elements
            masked_positions = labels[:, 2]
            # move masked_positions to the device
            masked_positions = masked_positions.to(self.device)
            with torch.no_grad():
                preds = self.model.predict_MLM(inputs, masked_positions, self.device)
                # compute the accuracy
                accuracy += torch.sum(preds == labels_MLM)

        accuracy = accuracy / len(dataloader.dataset)
        return accuracy