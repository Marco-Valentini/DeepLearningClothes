import torch
from utility.custom_gather import custom_gather
import matplotlib.pyplot as plt


class umBERT_trainer():
    """
    This class is used to train the umBERT model in three different ways:
    - BERT-like: MLM + classification tasks
    - MLM only
    - classification only (BC)
    """

    def __init__(self, model, optimizer, criterion, device, n_epochs=500):
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
                    # do a one-hot encoding of the labels of the classification task and move them to the device
                    labels_BC = torch.nn.functional.one_hot(labels_BC, num_classes=2).to(self.device)

                    # take the labels of the MLM task
                    labels_MLM = labels[:, 1]
                    # convert the tensor labels_MLM to LongTensor
                    labels_MLM = labels_MLM.type(torch.LongTensor)
                    # do a one-hot encoding of the labels of the MLM task and move them to the device
                    labels_MLM = torch.nn.functional.one_hot(labels_MLM, num_classes=self.model.catalogue_size).to(self.device)

                    # take the positions of the masked elements
                    masked_positions = labels[:, 2]
                    # move masked_positions to the device
                    masked_positions = masked_positions.to(self.device)

                    self.optimizer.zero_grad()  # zero the gradients

                    with torch.set_grad_enabled(phase == 'train'):  # set the gradient computation only if in training phase
                        # compute the output of the model (forward pass) [batch_size, seq_len, d_model]
                        output = self.model.forward(inputs)

                        # compute the loss of the MLM task
                        masked_elements = custom_gather(outputs=output, masked_positions=masked_positions,
                                                        device=self.device)  # select the masked elements

                        logits = self.model.ffnn(masked_elements)  # compute the logits of the MLM task
                        loss_MLM = self.criterion['MLM'](logits, labels_MLM)  # compute the loss of the MLM task

                        clf = self.model.Binary_Classifier(output[:, 0, :])  # compute the logits of the classification task
                        # convert the type of clf and labels_BC to the type required by the loss function (BCEWithLogitsLoss)
                        clf = clf.type(torch.FloatTensor).to(self.device)
                        labels_BC = labels_BC.type(torch.FloatTensor).to(self.device)
                        # compute the loss of the classification task
                        loss_BC = self.criterion['BC'](clf, labels_BC)
                        print(f'--- MLM loss: {loss_MLM} ---')
                        print(f'--- mean MLM loss: {loss_MLM.mean()} ---')
                        print(f'--- BC loss: {loss_BC} ---')
                        print(f'--- mean BC loss: {loss_BC.mean()} ---')
                        # compute the total loss (sum of the average values of the two losses)
                        loss = loss_MLM.mean() + loss_BC.mean()

                        if phase == 'train':
                            loss.backward()  # compute the gradients of the loss
                            self.optimizer.step()  # update the parameters

                    # update the loss value (multiply by the batch size)
                    running_loss += loss.item() * inputs.size(0)

                    # update the accuracy of the classification task
                    predictions_BC = torch.max(self.model.sigmoid(clf), dim=1).indices
                    labels_BC = torch.max(labels_BC, dim=1).indices
                    accuracy_classification += torch.sum(predictions_BC == labels_BC)

                    # update the accuracy of the MLM task
                    prediction_MLM = torch.max(self.model.softmax(logits), dim=1).indices
                    accuracy_MLM += torch.sum(prediction_MLM == labels_MLM)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)  # compute the average loss of the epoch
                # compute the average accuracy of the classification task of the epoch
                epoch_accuracy_classification = accuracy_classification / len(dataloaders[phase].dataset)
                # compute the average accuracy of the MLM task of the epoch
                epoch_accuracy_MLM = accuracy_MLM / len(dataloaders[phase].dataset)

                if phase == 'train':
                    train_loss.append(epoch_loss)
                    train_acc_BC.append(epoch_accuracy_classification)
                    train_acc_MLM.append(epoch_accuracy_MLM)
                else:
                    val_loss.append(epoch_loss)
                    val_acc_BC.append(epoch_accuracy_classification)
                    val_acc_MLM.append(epoch_accuracy_MLM)

                print(f'{phase} - correct clf predictions: {accuracy_classification}/ {len(dataloaders[phase].dataset)}')
                print(f'{phase} - correct MLM predictions: {accuracy_MLM}/ {len(dataloaders[phase].dataset)}')
                print(f'{phase} - len dataset: {len(dataloaders[phase].dataset)}')
                print(f'{phase} - Loss: {epoch_loss}')
                print(f'{phase} - Accuracy (Classification): {epoch_accuracy_classification}')
                print(f'{phase} - Accuracy (MLM): {epoch_accuracy_MLM}')

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
                        pred_labels = torch.max(self.model.sigmoid(clf), dim=1).indices  # compute the predicted labels
                        pred_labels = pred_labels.unsqueeze(-1)
                        #TODO check that clf dimension is 8,2
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
                    train_acc.append(epoch_accuracy)
                else:
                    val_loss.append(epoch_loss)
                    val_acc.append(epoch_accuracy)

                print(f'{phase} Loss: {epoch_loss:.10f}')
                print(f'{phase} Accuracy : {epoch_accuracy:.10f}')
        plt.plot(train_loss, label='train loss')
        plt.plot(val_loss, label='val loss')
        plt.legend()
        plt.show()
        plt.plot(train_acc, label='train acc')
        plt.plot(val_acc, label='val acc')
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
                    labels_MLM = labels[:,0]
                    masked_positions = labels[:,1]
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
                        pred_labels = torch.max(self.model.softmax(logits,dim=1), dim=1).indices
                        labels_MLM_one_hot = torch.nn.functional.one_hot(labels_MLM.long(), num_classes=self.model.catalogue_size).to(self.device)
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
                    train_acc.append(epoch_accuracy)
                else:
                    val_loss.append(epoch_loss)
                    val_acc.append(epoch_accuracy)
        plt.plot(train_loss, label='train loss')
        plt.plot(val_loss, label='val loss')
        plt.legend()
        plt.show()
        plt.plot(train_acc, label='train acc')
        plt.plot(val_acc, label='val acc')
        plt.legend()
        plt.show()

