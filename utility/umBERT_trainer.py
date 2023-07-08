import torch
from utility.custom_gather import custom_gather


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
                        output = self.model.forward(inputs)  # compute the output of the model (forward pass) [batch_size, seq_len, d_model]
                        masked_elements = custom_gather(outputs=output, masked_positions=masked_positions, device=self.device)  # select the masked elements

                        logits = self.model.ffnn(masked_elements)  # compute the logits of the MLM task
                        loss_MLM = self.criterion['MLM'](logits, labels_MLM)  # compute the loss of the MLM task

                        clf = self.model.Binary_Classifier(output[:, 0, :])  # compute the logits of the classification task
                        clf = clf.type(torch.LongTensor).to(self.device)  # convert clf to long tensor
                        # compute the loss of the classification task
                        print(f'clf shape: {clf.shape}')
                        print(f'clf type: {clf.type()}')
                        print(f'labels_BC shape: {labels_BC.shape}')
                        print(f'labels_BC type: {labels_BC.type()}')
                        loss_BC = self.criterion['BC'](clf, labels_BC)
                        # compute the total loss (sum of the average values of the two losses)
                        loss = loss_MLM.mean() + loss_BC.mean()

                        if phase == 'train':
                            loss.backward()  # compute the gradients of the loss
                            self.optimizer.step()  # update the parameters

                    running_loss += loss.item() * inputs.size(0)  # update the loss value (multiply by the batch size)
                    # update the accuracy of the classification task
                    accuracy_classification += torch.sum(torch.argmax(clf, dim=1) == labels_BC)
                    accuracy_MLM += torch.sum(torch.argmax(logits, dim=1) == labels_MLM)  # update the accuracy of the MLM task

                epoch_loss = running_loss / len(dataloaders[phase].dataset)  # compute the average loss of the epoch
                # compute the average accuracy of the classification task of the epoch
                epoch_accuracy_classification = accuracy_classification / len(dataloaders[phase].dataset)
                # compute the average accuracy of the MLM task of the epoch
                epoch_accuracy_MLM = accuracy_MLM / len(dataloaders[phase].dataset)

                print(f'{phase} Loss: {epoch_loss}')
                print(f'{phase} Accuracy (Classification): {epoch_accuracy_classification}')
                print(f'{phase} Accuracy (MLM): {epoch_accuracy_MLM}')

    def pre_train_BC(self, dataloaders):
        """
        This function performs the pre-training of the umBERT model only on the classification tasks
        :param dataloaders: the dataloaders used to load the data (train and validation)
        :return: None
        """
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
                        # print(f'Shape of clf output : {clf.shape}')
                        # print(f'output of clf is : {clf}')
                        # clf will be the max value between the two final logits
                        pred_labels = torch.max(self.model.sigmoid(clf), dim=1).indices  # compute the predicted labels
                        # print(f'Shape of pred_labels : {pred_labels.shape}')
                        # print(f'pred_labels is : {pred_labels}')
                        pred_labels = pred_labels.unsqueeze(-1)
                        # print(f'Shape of pred_labels after unsqueeze : {pred_labels.shape}')
                        # print(f'pred_labels after unsqueeze is : {pred_labels}')
                        # #TODO check that clf dimension is 8,2
                        clf = torch.max(clf, dim=1).values # half the dimension to give the predicted logits to the loss
                        # print(f'Shape of clf after max : {clf.shape}')
                        clf = clf.unsqueeze(-1)
                        # print(f'Shape of clf after unsqueeze : {clf.shape}')
                        # print(f'Shape of input : {inputs.shape}')
                        # print(f'Shape of labels : {labels.shape}')
                        labels = labels.unsqueeze(-1)
                        # print(f'Shape of labels after unsqueeze : {labels.shape}')
                        # print(f'Shape of output : {clf.shape}')
                        loss = self.criterion(clf, labels)  # compute the loss

                        if phase == 'train':
                            loss.backward()  # compute the gradients of the loss
                            self.optimizer.step()  # update the parameters

                    running_loss += loss.item() * inputs.size(0)  # update the loss value (multiply by the batch size)
                    # the accuracy must be computed on the activated output
                    # print(f'Predicted labels : {pred_labels}')
                    # print(f'Labels : {labels}')
                    # print(f'The sum of correct predictions {torch.sum(pred_labels == labels)}')
                    accuracy += torch.sum(pred_labels == labels)  # update the accuracy
                    # print(f'Accuracy : {accuracy:10f}')

                print(f'length of dataset is : {len(dataloaders[phase].dataset)}')
                epoch_loss = running_loss / len(dataloaders[phase].dataset)  # compute the average loss of the epoch
                epoch_accuracy = accuracy / len(dataloaders[phase].dataset)  # compute the average accuracy of the epoch

                print(f'{phase} Loss: {epoch_loss:.10f}')
                print(f'{phase} Accuracy : {epoch_accuracy:.10f}')

    def pre_train_MLM(self, dataloaders, masked_positions):
        """
        This function performs the pre-training of the umBERT model only on the MLM task
        :param dataloaders: the dataloaders used to load the data (train and validation)
        :param masked_positions: the positions of the masked elements (train and validation)
        :return: None
        """

        for epoch in range(self.n_epochs):
            for phase in ['train', 'val']:
                print(f'Epoch: {epoch + 1}/{self.n_epochs} | Phase: {phase}')
                if phase == 'train':
                    self.model.train()  # set model to training mode
                else:
                    self.model.eval()  # set model to evaluate mode

                running_loss = 0.0  # keep track of the loss
                accuracy = 0.0  # keep track of the accuracy of the MLM task

                for inputs,labels in dataloaders[phase]:  # for each batch
                    inputs = inputs.to(self.device)  # move the data to the device
                    labels = labels.to(self.device)  # move the data to the device

                    self.optimizer.zero_grad()  # zero the gradients

                    with torch.set_grad_enabled(
                            phase == 'train'):  # set the gradient computation only if in training phase
                        output = self.model.forward(inputs)  # compute the output of the model (forward pass)
                        masked_elements = custom_gather(output,masked_positions,self.device)  # select the masked elements
                        logits = self.model.ffnn(masked_elements)  # compute the logits
                        loss = self.criterion(logits, labels)  # compute the loss

                        if phase == 'train':
                            loss.backward()  # compute the gradients of the loss
                            self.optimizer.step()  # update the parameters

                    running_loss += loss.item() * inputs.size(0)  # update the loss value (multiply by the batch size)
                    accuracy += torch.sum(
                        torch.argmax(logits, dim=1) == labels)  # update the accuracy of the MLM task

                epoch_loss = running_loss / len(dataloaders[phase].dataset)  # compute the average loss of the epoch
                epoch_accuracy = accuracy / len(
                    dataloaders[phase].dataset)  # compute the average accuracy of the MLM task of the epoch

                print(f'{phase} Loss: {epoch_loss}')
                print(f'{phase} Accuracy: {epoch_accuracy}')