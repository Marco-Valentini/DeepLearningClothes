import torch
import torch.nn as nn
import torch.nn.functional as F


class umBERT(nn.Module):
    def __init__(self, catalogue_size, d_model=512, num_encoders=6, num_heads=8, dropout=0, dim_feedforward=None):
        super(umBERT, self).__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 4  # according to the paper
        self.dim_feedforward = dim_feedforward
        self.catalogue_size = catalogue_size  # number of items in the catalogue
        self.dropout = dropout  # dropout probability
        self.encoder_stack = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation="gelu"
        ), num_layers=num_encoders)  # the encoder stack is a transformer encoder stack
        self.ffnn = nn.Linear(d_model, catalogue_size)  # the output of the transformer is fed to a linear layer
        self.softmax = F.softmax  # the output of the linear layer is fed to a softmax layer
        # first task of pre-training #TODO check this later
        self.Binary_Classifier = nn.Linear(d_model, 2)

    def forward(self, outfit):
        """
        This function takes as input an outfit and returns the logits
        :param outfit: the outfit embedding
        :return: the logits
        """
        return self.encoder_stack(outfit)

    def predict(self, test_outfit):
        """
        This function takes as input an outfit and returns the probability distribution over the catalogue (after fine-tuning)
        :param test_outfit: the outfit embedding
        :return: the probability distribution over the catalogue
        """
        return self.softmax(self.ffnn(self.forward(test_outfit)))  # returns the probability distribution (in
        # the main will choose the related item)

    def pre_train_BC(self, optimizer, criterion, dataloaders, labels_classification, device,
                     n_epochs=500):
        for epoch in range(n_epochs):
            for phase in ['train', 'val']:
                print(f'Epoch: {epoch + 1}/{n_epochs} | Phase: {phase}')
                if phase == 'train':
                    self.train()  # set model to training mode
                else:
                    self.eval()  # set model to evaluate mode

                running_loss = 0.0  # keep track of the loss
                accuracy = 0.0  # keep track of the accuracy of the classification task

                for inputs in dataloaders[phase]:  # for each batch
                    inputs = inputs.to(device)  # move the data to the device

                    optimizer.zero_grad()  # zero the gradients

                    with torch.set_grad_enabled(
                            phase == 'train'):  # set the gradient computation only if in training phase
                        output = self.forward(inputs)  # compute the output of the model (forward pass)
                        clf = self.Binary_Classifier(output[0, :, :])  # compute the logits
                        loss = criterion(clf, labels_classification[phase])  # compute the loss

                        if phase == 'train':
                            loss.backward()  # compute the gradients of the loss
                            optimizer.step()  # update the parameters

                    running_loss += loss.item() * inputs.size(0)  # update the loss value (multiply by the batch size)
                    accuracy += torch.sum(
                        torch.argmax(clf, dim=1) == labels_classification[phase])  # update the accuracy

                epoch_loss = running_loss / len(dataloaders[phase].dataset)  # compute the average loss of the epoch
                epoch_accuracy = accuracy / len(dataloaders[phase].dataset)  # compute the average accuracy of the epoch

                print(f'{phase} Loss: {epoch_loss}')
                print(f'{phase} Accuracy : {epoch_accuracy}')

    def pre_train_MLM(self, optimizer, criterion, dataloaders, labels_ids, masked_positions, device,
                      n_epochs=500):  # labels sono le posizioni dei vestiti nel catalogo, precedentemente calcolate da masking input
        """
        This function performs the pre-training of the umBERT model only on the MLM task
        :param optimizer: the optimizer used to update the parameters
        :param criterion: the loss function used to compute the loss
        :param dataloaders: the dataloaders used to load the data (train and validation)
        :param labels_ids: the labels of the MLM task (train and validation) (the labels are the ids of the items in the catalogue)
        :param masked_positions: the positions of the masked elements (train and validation)
        :param device: the device used to perform the computations (CPU or GPU)
        :param n_epochs: the number of epochs of the pre-training
        :return:
        """

        for epoch in range(n_epochs):
            for phase in ['train', 'val']:
                print(f'Epoch: {epoch + 1}/{n_epochs} | Phase: {phase}')
                if phase == 'train':
                    self.train()  # set model to training mode
                else:
                    self.eval()  # set model to evaluate mode

                running_loss = 0.0  # keep track of the loss
                accuracy = 0.0  # keep track of the accuracy of the MLM task

                for inputs in dataloaders[phase]:  # for each batch
                    inputs = inputs.to(device)  # move the data to the device

                    optimizer.zero_grad()  # zero the gradients

                    with torch.set_grad_enabled(
                            phase == 'train'):  # set the gradient computation only if in training phase
                        output = self.forward(inputs)  # compute the output of the model (forward pass)
                        masked_elements = torch.gather(output, 0, masked_positions[phase])  # select the masked elements
                        logits = self.ffnn(masked_elements)  # compute the logits
                        loss = criterion(logits, labels_ids[phase])  # compute the loss

                        if phase == 'train':
                            loss.backward()  # compute the gradients of the loss
                            optimizer.step()  # update the parameters

                    running_loss += loss.item() * inputs.size(0)  # update the loss value (multiply by the batch size)
                    accuracy += torch.sum(
                        torch.argmax(logits, dim=1) == labels_ids[phase])  # update the accuracy of the MLM task

                epoch_loss = running_loss / len(dataloaders[phase].dataset)  # compute the average loss of the epoch
                epoch_accuracy = accuracy / len(
                    dataloaders[phase].dataset)  # compute the average accuracy of the MLM task of the epoch

                print(f'{phase} Loss: {epoch_loss}')
                print(f'{phase} Accuracy: {epoch_accuracy}')

    def pre_train_BERT_like(self, optimizer, criterion, dataloaders, labels_classification, labels_ids,
                            masked_positions, device, n_epochs=500):
        """
        This function performs the pre-training of the umBERT model in a BERT-like fashion (MLM + classification tasks)
        :param optimizer: the optimizer used to update the parameters
        :param criterion: the loss function used to compute the loss
        :param dataloaders: the dataloaders used to load the data (train and validation)
        :param labels_classification: the labels of the classification task (train and validation)
        :param labels_ids: the labels of the MLM task (train and validation) (the labels are the ids of the items in the catalogue)
        :param masked_positions: the positions of the masked elements in the outfit(train and validation)
        :param device: the device used to perform the computations (CPU or GPU)
        :param n_epochs: the number of epochs used to pre-train the model
        :return:
        """
        for epoch in range(n_epochs):
            for phase in ['train', 'val']:
                print(f'Epoch: {epoch + 1}/{n_epochs} | Phase: {phase}')
                if phase == 'train':
                    self.train()  # set model to training mode
                else:
                    self.eval()  # set model to evaluate mode

                running_loss = 0.0  # keep track of the loss
                accuracy_classification = 0.0  # keep track of the accuracy of the classification task
                accuracy_MLM = 0.0  # keep track of the accuracy of the MLM task

                for inputs in dataloaders[phase]:  # for each batch
                    inputs = inputs.to(device)  # move the data to the device

                    optimizer.zero_grad()  # zero the gradients

                    with torch.set_grad_enabled(
                            phase == 'train'):  # set the gradient computation only if in training phase
                        output = self.forward(inputs)  # compute the output of the model (forward pass)
                        masked_elements = torch.gather(output, 0, masked_positions[phase])  # select the masked elements
                        logits = self.ffnn(masked_elements)  # compute the logits of the MLM task
                        loss_MLM = criterion(logits, labels_ids[phase])  # compute the loss of the MLM task
                        clf = self.Binary_Classifier(output[0, :, :])  # compute the logits of the classification task
                        loss_BC = criterion(clf,
                                            labels_classification[phase])  # compute the loss of the classification task
                        loss = loss_MLM.mean() + loss_BC.mean()  # compute the total loss (sum of the average values of the two losses)

                        if phase == 'train':
                            loss.backward()  # compute the gradients of the loss
                            optimizer.step()  # update the parameters

                    running_loss += loss.item() * inputs.size(0)  # update the loss value (multiply by the batch size)
                    accuracy_classification += torch.sum(torch.argmax(clf, dim=1) == labels_classification[
                        phase])  # update the accuracy of the classification task
                    accuracy_MLM += torch.sum(
                        torch.argmax(logits, dim=1) == labels_ids[phase])  # update the accuracy of the MLM task

                epoch_loss = running_loss / len(dataloaders[phase].dataset)  # compute the average loss of the epoch
                epoch_accuracy_classification = accuracy_classification / len(
                    dataloaders[phase].dataset)  # compute the average accuracy of the classification task of the epoch
                epoch_accuracy_MLM = accuracy_MLM / len(
                    dataloaders[phase].dataset)  # compute the average accuracy of the MLM task of the epoch

                print(f'{phase} Loss: {epoch_loss}')
                print(f'{phase} Accuracy (Classification): {epoch_accuracy_classification}')
                print(f'{phase} Accuracy (MLM): {epoch_accuracy_MLM}')

    def predict_BC(self, test_set):
        return self.softmax(
            self.Binary_Classifier(self.forward(test_set)))  # says if the input outfit is compatible or not

    def predict_MLM(self, test_set):
        return self.softmax(self.ffnn(self.forward(test_set)))  # says which is the masket item
