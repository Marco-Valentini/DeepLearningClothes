import torch
from utility.custom_gather import custom_gather


class umBERT_trainer():
    def __init__(self, model, optimizer, criterion, device, n_epochs=500):
        self.model = model
        self.optimizer = optimizer  # the optimizer used to update the parameters
        self.criterion = criterion  # the loss function used to compute the loss
        self.device = device  # the device used to perform the computations (CPU or GPU)
        self.n_epochs = n_epochs  # the number of epochs used to pre-train the model

    def pre_train_BERT_like(self, dataloaders, labels_classification, labels_ids, masked_positions):
        """
        This function performs the pre-training of the umBERT model in a BERT-like fashion (MLM + classification tasks)
        :param dataloaders: the dataloaders used to load the data (train and validation)
        :param labels_classification: the labels of the classification task (train and validation)
        :param labels_ids: the labels of the MLM task (train and validation) (the labels are the ids of the items in the catalogue)
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

                for inputs in dataloaders[phase]:  # for each batch
                    inputs = inputs.to(self.device)  # move the data to the device

                    self.optimizer.zero_grad()  # zero the gradients

                    with torch.set_grad_enabled(
                            phase == 'train'):  # set the gradient computation only if in training phase
                        output = self.model.forward(inputs)  # compute the output of the model (forward pass) [batch_size, seq_len, d_model]
                        masked_elements = custom_gather(outputs=output, masked_positions=masked_positions, device=self.device)  # select the masked elements
                        logits = self.model.ffnn(masked_elements)  # compute the logits of the MLM task
                        loss_MLM = self.criterion(logits, labels_ids[phase])  # compute the loss of the MLM task
                        clf = self.model.Binary_Classifier(
                            output[0, :, :])  # compute the logits of the classification task
                        # compute the loss of the classification task
                        loss_BC = self.criterion(clf, labels_classification[phase])
                        # compute the total loss (sum of the average values of the two losses)
                        loss = loss_MLM.mean() + loss_BC.mean()

                        if phase == 'train':
                            loss.backward()  # compute the gradients of the loss
                            self.optimizer.step()  # update the parameters

                    running_loss += loss.item() * inputs.size(0)  # update the loss value (multiply by the batch size)
                    # update the accuracy of the classification task
                    accuracy_classification += torch.sum(torch.argmax(clf, dim=1) == labels_classification[phase])
                    accuracy_MLM += torch.sum(
                        torch.argmax(logits, dim=1) == labels_ids[phase])  # update the accuracy of the MLM task

                epoch_loss = running_loss / len(dataloaders[phase].dataset)  # compute the average loss of the epoch
                # compute the average accuracy of the classification task of the epoch
                epoch_accuracy_classification = accuracy_classification / len(dataloaders[phase].dataset)
                # compute the average accuracy of the MLM task of the epoch
                epoch_accuracy_MLM = accuracy_MLM / len(dataloaders[phase].dataset)

                print(f'{phase} Loss: {epoch_loss}')
                print(f'{phase} Accuracy (Classification): {epoch_accuracy_classification}')
                print(f'{phase} Accuracy (MLM): {epoch_accuracy_MLM}')

    def pre_train_BC(self, dataloaders, labels_classification):
        """
        This function performs the pre-training of the umBERT model only on the classification tasks
        :param dataloaders: the dataloaders used to load the data (train and validation)
        :param labels_classification: the labels of the classification task (train and validation)
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

                for inputs in dataloaders[phase]:  # for each batch
                    inputs = inputs.to(self.device)  # move the data to the device

                    self.optimizer.zero_grad()  # zero the gradients

                    with torch.set_grad_enabled(
                            phase == 'train'):  # set the gradient computation only if in training phase
                        output = self.model.forward(inputs)  # compute the output of the model (forward pass)
                        clf = self.model.Binary_Classifier(output[:, 0, :])  # compute the logits
                        loss = self.criterion(clf, labels_classification[phase])  # compute the loss

                        if phase == 'train':
                            loss.backward()  # compute the gradients of the loss
                            self.optimizer.step()  # update the parameters

                    running_loss += loss.item() * inputs.size(0)  # update the loss value (multiply by the batch size)
                    accuracy += torch.sum(torch.argmax(clf, dim=1) == labels_classification[phase])  # update the accuracy

                epoch_loss = running_loss / len(dataloaders[phase].dataset)  # compute the average loss of the epoch
                epoch_accuracy = accuracy / len(dataloaders[phase].dataset)  # compute the average accuracy of the epoch

                print(f'{phase} Loss: {epoch_loss}')
                print(f'{phase} Accuracy : {epoch_accuracy}')

    def pre_train_MLM(self, dataloaders, labels_ids, masked_positions):  # labels sono le posizioni dei vestiti nel catalogo, precedentemente calcolate da masking input
        """
        This function performs the pre-training of the umBERT model only on the MLM task
        :param dataloaders: the dataloaders used to load the data (train and validation)
        :param labels_ids: the labels of the MLM task (train and validation) (the labels are the ids of the items in the catalogue)
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

                for inputs in dataloaders[phase]:  # for each batch
                    inputs = inputs.to(self.device)  # move the data to the device

                    self.optimizer.zero_grad()  # zero the gradients

                    with torch.set_grad_enabled(
                            phase == 'train'):  # set the gradient computation only if in training phase
                        output = self.model.forward(inputs)  # compute the output of the model (forward pass)
                        masked_elements = torch.gather(output, 0, masked_positions[phase])  # select the masked elements
                        logits = self.model.ffnn(masked_elements)  # compute the logits
                        loss = self.criterion(logits, labels_ids[phase])  # compute the loss

                        if phase == 'train':
                            loss.backward()  # compute the gradients of the loss
                            self.optimizer.step()  # update the parameters

                    running_loss += loss.item() * inputs.size(0)  # update the loss value (multiply by the batch size)
                    accuracy += torch.sum(
                        torch.argmax(logits, dim=1) == labels_ids[phase])  # update the accuracy of the MLM task

                epoch_loss = running_loss / len(dataloaders[phase].dataset)  # compute the average loss of the epoch
                epoch_accuracy = accuracy / len(
                    dataloaders[phase].dataset)  # compute the average accuracy of the MLM task of the epoch

                print(f'{phase} Loss: {epoch_loss}')
                print(f'{phase} Accuracy: {epoch_accuracy}')