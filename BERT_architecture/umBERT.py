import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random


# import mask_image as MASK #TODO: run the CNN to compute mask embedding


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

    def pre_train_BC(self, optimizer, criterion, trainloader, labels,
                     n_epochs=500):  # TODO create together trainloader and labels

        for epoch in range(n_epochs):
            print(f'Epoch of training: {epoch}, percentage {epoch/n_epochs*100}%')
            optimizer.zero_grad()
            output = self.forward(trainloader)
            clf = self.Binary_Classifier(output[0,:,:]) # just a linear because the criterion is the cross entropy loss
            # compute loss in masked language task
            loss = criterion(clf, labels)
            # compute the gradients
            loss.backward()
            # update the parameters
            optimizer.step()

    def pre_train_MLM(self, optimizer, criterion, trainloader, labels, masked_positions,
                     n_epochs=500): # labels sono le posizioni dei vestiti nel catalogo, precedentemente calcolate da masking input

        for epoch in range(n_epochs):
            print(f'Epoch of training: {epoch}, percentage {epoch/n_epochs*100}%')
            optimizer.zero_grad()
            output = self.forward(trainloader)
            masked_elements = torch.gather(output,0,masked_positions)
            logits = self.ffnn(masked_elements) # just a linear because the criterion is the cross entropy loss
            # compute loss in masked language task
            loss = criterion(logits, labels)
            # compute the gradients
            loss.backward()
            # update the parameters
            optimizer.step()


    def pre_train_BERT_like(self, optimizer, criterion, trainloader, labels_classification, labels_ids, masked_positions, n_epochs=500):
        # labels_ids sono le posizioni dei vestiti nel catalogo, precedentemente calcolate da masking input
        for epoch in range(n_epochs):
            print(f'Epoch of training: {epoch}, percentage {epoch/n_epochs*100}%')
            optimizer.zero_grad()
            output = self.forward(trainloader)
            masked_elements = torch.gather(output,0,masked_positions)
            # predict the masked embedding (MASK)
            logits = self.ffnn(masked_elements) # just a linear because the criterion is the cross entropy loss
            # compute loss in masked language task
            loss_MLM = criterion(logits, labels_ids)
            # classify the outfit (CLS)
            clf = self.Binary_Classifier(output[0,:,:]) # just a linear because the criterion is the cross entropy loss, applied on the CLS embedding
            # compute loss in binary classification task
            loss_BC = criterion(clf, labels_classification)
            # compute the gradients
            loss = loss_MLM.mean() + loss_BC.mean()
            loss.backward()
            # update the parameters
            optimizer.step()

    def predict_BC(self,test_set):
        return self.softmax(self.Binary_Classifier(self.forward(test_set))) # says if the input outfit is compatible or not

    def predict_MLM(self,test_set):
        return self.softmax(self.ffnn(self.forward(test_set))) # says which is the masket item