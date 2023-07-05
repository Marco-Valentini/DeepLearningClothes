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
        This function takes as input an outfit and returns the probability distribution over the catalogue
        :param test_outfit: the outfit embedding
        :return: the probability distribution over the catalogue
        """
        return self.softmax(self.ffnn(self.forward(test_outfit)))  # returns the probability distribution (in
        # the main will choose the related item)

    def pre_train_LM(self, optimizer, criterion, trainloader, labels,
                     n_epochs=500):  # TODO create together trainloader and labels

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            output = self.forward(trainloader)
            clf = self.Binary_Classifier(output) # just a linear because the criterion is the cross entropy loss
            # compute loss in masked language task
            loss = criterion(output, labels)
            # compute the gradients
            loss.backward()
            # update the parameters
            optimizer.step()
    def predict_LM(self,test_set):
        return self.softmax(self.Binary_Classifier(self.forward(test_set)))