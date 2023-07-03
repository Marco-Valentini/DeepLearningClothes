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
            dim_feedforward = d_model * 4
        self.dim_feedforward = dim_feedforward
        self.catalogue_size = catalogue_size
        self.dropout = dropout
        self.encoder_stack = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation="gelu"
        ), num_layers=num_encoders)
        self.ffnn = nn.Linear(dim_feedforward, catalogue_size)
        self.softmax = F.softmax

    def forward(self, outfit):
        return self.ffnn(self.encoder_stack(outfit)) # returns the logits

    def predict(self,test_outfit):
        return self.softmax(self.forward(test_outfit)) # returns the probability distribution (in
                                                        # the main will choose the related item)


    def pre_train_LM(self, optimizer, criterion, trainloader,labels, n_epochs=500): # TODO create together trainloader and labels
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            output = self.forward(trainloader)
            # compute loss in masked language task
            loss = criterion(output,labels)
            # compute the gradients
            loss.backward()
            # update the parameters
            optimizer.step()

