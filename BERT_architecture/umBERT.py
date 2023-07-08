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
            activation="gelu",
            batch_first=True  # Set batch_first to True
        ), num_layers=num_encoders)  # the encoder stack is a transformer encoder stack
        self.ffnn = nn.Linear(d_model, catalogue_size)  # the output of the transformer is fed to a linear layer
        self.softmax = F.softmax  # the output of the linear layer is fed to a softmax layer
        self.Binary_Classifier = nn.Linear(d_model, 2)  # second task: binary classification
        self.sigmoid = F.sigmoid  # the output of the binary classifier layer is fed to a sigmoid layer

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

    def predict_BC(self, test_set):
        return self.softmax(
            self.Binary_Classifier(self.forward(test_set)))  # says if the input outfit is compatible or not

    def predict_MLM(self, test_set):
        return self.softmax(self.ffnn(self.forward(test_set)))  # says which is the masket item
