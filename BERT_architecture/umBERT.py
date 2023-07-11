import torch
import torch.nn as nn
import torch.nn.functional as F

from utility.custom_gather import custom_gather


class umBERT(nn.Module):
    """
    This class implements the umBERT architecture
    """
    def __init__(self, catalogue_size, d_model=512, num_encoders=6, num_heads=8, dropout=0, dim_feedforward=None):
        """
        the constructor of the class umBERT
        :param catalogue_size: the number of items in the catalogue (the number of classes)
        :param d_model: the dimension of the embeddings
        :param num_encoders: the number of encoders in the encoder stack
        :param num_heads: the number of heads in the multi-head attention
        :param dropout: the dropout probability for the transformer
        :param dim_feedforward: the dimension of the feedforward layer in the encoder stack (if None, it is set to d_model*4)
        """
        super(umBERT, self).__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 4  # according to the paper
        self.d_model = d_model  # the dimension of the embeddings
        self.num_encoders = num_encoders  # the number of encoders in the encoder stack
        self.num_heads = num_heads  # the number of heads in the multi-head attention
        self.catalogue_size = catalogue_size  # number of items in the catalogue
        self.dim_feedforward = dim_feedforward
        self.catalogue_size = catalogue_size  # number of items in the catalogue
        self.dropout = dropout  # dropout probability
        self.encoder_stack = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,  # Set batch_first to True
        ), num_layers=num_encoders, enable_nested_tensor=False)  # the encoder stack is a transformer encoder stack
        self.ffnn = nn.Linear(d_model, catalogue_size)  # the output of the transformer is fed to a linear layer
        self.softmax = F.softmax  # the output of the linear layer is fed to a softmax layer
        self.Binary_Classifier = nn.Linear(d_model, 2)  # second task: binary classification

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
        return torch.max(self.softmax(self.ffnn(self.forward(test_outfit))),dim=1).indices  # TODO fix after fine tuning


    def predict_BC(self, test_set):
        """
        This function takes as input an outfit and returns if the outfit is compatible or not
        :param test_set: the outfit embedding
        :return: the probability distribution over the catalogue
        """
        # says if the input outfit is compatible or not
        output = self.forward(test_set)  # compute the output of the model (forward pass)
        clf = self.Binary_Classifier(output[:, 0, :])  # compute the logits
        # clf will be the max value between the two final logits
        pred_labels = torch.max(self.softmax(clf, dim=1), dim=1).indices  # compute the predicted labels
        return pred_labels  # TODO check for dim

    def predict_MLM(self, test_set, masked_positions, device):
        """
        This function takes as input an outfit and returns the probability distribution over the catalogue
        :param test_set: the outfit embedding
        :return: the probability distribution over the catalogue
        """
        output = self.forward(test_set)  # compute the output of the transformer
        masked_elements = custom_gather(output, masked_positions, device)  # select the masked elements
        logits = self.ffnn(masked_elements)  # compute the logits
        logits = logits.view(-1, self.catalogue_size)
        pred_labels = torch.max(self.softmax(logits, dim=1), dim=1).indices
        return pred_labels  # says which is the masked item
        # TODO check for dim