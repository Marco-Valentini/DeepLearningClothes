import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class umBERT2(nn.Module):
    """
    This class implements the umBERT2 architecture.
    This model assumes that the items in each outfit are ordered in the following way: shoes, tops, accessories, bottoms;
    so it keeps track of the catalogue size of each category in the outfit and uses this information to train 4 different
    ffnn (one for each category) to predict not only the masked item (MLM task) but also the non-masked ones (reconstruction task).
    """
    def __init__(self, d_model=512, num_encoders=6, num_heads=1, dropout=0, dim_feedforward=None):
        """
        the constructor of the class umBERT2
        :param d_model: the dimension of the embeddings
        :param num_encoders: the number of encoders in the encoder stack
        :param num_heads: the number of heads in the multi-head attention
        :param dropout: the dropout probability for the transformer
        :param dim_feedforward: the dimension of the feedforward layer in the encoder stack (if None, it is set to d_model*4)
        """
        super(umBERT2, self).__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 4  # according to the paper
        self.num_encoders = num_encoders  # the number of encoders in the encoder stack
        self.num_heads = num_heads  # the number of heads in the multi-head attention
        self.d_model = d_model  # the dimension of the embeddings
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout  # dropout probability
        self.encoder_stack = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,  # Set batch_first to True
        ), num_layers=num_encoders, enable_nested_tensor=(num_heads % 2 == 0))  # the encoder stack is a transformer encoder stack
        self.ffnn_shoes = nn.Linear(d_model, d_model)  # for prediction of the first item in the outfit
        self.ffnn_tops = nn.Linear(d_model, d_model)  # for prediction of the second item in the outfit
        self.ffnn_acc = nn.Linear(d_model, d_model)  # for prediction of the third item in the outfit
        self.ffnn_bottoms = nn.Linear(d_model, d_model)  # for prediction of the fourth item in the outfit
        self.softmax = F.softmax  # the output of the linear layer is fed to a softmax layer
        self.Binary_Classifier = nn.Linear(d_model, 2)  # second task: binary classification

    def forward(self, outfit):
        """
        This function takes as input an outfit and returns a logit for each item in the outfit.
        :param outfit: the outfit embeddings
        :return: a dictionary of logits for each item in the outfit
        """
        # compute the logits for each item in the outfit
        output = self.encoder_stack(outfit)  # compute the output of the encoders stack (forward pass)
        # compute the logits for each item in the outfit
        clf = self.Binary_Classifier(output[:, 0, :])  # compute the logits for the compatibility task
        recons_embeddings_shoes = self.ffnn_shoes(output[:, 1, :])  # compute the logits for the first item in the outfit
        recons_embeddings_tops = self.ffnn_tops(output[:, 2, :])  # compute the logits for the second item in the outfit
        recons_embeddings_acc = self.ffnn_acc(output[:, 3, :])  # compute the logits for the third item in the outfit
        recons_embeddings_bottoms = self.ffnn_bottoms(output[:, 4, :])  # compute the logits for the fourth item in the outfit

        return {'clf': clf,
                'shoes': recons_embeddings_shoes,
                'tops': recons_embeddings_tops,
                'accessories': recons_embeddings_acc,
                'bottoms': recons_embeddings_bottoms}
    def forward_fine_tune(self, outfit):
        """
        This function takes as input an outfit and returns a logit for each item in the outfit.
        :param outfit: the outfit embeddings
        :return: a dictionary of logits for each item in the outfit
        """
        # compute the logits for each item in the outfit
        output = self.encoder_stack(outfit)  # compute the output of the encoders stack (forward pass)
        # compute the logits for each item in the outfit
        # we don't care about clf logits in position 0
        recons_embeddings_shoes = self.ffnn_shoes(output[:, 1, :])  # compute the logits for the first item in the outfit
        recons_embeddings_tops = self.ffnn_tops(output[:, 2, :])  # compute the logits for the second item in the outfit
        recons_embeddings_acc = self.ffnn_acc(output[:, 3, :])  # compute the logits for the third item in the outfit
        recons_embeddings_bottoms = self.ffnn_bottoms(output[:, 4, :])  # compute the logits for the fourth item in the outfit

        return {'shoes': recons_embeddings_shoes,
                'tops': recons_embeddings_tops,
                'accessories': recons_embeddings_acc,
                'bottoms': recons_embeddings_bottoms}

    def predict(self, test_outfit):
        """
        This function takes as input an outfit and returns the probability distribution over the catalogue
        for each item in the outfit.
        :param test_outfit: the outfit embedding
        :return: the probability distributions over each catalogue
        """
        # compute the logits for each item in the outfit
        output = self.forward(test_outfit)  # compute the output of the encoders stack (forward pass)
        # compute the probability distributions over the catalogues
        pred_cls = self.softmax(output['clf'], dim=1)  # compute the probability distribution over the compatibility
        pred_labels_shoes = self.softmax(output['logits_shoes'], dim=1)  # compute the probability distribution over the shoes catalogue
        pred_labels_tops = self.softmax(output['logits_tops'], dim=1)  # compute the probability distribution over the tops catalogue
        pred_labels_acc = self.softmax(output['logits_acc'], dim=1)  # compute the probability distribution over the acc catalogue
        pred_labels_bottoms = self.softmax(output['logits_bottoms'], dim=1)  # compute the probability distribution over the bottoms catalogue
        # compute the indices of the predicted items
        pred_cls = torch.max(pred_cls, dim=1).indices  # compute if the outfit is compatible or not
        pred_labels_shoes = torch.max(pred_labels_shoes, dim=1).indices  # compute the indices of the predicted items
        pred_labels_tops = torch.max(pred_labels_tops, dim=1).indices  # compute the indices of the predicted items
        pred_labels_acc = torch.max(pred_labels_acc, dim=1).indices  # compute the indices of the predicted items
        pred_labels_bottoms = torch.max(pred_labels_bottoms, dim=1).indices  # compute the indices of the predicted items
        return pred_cls, pred_labels_shoes, pred_labels_tops, pred_labels_acc, pred_labels_bottoms