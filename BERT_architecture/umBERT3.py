import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class umBERT3(nn.Module):
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
        super(umBERT3, self).__init__()
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
        ), num_layers=num_encoders,
            enable_nested_tensor=(num_heads % 2 == 0))  # the encoder stack is a transformer encoder stack
        self.ffnn_shoes = nn.Linear(d_model, d_model)  # for prediction of the first item in the outfit
        self.ffnn_tops = nn.Linear(d_model, d_model)  # for prediction of the second item in the outfit
        self.ffnn_acc = nn.Linear(d_model, d_model)  # for prediction of the third item in the outfit
        self.ffnn_bottoms = nn.Linear(d_model, d_model)  # for prediction of the fourth item in the outfit
        self.softmax = F.softmax  # the output of the linear layer is fed to a softmax layer
        self.Binary_Classifier = nn.Linear(d_model, 2)  # second task: binary classification

    def add_CLS(self, inputs):
        pass

    def mask_random_item(self, inputs):
        pass

    def fill_in_the_blank_masking(self, inputs):
        """

        :return: outputs: the inputs but with masked items,
        masked_positions: the position in the outfit of the masked item,
        masked_items: the original item in the outfit, a tensor
        """
        pass

    def forward(self, outfits):
        """
        This function takes as input an outfit and returns a logit for each item in the outfit.
        :param outfits: the outfit embeddings, a tensor of shape (batch_size, seq_len, d_model)
        :return: the output of the encoder stack, a tensor of shape (batch_size, seq_len, d_model)
        """
        return self.encoder_stack(outfits)

    def forward_BC(self, inputs):
        outputs = self.add_CLS(inputs)  # TODO: implement this function
        outputs = self.forward(outputs)
        logits = self.Binary_Classifier(outputs[:, 0, :])  # outputs is a tensor of shape (batch_size, seq_len, d_model)
        return logits

    def forward_MLM(self, inputs):
        outputs = self.add_CLS(inputs)  # TODO: implement this function
        outputs = self.mask_random_item(outputs)  # TODO: implement this function
        outputs = self.forward(outputs)
        logits_shoes = self.ffnn(outputs[:, 1, :])
        logits_tops = self.ffnn(outputs[:, 2, :])
        logits_acc = self.ffnn(outputs[:, 3, :])
        logits_bottoms = self.ffnn(outputs[:, 4, :])
        return logits_shoes, logits_tops, logits_acc, logits_bottoms

    def forward_fill_in_the_blank(self, inputs):
        outputs = self.add_CLS(inputs)  # TODO: implement this function
        outputs, masked_positions, masked_items = self.fill_in_the_blank_masking(outputs)  # TODO: implement this function
        outputs = self.forward(outputs)
        logits_shoes = self.ffnn(outputs[:, 1, :])
        logits_tops = self.ffnn(outputs[:, 2, :])
        logits_acc = self.ffnn(outputs[:, 3, :])
        logits_bottoms = self.ffnn(outputs[:, 4, :])
        masked_logits = []
        for i in range(len(masked_items)):
            if masked_positions[i] == 1:
                masked_logits.append(logits_shoes[i])
            elif masked_positions[i] == 2:
                masked_logits.append(logits_tops[i])
            elif masked_positions[i] == 3:
                masked_logits.append(logits_acc[i])
            elif masked_positions[i] == 4:
                masked_logits.append(logits_bottoms[i])
        return torch.Tensor(masked_logits), torch.Tensor(masked_items)



