import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
# device = torch.device("cpu")
class umBERT3(nn.Module):
    """
    This class implements the umBERT2 architecture.
    This model assumes that the items in each outfit are ordered in the following way: shoes, tops, accessories, bottoms;
    so it keeps track of the catalogue size of each category in the outfit and uses this information to train 4 different
    ffnn (one for each category) to predict not only the masked item (MLM task) but also the non-masked ones (reconstruction task).
    """

    def __init__(self, embeddings, embeddings_dict, num_encoders=6, num_heads=1, dropout=0, CLS =None,MASK = None,dim_feedforward=None):
        """
        the constructor of the class umBERT2
        :param embeddings: the catalogue of the embeddings
        :param num_encoders: the number of encoders in the encoder stack
        :param num_heads: the number of heads in the multi-head attention
        :param dropout: the dropout probability for the transformer
        :param dim_feedforward: the dimension of the feedforward layer in the encoder stack (if None, it is set to d_model*4)
        """
        super(umBERT3, self).__init__()
        # the umBERT parameters
        if dim_feedforward is None:
            dim_feedforward = embeddings.shape[1] * 4  # according to the paper
        self.num_encoders = num_encoders  # the number of encoders in the encoder stack
        self.num_heads = num_heads  # the number of heads in the multi-head attention
        self.d_model = embeddings.shape[1]  # the dimension of the embeddings
        self.dim_feedforward = dim_feedforward
        self.catalogue_dict = embeddings_dict
        self.dropout = dropout  # dropout probability
        if CLS is None:
            # CLS = torch.nn.Parameter(torch.randn(1, self.d_model).to(device))  # the CLS token
            CLS = torch.randn(1, self.d_model).to(device)
        self.CLS = CLS # the CLS token
        if MASK is None:
            MASK = torch.nn.Parameter(torch.randn((1, self.d_model)).to(device))
        self.MASK = MASK  # the MASK token
        self.catalogue = embeddings  # the catalogue of embeddings
        # The umBERT architecture
        self.encoder_stack = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,  # Set batch_first to True
        ), num_layers=num_encoders,
            enable_nested_tensor=(num_heads % 2 == 0))  # the encoder stack is a transformer encoder stack
        self.ffnn_shoes = nn.Linear(self.d_model, self.d_model)  # for prediction of the first item in the outfit
        self.ffnn_tops = nn.Linear(self.d_model, self.d_model)  # for prediction of the second item in the outfit
        self.ffnn_acc = nn.Linear(self.d_model, self.d_model)  # for prediction of the third item in the outfit
        self.ffnn_bottoms = nn.Linear(self.d_model, self.d_model)  # for prediction of the fourth item in the outfit
        self.softmax = F.softmax  # the output of the linear layer is fed to a softmax layer
        self.Binary_Classifier = nn.Linear(self.d_model, 2)  # second task: binary classification

    def add_CLS(self, inputs):
        """
        for each outfit in inputs, this function adds the CLS token at the beginning of the outfit
        :param inputs: the outfits, a tensor of shape (batch_size, seq_len, d_model)
        :return: the outfits with the CLS token, a tensor of shape (batch_size, seq_len+1, d_model)
        """
        # create a tensor of shape (batch_size, seq_len+1, d_model)
        CLS = self.CLS.repeat(inputs.shape[0], 1, 1)
        return  torch.cat((CLS,inputs), dim=1)

    def mask_random_item(self, inputs):
        """
        This function randomly masks one item in each outfit in inputs
        for each outfit, choose randomly one item:
        - with a probability of 80%, mask the chosen item
        - with a probability of 10%, replace the chosen item with a random item in the catalogue
        - with a probability of 10%, do nothing
        :param inputs: the outfits, a tensor of shape (batch_size, seq_len, d_model)
        :return: the outfits with the masked item, a tensor of shape (batch_size, seq_len, d_model) (CLS is added later),
        """
        for i in range(inputs.shape[0]):  # for each outfit in inputs
            # choose randomly one item in the outfit
            random_index = np.random.randint(1, inputs.shape[1])  # choose randomly one item in the outfit
            prob = np.random.rand()
            if prob < 0.8:  # with a probability of 80%, mask the chosen item
                inputs[i, random_index, :] = self.MASK  # mask the chosen item
            elif prob < 0.9:  # with a probability of 10%, replace the chosen item with a random item in the catalogue
                random_item = np.random.randint(0, self.catalogue.shape[0])  # pick a random item from the catalogue
                inputs[i, random_index, :] = torch.Tensor(self.catalogue[random_item, :])  # replace the chosen item with the random item
            # else:  # with a probability of 10%, do nothing
        return inputs


    def fill_in_the_blank_masking(self, inputs):
        """
        This function, for each outfit, generate 4 outfits with one item masked in each outfit
        :param inputs: the outfits, a tensor of shape (batch_size, seq_len, d_model)
        :return: outputs: the inputs but with masked items,
        masked_positions: the position in the outfit of the masked item,
        masked_items: the original item in the outfit, a tensor
        """
        masked_positions = []  # the position in the outfit of the masked item
        masked_items = []  # the original item in the outfit, a tensor
        # create a copy of the input
        outputs = inputs.clone()
        # repeat each outfit of the input 4 times
        outputs = torch.repeat_interleave(outputs, 4, dim=0)
        for i in range(inputs.shape[0]):  # for each outfit in inputs
            for j in range(inputs.shape[1]):  # for each outfit in inputs
                # mask the item at position j
                masked_positions.append(j)  # add the position of the masked item to the list
                masked_items.append(inputs[i][j])  # add the masked item to the list
                outputs[i * 4 + j][j] = self.MASK  # mask the chosen item
        return outputs, masked_positions, masked_items

    def forward(self, outfits):
        """
        This function takes as input an outfit and returns a logit for each item in the outfit.
        :param outfits: the outfit embeddings, a tensor of shape (batch_size, seq_len, d_model)
        :return: the output of the encoder stack, a tensor of shape (batch_size, seq_len, d_model)
        """
        return self.encoder_stack(outfits)

    def forward_BC(self, inputs):
        outputs = self.add_CLS(inputs)
        outputs = self.forward(outputs)
        logits = self.Binary_Classifier(outputs[:, 0, :])  # outputs is a tensor of shape (batch_size, seq_len, d_model)
        return logits

    def forward_MLM(self, inputs):
        outputs = self.mask_random_item(inputs)  # mask the items randomly
        outputs = self.add_CLS(outputs)  # add the CLS token at the beginning of each outfit
        outputs = self.forward(outputs)  # pass the outfits through the encoder stack
        logits_shoes = self.ffnn_shoes(outputs[:, 1, :])  # compute the logits for the shoes
        logits_tops = self.ffnn_tops(outputs[:, 2, :])  # compute the logits for the tops
        logits_acc = self.ffnn_acc(outputs[:, 3, :])  # compute the logits for the accessories
        logits_bottoms = self.ffnn_bottoms(outputs[:, 4, :])  # compute the logits for the bottoms
        return logits_shoes, logits_tops, logits_acc, logits_bottoms

    def forward_reconstruction(self, inputs):
        outputs = self.add_CLS(inputs)  # add the CLS token at the beginning of each outfit
        outputs = self.forward(outputs)  # pass the outfits through the encoder stack
        logits_shoes = self.ffnn_shoes(outputs[:, 1, :])  # compute the logits for the shoes
        logits_tops = self.ffnn_tops(outputs[:, 2, :])  # compute the logits for the tops
        logits_acc = self.ffnn_acc(outputs[:, 3, :])  # compute the logits for the accessories
        logits_bottoms = self.ffnn_bottoms(outputs[:, 4, :])  # compute the logits for the bottoms
        return logits_shoes, logits_tops, logits_acc, logits_bottoms

    def forward_fill_in_the_blank(self, inputs):
        outputs, masked_positions, masked_items = self.fill_in_the_blank_masking(inputs)  # mask the items
        outputs = self.add_CLS(outputs)  # add the CLS token at the beginning of each outfit
        outputs = self.forward(outputs)  # pass the outfits through the encoder stack
        logits_shoes = self.ffnn_shoes(outputs[:, 1, :])  # compute the logits for the shoes
        logits_tops = self.ffnn_tops(outputs[:, 2, :])  # compute the logits for the tops
        logits_acc = self.ffnn_acc(outputs[:, 3, :])  # compute the logits for the accessories
        logits_bottoms = self.ffnn_bottoms(outputs[:, 4, :])  # compute the logits for the bottoms
        masked_logits = []  # the predictions for the masked items
        for i in range(len(masked_items)):  # for each outfit
            if masked_positions[i] == 0:  # if the masked item is a shoe
                masked_logits.append(logits_shoes[i])  # add the logits of the masked item to the list
            elif masked_positions[i] == 1:  # if the masked item is a top
                masked_logits.append(logits_tops[i])  # add the logits of the masked item to the list
            elif masked_positions[i] == 2:  # if the masked item is an accessory
                masked_logits.append(logits_acc[i])  # add the logits of the masked item to the list
            elif masked_positions[i] == 3:  # if the masked item is a bottom
                masked_logits.append(logits_bottoms[i])  # add the logits of the masked item to the list
        return torch.stack(masked_logits).to(device), torch.stack(masked_items).to(device), masked_positions



