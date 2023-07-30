import torch
import torch.nn as nn

# device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
device = torch.device("cpu")


class umBERT3(nn.Module):
    """
    This class implements the umBERT2 architecture.
    This model assumes that the items in each outfit are ordered in the following way: shoes, tops, accessories, bottoms;
    so it keeps track of the catalogue size of each category in the outfit and uses this information to train 4 different
    ffnn (one for each category) to predict not only the masked item (MLM task) but also the non-masked ones (reconstruction task).
    """

    def __init__(self, embeddings, embeddings_dict, num_encoders=6, num_heads=1, dropout=0,
                 MASK_dict=None, dim_feedforward=None):
        """
        the constructor of the class umBERT2
        :param embeddings: the catalogue of the embeddings
        :param embeddings_dict: the catalogue of the embeddings in dictionary form
        :param num_encoders: the number of encoders in the encoder stack
        :param num_heads: the number of heads in the multi-head attention
        :param dropout: the dropout probability for the transformer
        :param MASK_dict: the dictionary of the MASK tokens
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

        if MASK_dict is None:
            MASK_dict = {'shoes': torch.nn.Parameter(torch.randn((1, self.d_model)).to(device)),
                         'tops': torch.nn.Parameter(torch.randn((1, self.d_model)).to(device)),
                         'accessories': torch.nn.Parameter(torch.randn((1, self.d_model)).to(device)),
                         'bottoms': torch.nn.Parameter(torch.randn((1, self.d_model)).to(device))}
        self.MASK_dict = MASK_dict  # the MASK token
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
        self.dec_shoes = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Linear(self.d_model * 2, self.d_model)
        )
        self.dec_tops = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Linear(self.d_model * 2, self.d_model)
        )
        self.dec_acc = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Linear(self.d_model * 2, self.d_model)
        )
        self.dec_bottoms = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Linear(self.d_model * 2, self.d_model)
        )

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
                if j == 0:  # if the masked item is a shoe
                    outputs[i * 4 + j][j] = self.MASK_dict['shoes']  # mask the chosen item
                elif j == 1:  # if the masked item is a top
                    outputs[i * 4 + j][j] = self.MASK_dict['tops']  # mask the chosen item
                elif j == 2:  # if the masked item is an accessory
                    outputs[i * 4 + j][j] = self.MASK_dict['accessories']  # mask the chosen item
                elif j == 3:  # if the masked item is a bottom
                    outputs[i * 4 + j][j] = self.MASK_dict['bottoms']  # mask the chosen item
        return outputs, masked_positions, masked_items

    def forward(self, outfits):
        """
        This function takes as input an outfit and returns a logit for each item in the outfit.
        :param outfits: the outfit embeddings, a tensor of shape (batch_size, seq_len, d_model)
        :return: the output of the encoder stack, a tensor of shape (batch_size, seq_len, d_model)
        """
        return self.encoder_stack(outfits)

    def forward_reconstruction(self, inputs):
        """
        This function takes as input an outfit and returns a recostructed embedding for each item in the outfit.
        :param inputs: the outfit embeddings, a tensor of shape (batch_size, seq_len, d_model)
        :return: the reconstructed embeddings for each item in the outfits
        """
        outputs = self.forward(inputs)  # pass the outfits through the encoder stack
        rec_shoes = self.dec_shoes(outputs[:, 0, :])  # compute the reconstructed embeddings for the shoes
        rec_tops = self.dec_tops(outputs[:, 1, :])  # compute the reconstructed embeddings for the tops
        rec_acc = self.dec_acc(outputs[:, 2, :])  # compute the reconstructed embeddings for the accessories
        rec_bottoms = self.dec_bottoms(outputs[:, 3, :])  # compute the reconstructed embeddings for the bottoms
        return rec_shoes, rec_tops, rec_acc, rec_bottoms

    def forward_fill_in_the_blank(self, inputs):
        """

        :param inputs:
        :return:
        """
        outputs, masked_positions, masked_items = self.fill_in_the_blank_masking(inputs)  # mask the items
        outputs = self.forward(outputs)  # pass the outfits through the encoder stack
        rec_shoes = self.dec_shoes(outputs[:, 0, :])  # reconstruct the embeddings for the shoes
        rec_tops = self.dec_tops(outputs[:, 1, :])  # reconstruct the embeddings for the tops
        rec_acc = self.dec_acc(outputs[:, 2, :])  # reconstruct the embeddings for the accessories
        rec_bottoms = self.dec_bottoms(outputs[:, 3, :])  # reconstruct the embeddings for the bottoms
        rec_masked = []  # the reconstructed embeddings for the masked items
        for i in range(len(masked_items)):  # for each outfit
            if masked_positions[i] == 0:  # if the masked item is a shoe
                rec_masked.append(rec_shoes[i])  # add the reconstructed embedding of the masked item to the list
            elif masked_positions[i] == 1:  # if the masked item is a top
                rec_masked.append(rec_tops[i])  # add the reconstructed embedding of the masked item to the list
            elif masked_positions[i] == 2:  # if the masked item is an accessory
                rec_masked.append(rec_acc[i])  # add the reconstructed embedding of the masked item to the list
            elif masked_positions[i] == 3:  # if the masked item is a bottom
                rec_masked.append(rec_bottoms[i])  # add the reconstructed embedding of the masked item to the list
        return rec_shoes, rec_tops, rec_acc, rec_bottoms, torch.stack(rec_masked).to(device), \
            torch.stack(masked_items).to(device), masked_positions
