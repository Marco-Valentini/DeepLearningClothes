import itertools
import os

import pandas as pd
import torch

# set the working directory to the path of the file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

catalogue = pd.read_csv('../reduced_data/reduced_catalogue.csv')

def mask_one_item_per_time(outfit: torch.Tensor, outfit_dataframe: pd.DataFrame, MASK, input_contains_CLS, device, output_in_batch_first=True):
    """
    this function takes as input an outfit (a tensor of shape (5, n_outfits, emb_size)) and a dataframe containing the labels_train of the items in the outfit.
    It returns the outfit with a masked item (one item per time for each outfit) and the label of the masked item.
    :param outfit: the outfit to mask (a tensor of shape (5, n_outfits, emb_size))
    :param outfit_dataframe: Dataframe containing the labels_train of the items in the outfit
    :param MASK: the embedding token MASK (a tensor of shape (1, emb_size))
    :param input_contains_CLS: if True, the CLS token is preserved, if False assume that the CLS token is not present in the tensor
    :param batch_first: if True, the batch dimension of the output is the first dimension of the tensor (default: True)
    :return: if batch_first=False:the outfit with a masked item (a tensor of shape (5, n_outfits*sequence_length, emb_size) if with_CLS=True, (4, n_outfits, emb_size) otherwise),
    otherwise: the outfit with a masked item (a tensor of shape (n_outfits*sequence_length, 5, emb_size) if with_CLS=True, (n_outfits, 4, emb_size) otherwise),
    """
    if input_contains_CLS:
        CLS = outfit[0, :, :].to(device)  # CLS is the first element of the tensor, extract it
        outfit = outfit[1:, :, :]  # remove CLS from the tensor
    labels = []
    masked_indexes = []  # initialize the list of the indexes of the masked items
    # Create output tensor
    # Dimensions
    sequence_length = outfit.shape[0]  # 4
    n_outfits = outfit.shape[1]
    emb_size = outfit.shape[2]
    masked_outfit = torch.zeros(sequence_length, n_outfits * sequence_length, emb_size).to(device)  # (4, n_outfits*4, emb_size)
    for i in range(outfit.shape[1]):  # for each outfit
        # outfit_labels is of the form [211990161,183179503,190445143,211444470]# for each outfit
        outfit_labels = outfit_dataframe.loc[i].values  # get the labels_train of the items in the outfit
        for masked_idx in range(4):
            # save the index of the masked item
            masked_indexes.append(masked_idx)
            label = outfit_labels[masked_idx]  # label is the ID of the masked item in the catalogue
            labels.append(label)  # save the label of the masked item
            masked_outfit[masked_idx, i * sequence_length + masked_idx, :] = torch.from_numpy(MASK)  # mask the item

    if input_contains_CLS:
        # make CLS a tensor of the same shape of the masked_outfit_train tensor
        CLS = CLS.repeat(1, sequence_length, 1)
        print(f"CLS device: {CLS.device}")
        print(f"masked_outfit device: {masked_outfit.device}")
        masked_outfit = torch.cat((CLS, masked_outfit), dim=0)  # add CLS to the masked_outfit_train tensor
        # add 1 to the indexes because of CLS
        masked_indexes = [x + 1 for x in masked_indexes]

    if output_in_batch_first:
        # transpose the tensor dataset to have the batch dimension first (as required by the model)
        masked_outfit = masked_outfit.transpose(0, 1)

    return masked_outfit, masked_indexes, labels

def create_permutations_per_outfit(outfit: torch.Tensor, with_CLS=True):
    """
    This function takes as input an outfit (a tensor of shape (5, n_outfits, emb_size)) and returns all the possible permutations of the outfit.
    This help to make the model more robust to the order of the items in the outfit.
    :param outfit: the outfit to mask (a tensor of shape (5, n_outfits, emb_size))
    :param with_CLS: if True, the CLS token is preserved, otherwise it is removed from the tensor (default: True)
    :return: the tensor containing all the possible permutations of the outfit (a tensor of shape (n_outfits*24, 5, emb_size) if with_CLS=True, (n_outfits*24, 4, emb_size) otherwise)
    """
    CLS = outfit[0, :, :]  # CLS is the first element of the tensor, preserve it
    outfit = outfit[1:, :, :]  # remove CLS from the tensor
    # Dimensions
    sequence_length = outfit.shape[0]
    n_outfits = outfit.shape[1]
    emb_size = outfit.shape[2]
    # Generate all permutations
    permutations = list(itertools.permutations(range(sequence_length)))
    # Create output tensor
    permuted_outfits = torch.zeros(sequence_length, n_outfits * len(permutations), emb_size)  # (4, n_outfits*24, emb_size)
    # Fill output tensor with permutations
    for i, outfit in enumerate(outfit.transpose(0, 1)):  # for each outfit
        for j, permutation in enumerate(permutations):  # for each permutation
            permuted_outfit = outfit[list(permutation)]  # permute the outfit
            permuted_outfits[:, i * len(permutations) + j, :] = permuted_outfit  # save the permuted outfit

    if with_CLS:
        CLS = CLS.unsqueeze(0).repeat(1, len(permutations), 1)  # make CLS a tensor of the same shape of the permuted_outfits tensor
        permuted_outfits = torch.cat((CLS, permuted_outfits), dim=0)  # add CLS to the masked_outfit_train tensor

    return permuted_outfits

