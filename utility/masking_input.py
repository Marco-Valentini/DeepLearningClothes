import pandas as pd
import random
import torch

catalogue = pd.read_csv('../reduced_data/reduced_catalogue.csv')


def masking_input(outfit: torch.Tensor, outfit_dataframe: pd.DataFrame, MASK, with_CLS=True):
    """
    This function takes as input an outfit (a tensor of shape (4, 768, n_outfits)) and a dataframe containing the labels of the items in the outfit.
    It returns the outfit with a masked item (randomly chosen) and the label of the masked item.
    :param outfit: the outfit to mask (a tensor of shape (4, 768, n_outfits))
    :param outfit_dataframe: Dataframe containing the labels of the items in the outfit
    :param MASK: the embedding token MASK (a tensor of shape (1, 768))
    :param with_CLS: if True, the CLS token is preserved, otherwise it is removed from the tensor (default: True)
    :return: the outfit with a masked item (a tensor of shape (4, 768, n_outfits)),
    the indexes of the masked items (a tensor of shape (n_outfits, 2))
    and the labels of the masked items (a list of length n_outfits)
    """
    CLS = outfit[0, :, :]  # CLS is the first element of the tensor, preserve it
    outfit = outfit[1:, :, :]  # remove CLS from the tensor
    labels = []
    masked_indexes = []

    for i in range(outfit.shape[1]):  # for each outfit
        # outfit_labels is of the form [211990161,183179503,190445143,211444470]# for each outfit
        outfit_labels = outfit_dataframe.loc[i].values  # get the labels of the items in the outfit
        masked_idx = random.randrange(0, 4)  # choose a random item to mask
        # save the index of the masked item
        masked_indexes.append([i, masked_idx])
        label = list(catalogue['ID'].values).index(
            outfit_labels[masked_idx])  # label is the position of the masked item in the catalogue
        labels.append(label)  # save the label of the masked item
        # mask the item
        if random.uniform(0, 1) < 0.8:  # 80% of the time, replace the item with the MASK token
            outfit[masked_idx, i, :] = torch.from_numpy(MASK)
        elif random.uniform(0, 1) < 0.5:  # 10% of the time, replace the item with a random item from the catalogue
            random_idx_outfit = random.randrange(0, outfit.shape[2])
            random_idx_item = random.randrange(0, outfit.shape[0])
            outfit[masked_idx, i, :] = outfit[random_idx_item, random_idx_outfit, :]

    if with_CLS:
        outfit = torch.cat((CLS.unsqueeze(0), outfit), dim=0)  # add CLS to the tensor
        # add 1 to the indexes because of CLS
        masked_indexes = [[x[0], x[1] + 1] for x in masked_indexes]


    # transpose the tensor dataset to have the batch dimension first (as required by the model)
    outfit = outfit.transpose(0, 1)
    # max_indexes is a list of lists of length 2, where each sublist contains the index of the outfit and the index of
    # the masked item in the outfit. For example, [[0, 1], [1, 2]] means that the first outfit has the second item
    # masked, and the second outfit has the third item masked.
    # we will use masked_indexes to retrieve the embeddings of the masked items from the output of the model using
    # the gather function, so we need to convert it to a tensor of shape (n_outfits, masked_idx)
    masked_indexes = torch.LongTensor(masked_indexes)

    return outfit, masked_indexes, labels
