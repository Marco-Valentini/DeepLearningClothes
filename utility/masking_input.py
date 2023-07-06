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
    :return: the outfit with a masked item (randomly chosen) and the label of the masked item
    """
    CLS = outfit[0, :, :]  # CLS is the first element of the tensor, preserve it
    outfit = outfit[1:, :, :]  # remove CLS from the tensor
    labels = []
    masked_indexes = []

    for i in range(outfit.shape[2]):
        # outfit_labels is of the form [211990161,183179503,190445143,211444470]# for each outfit
        outfit_labels = outfit_dataframe.loc[i].values
        masked_idx = random.randrange(0, 4)
        masked_indexes.append(masked_idx)
        label = list(catalogue['ID'].values).index(
            outfit_labels[masked_idx])  # label is the position of the masked item in the catalogue
        labels.append(label)
        if random.uniform(0, 1) < 0.8:
            outfit[masked_idx, i, :] = torch.from_numpy(MASK)
        elif random.uniform(0, 1) < 0.5:
            random_idx_outfit = random.randrange(0, outfit.shape[2])
            random_idx_item = random.randrange(0, outfit.shape[0])
            outfit[masked_idx, i, :] = outfit[random_idx_item, random_idx_outfit, :]

    if with_CLS:
        outfit = torch.cat((CLS.unsqueeze(0), outfit), dim=0)
        masked_indexes = [idx + 1 for idx in masked_indexes]
    return outfit, masked_indexes, labels
