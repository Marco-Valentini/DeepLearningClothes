import pandas as pd
import torch
import numpy as np

np.random.seed(42)


def create_tensor_dataset_for_BC_from_dataframe(df_outfit: pd.DataFrame, embeddings: torch.Tensor, ids, CLS):
    """
    This function takes as input a dataframe containing the labels of the items in the outfit, the embeddings of the items and the ids of the items.
    It returns a tensor of shape (seq_len, n_outfits, embedding_size) containing the embeddings of the items in the outfit and the CLS token.
    :param df_outfit:  dataframe containing the labels of the items in the outfit
    :param embeddings:  embeddings of the items in the outfit (a tensor of shape (n_items, embedding_size))
    :param ids:  ids of the items in the outfit (a list of length n_items)
    :param CLS: the embedding token CLS (a tensor of shape (1, embedding_size))
    :return: a tensor of shape (seq_len, n_outfits, embedding_size) containing the embeddings of the items in the outfit and the CLS token
    """
    dataset = np.zeros((5, df_outfit.shape[0], embeddings.shape[1]))
    for i in range(df_outfit.shape[0]):  # for each outfit
        for j in range(df_outfit.shape[1] + 1):  # for each item in the outfit
            if j == 0:
                # aggiungi CLS
                dataset[j, i, :] = CLS
            else:
                ID = df_outfit.iloc[i, j - 1]
                index_item = ids.index(ID)
                embedding = embeddings[index_item]
                dataset[j, i, :] = embedding
                # do not transpose to not create conflicts with masking input
    return torch.Tensor(dataset)
