import pandas as pd
import torch
import numpy as np

np.random.seed(42)

def create_tensor_dataset_from_dataframe(df_outfit:pd.DataFrame,embeddings:torch.Tensor,ids):
    CLS = np.random.rand(1, embeddings.shape[1])
    dataset = np.zeros((5,embeddings.shape[1],df_outfit.shape[0]))
    for i in range(df_outfit.shape[0]):
        for j in range(df_outfit.shape[1]+1):
            if j == 0:
                #aggiungi CLS
                dataset[j,:,i] = CLS
            else:
                ID = df_outfit.iloc[i,j-1]
                index_item = ids.index(ID)
                embedding = embeddings[index_item]
                dataset[j,:,i] = embedding
    return torch.Tensor(dataset)





