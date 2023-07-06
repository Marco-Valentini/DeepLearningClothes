import pandas as pd
#TODO import embedding token MASK
import random

import torch

catalogue = pd.read_csv('../reduced_data/reduced_catalogue.csv')
MASK = None # TODO sostituire con relativo embedding #TODO decidere questo embedding
def mask_item(outfit:torch.Tensor,outfit_labels,with_CLS=True):
    # outfit è un tensore di dimensione 5x512xbatchsize
    # supponendo che outfit labels sia del tipo [211990161,183179503,190445143,211444470] cioè 4 ID di vestiti
    CLS = outfit[1,:,:] # CLS è il primo elemento del tensore, conserviamolo
    outfit = outfit[1:,:,:] # togli CLS
    labels = []
    masked_indeces = []
    for i in range(outfit.shape[2]):  # per ogni outfit
        masked_idx = random.randrange(0,4)
        masked_indeces.append(masked_idx)
        label = list(catalogue['ID'].values).index(outfit_labels[masked_idx]) # label is the position of the masked item in the catalogue
        labels.append(label)
        if random.uniform(0,1) < 0.8:
            outfit[masked_idx,:,i] = MASK
        elif random.uniform(0,1) < 0.5:
            random_idx_outfit = random.randrange(0,outfit.shape[2])
            random_idx_item = random.randrange(0,outfit.shape[0])
            outfit[masked_idx,:,i] = outfit[random_idx_item,:,random_idx_outfit]
    if with_CLS:
        outfit = torch.cat((CLS,outfit),dim=0)
        masked_indeces = [idx+1 for idx in masked_indeces]
    return outfit,masked_indeces,labels
