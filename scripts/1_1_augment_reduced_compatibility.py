import os
import json
from copy import deepcopy
import pandas as pd
import numpy as np
import torch

# set the working directory to the path of the file
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# load the dataset
print('Loading the compatibility dataset...')
df = pd.read_csv('../reduced_data/reduced_compatibility.csv')
# remove all the non-compatible outfits
df = df[df['compatibility'] == 1].drop(columns=['compatibility'])
df.reset_index(drop=True, inplace=True)
with open("../reduced_data/AE_IDs_list", "r") as fp:
    IDs = json.load(fp)
# load the embeddings
with open(f'../reduced_data/AE_embeddings_128.npy', 'rb') as f:
    embeddings = np.load(f)
print('Dataset loaded')

# create the mappings and the item sets
total_mapping = {i: id for i, id in enumerate(IDs)}
total_mapping_reverse = {v: k for k, v in total_mapping.items()}

new_outfits = []

for i in range(len(df)):  # For each outfit
    outfit = df.iloc[i]
    outfit['label'] = i
    new_outfits.append(outfit)  # Add the outfit to the list of new outfits
    for j in range(4):  # For each item in the outfit
        item = outfit[j]
        item_embedding = embeddings[total_mapping_reverse[item]]  # Get the embedding of the item
        distances = torch.cdist(torch.Tensor(item_embedding).unsqueeze(0).unsqueeze(0), torch.Tensor(embeddings))  # Compute the distances between the item and all the other items
        closest_items = torch.argsort(distances).squeeze(0)[:5]  # Get the top 5 closest items
        for k in range(5):  # For each of the top 5 closest items
            new_item = total_mapping[closest_items[0][k].item()]  # Get the ID of the item
            if new_item not in outfit.values:  # If the item is not already in the outfit
                new_outfit = deepcopy(outfit)  # Copy the outfit
                new_outfit[j] = new_item  # Replace the item with the new one
                new_outfits.append(new_outfit)  # Add the new outfit to the list of new outfits

# Create a new DataFrame from the list of new outfits
new_df = pd.DataFrame(new_outfits, columns=['item_1', 'item_2', 'item_3', 'item_4', 'label'])

print('Saving the new dataset...')
new_df.to_csv('./reduced_data/reduced_compatibility_augmented.csv', index=False)

