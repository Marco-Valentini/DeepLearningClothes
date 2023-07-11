# in this script take the training outfits and order them in a way such that in each outfit
# the clothes are in the order ['tops', 'bottoms', 'shoes', 'accessories']
# import the required libraries
import pandas as pd
import numpy as np
import torch
import json
import os
from utility.remove_unused_items_from_catalogue import remove_unused_items_from_catalogue

# set the working directory to the path of the file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# reorder the outfits in a dataframe such that in each outfit the clothes are in the order ['tops', 'bottoms', 'shoes', 'accessories']
def reorder_outfits(outfits):
    ordered_outfit = outfits.copy()
    for i in range(outfits.shape[0]):
        ordered_list = [None,None,None,None]
        for item in outfits.loc[i].values:
            category = catalogue['Semantic_category'].values[list(catalogue['ID'].values).index(item)]
            if category == 'tops':
                ordered_list[0] = item
            elif category == 'bottoms':
                ordered_list[1] = item
            elif category == 'shoes':
                ordered_list[2] = item
            elif category == 'accessories':
                ordered_list[3] = item
        ordered_outfit.loc[i,:] = ordered_list
    return ordered_outfit
def count_categories(column):
    # to check that all went good
    count_tops = 0
    count_bottoms = 0
    count_accessories = 0
    count_shoes = 0
    for item in column:
        category = catalogue['Semantic_category'].values[list(catalogue['ID'].values).index(int(item))]
        if category == 'tops':
            count_tops += 1
        elif category == 'bottoms':
            count_bottoms += 1
        elif category == 'shoes':
            count_shoes += 1
        elif category == 'accessories':
            count_accessories +=1
    return count_tops,count_bottoms, count_shoes, count_accessories


# load the catalogue
catalogue = pd.read_csv('../reduced_data/reduced_catalogue.csv')  # this contains ID and semantic category of each item

print("Catalogue loaded")

# create 4 reduced catalogues
catalogue_tops = catalogue[catalogue['Semantic_category'] == 'tops']
catalogue_tops = catalogue_tops.reset_index(drop=True)
catalogue_bottoms = catalogue[catalogue['Semantic_category'] == 'bottoms']
catalogue_bottoms = catalogue_bottoms.reset_index(drop=True)
catalogue_shoes = catalogue[catalogue['Semantic_category'] == 'shoes']
catalogue_shoes = catalogue_shoes.reset_index(drop=True)
catalogue_accessories = catalogue[catalogue['Semantic_category'] == 'accessories']
catalogue_accessories = catalogue_accessories.reset_index(drop=True)

# load the training validation and test sets
train_dataframe = pd.read_csv('../reduced_data/reduced_compatibility_train.csv')
valid_dataframe = pd.read_csv('../reduced_data/reduced_compatibility_valid.csv')
test_dataframe = pd.read_csv('../reduced_data/reduced_compatibility_test.csv')

print(f"In the training set there are {train_dataframe.shape[0]} outfits")
print(f"In the validation set there are {valid_dataframe.shape[0]} outfits")
print(f"In the test set there are {test_dataframe.shape[0]} outfits")


print("Training validation and test dataframes loaded")

# print the proportion in each dataset
print(f'target variable count in train_dataframe : {train_dataframe["compatibility"].value_counts()}')
print(f'target variable count in validation_dataframe : {valid_dataframe["compatibility"].value_counts()}')
print(f'target variable count in test_dataframe : {test_dataframe["compatibility"].value_counts()}')

# shuffle the DataFrame rows
train_dataframe = train_dataframe.sample(frac=1)
valid_dataframe = valid_dataframe.sample(frac=1)
test_dataframe = test_dataframe.sample(frac=1)

# take the 80% of the test set and add to the training set
idx = int(0.8 * test_dataframe.shape[0])
train_dataframe = pd.concat([train_dataframe, test_dataframe.iloc[:idx, :]], axis=0)
test_dataframe = test_dataframe.iloc[idx:, :]

print("Training dataframe size after moving 80% of test set to training set")
print(train_dataframe.shape)
print("Test dataframe size after moving 80% of test set to training set")
print(test_dataframe.shape)
print("Validation dataframe size")
print(valid_dataframe.shape)

# check that the outfits are already ordered
count_tops, count_bottoms, count_shoes, count_accessories = count_categories(train_dataframe['item_1'])
print(f"Number of tops in column 1: {count_tops}")
print(f"Number of bottoms in column 1: {count_bottoms}")
print(f"Number of shoes in column 1: {count_shoes}")
print(f"Number of accessories in column 1: {count_accessories}")
count_tops, count_bottoms, count_shoes, count_accessories = count_categories(train_dataframe['item_2'])
print(f"Number of tops in column 2: {count_tops}")
print(f"Number of bottoms in column 2: {count_bottoms}")
print(f"Number of shoes in column 2: {count_shoes}")
print(f"Number of accessories in column 2: {count_accessories}")
count_tops, count_bottoms, count_shoes, count_accessories = count_categories(train_dataframe['item_3'])
print(f"Number of tops in column 3: {count_tops}")
print(f"Number of bottoms in column 3: {count_bottoms}")
print(f"Number of shoes in column 3: {count_shoes}")
print(f"Number of accessories in column 3: {count_accessories}")
count_tops, count_bottoms, count_shoes, count_accessories = count_categories(train_dataframe['item_4'])
print(f"Number of tops in column 4: {count_tops}")
print(f"Number of bottoms in column 4: {count_bottoms}")
print(f"Number of shoes in column 4: {count_shoes}")
print(f"Number of accessories in column 4: {count_accessories}")

# save the results
train_dataframe.to_csv('../reduced_data/reduced_compatibility_train2.csv', index=False)
test_dataframe.to_csv('../reduced_data/reduced_compatibility_test2.csv', index=False)

# reduce the catalogue to just the items used in the outfits
merged_dataframe = pd.concat([train_dataframe, valid_dataframe, test_dataframe], axis=0)
# check that in merged dataframe the items are ordered
print(f"Merged dataframe size {merged_dataframe.shape[0]}")
# take in each catalogue just the elements used in the oufits
catalogue_shoes_reduced = remove_unused_items_from_catalogue(catalogue_shoes, merged_dataframe['item_1'].values)
catalogue_shoes_reduced = catalogue_shoes_reduced.reset_index(drop=True)
catalogue_tops_reduced = remove_unused_items_from_catalogue(catalogue_tops, merged_dataframe['item_2'].values)
catalogue_tops_reduced = catalogue_tops_reduced.reset_index(drop=True)
catalogue_bottoms_reduced = remove_unused_items_from_catalogue(catalogue_bottoms, merged_dataframe['item_4'].values)
catalogue_bottoms_reduced = catalogue_bottoms_reduced.reset_index(drop=True)
catalogue_accessories_reduced = remove_unused_items_from_catalogue(catalogue_accessories, merged_dataframe['item_3'].values)
catalogue_accessories_reduced = catalogue_accessories_reduced.reset_index(drop=True)

print("Catalogues sizes have been reduced")

# save the reduced catalogues
catalogue_shoes_reduced.to_csv('../reduced_data/reduced_catalogue_shoes.csv', index=False)
catalogue_tops_reduced.to_csv('../reduced_data/reduced_catalogue_tops.csv', index=False)
catalogue_bottoms_reduced.to_csv('../reduced_data/reduced_catalogue_bottoms.csv', index=False)
catalogue_accessories_reduced.to_csv('../reduced_data/reduced_catalogue_accessories.csv', index=False)