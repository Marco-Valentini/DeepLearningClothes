# import the required libraries
import json
import pandas as pd
from utility.filter_outfit import filter_outfit
from display import *

# read the dataset information from the json
with open('../dataset/polyvore_outfits/polyvore_item_metadata.json') as json_file:
    data = json.load(json_file)
    print(pd.DataFrame(data=data.values(), index=data.keys()).head())  # shows first 5 rows of the dataset

# create a new dictionary with for each image its identifier and its category
data_new = {}
for el in data.keys():
    data_new[el] = data[el]['semantic_category']

# obtain a dataframe with images identifiers and related class
df = pd.DataFrame(data=data_new.items(), columns=['ID', 'Semantic_category'])

# explore distributions on data
print(df.describe())
print(df.value_counts())

# extract the rows of tops/bottoms/shoes/accessories
index_list = [x for (x, y) in enumerate(df['Semantic_category']) if y in ['tops', 'bottoms', 'shoes', 'accessories']]
df_reduced = df.iloc[index_list]

# show some results
display_items(df_reduced)

# save the results
catalogue_path = '../reduced_data/reduced_catalogue.csv'
df_reduced.to_csv(catalogue_path, index=False)

# check if the extraction has worked
print(df_reduced.value_counts(
    'Semantic_category'))  # this dataset will be used to compute the embeddings of the single items

with open('../dataset/polyvore_outfits/disjoint/train.json') as json_file:
    train_outfit = json.load(json_file)
    print(pd.DataFrame(data=train_outfit,columns=['items','set_id']).head()) # shows first 5 rows of the dataset

# extract from the outfit dataset only the ones which respect the given conditions
# train_outfit_cleaned = [filter_outfit(outfit, df_reduced) for outfit in train_outfit if
#                         filter_outfit(outfit, df_reduced)]

# show some results
# display_outfits(train_outfit_cleaned)

# repeat the same for validation and test sets
with open('../dataset/polyvore_outfits/disjoint/valid.json') as json_file:
    validation_outfit = json.load(json_file)
    print(pd.DataFrame(data=validation_outfit,columns=['items','set_id']).head()) # shows first 5 rows of the dataset

# extract from the outfit dataset only the ones which respect the given conditions
validation_outfit_cleaned = [filter_outfit(outfit, df_reduced) for outfit in validation_outfit if
                        filter_outfit(outfit, df_reduced)]

# repeat the same for validation and test sets
with open('../dataset/polyvore_outfits/disjoint/test.json') as json_file:
    test_outfit = json.load(json_file)
    print(pd.DataFrame(data=test_outfit,columns=['items','set_id']).head()) # shows first 5 rows of the dataset

# extract from the outfit dataset only the ones which respect the given conditions
test_outfit_cleaned = [filter_outfit(outfit, df_reduced) for outfit in test_outfit if
                        filter_outfit(outfit, df_reduced)]

# save the results
train_path = "../reduced_data/training_outfit.json"
validation_path = "../reduced_data/validation_outfit.json"
test_path = "../reduced_data/test_outfit.json"

# with open(train_path,"w") as json_file:
#     json.dump(train_outfit_cleaned,json_file)

with open(validation_path,"w") as json_file:
    json.dump(validation_outfit_cleaned,json_file)

with open(test_path,"w") as json_file:
    json.dump(test_outfit_cleaned,json_file)