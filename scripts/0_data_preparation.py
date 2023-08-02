# import the required libraries
import json
import os
import pandas as pd
from utility.display import *

# set the working directory to the path of the file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# read the dataset information from the json
with open('../dataset/polyvore_item_metadata.json') as json_file:
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
# this dataset will be used to compute the embeddings of the single items
print(df_reduced.value_counts('Semantic_category'))
