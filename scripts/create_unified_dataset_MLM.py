import json
import pandas as pd
# import the train/valid/test sets in the disjoint case
print('Importing the train/valid/test sets in the disjoint case...')
with open('../dataset/nondisjoint/train.json') as f:
    train_json = json.load(f)

with open('../dataset/nondisjoint/valid.json') as f:
    valid_json = json.load(f)

with open('../dataset/nondisjoint/test.json') as f:
    test_json = json.load(f)
print('Train/valid/test sets in the disjoint case loaded')
# import the train/valid/test sets in the nondisjoint case
print('Importing the train/valid/test sets in the nondisjoint case...')
with open('../dataset/disjoint/train.json') as f:
    disjoint_train_json = json.load(f)

with open('../dataset/disjoint/valid.json') as f:
    disjoint_valid_json = json.load(f)

with open('../dataset/disjoint/test.json') as f:
    disjoint_test_json = json.load(f)
print('Train/valid/test sets in the nondisjoint case loaded')
# concatenate all the lists into one big list to have all the possible outfits

all_outfits = train_json + valid_json + test_json + disjoint_train_json + disjoint_valid_json + disjoint_test_json

# remove the duplicates from the list
print("searching duplicate rows in the list")
drop_indeces = []
set_ids = []
for i in range(len(all_outfits)):
    if all_outfits[i]['set_id'] in set_ids:
        drop_indeces.append(i)
    else:
        set_ids.append(all_outfits[i]['set_id'])

# drop the duplicates
all_outfits = [i for j, i in enumerate(all_outfits) if j not in drop_indeces]
print("duplicates removed")

# remove the outfits with less than 4 items
drop_indeces = []
print("searching the outfits with less than 4 items")
for i in range(len(all_outfits)):
    if len(all_outfits[i]['items']) < 4:
        drop_indeces.append(i)

# drop the outfits with less than 4 items
all_outfits = [i for j, i in enumerate(all_outfits) if j not in drop_indeces]
print("outfits with less than 4 items removed")

# remove the items not belonging to the classes we are interested in from each outfit
# (e.g. if we are interested in shoes, tops, accessories and bottoms, we remove the items that are not in these categories)

# import the catalogue with only the items we are interested in
catalogue = pd.read_csv('../reduced_data/reduced_catalogue.csv')

# remove the items not belonging to the classes we are interested in from each outfit
print("removing the items not belonging to the classes we are interested in from each outfit")
for outfit in all_outfits:
    for item in outfit['items']:
        id = item['item_id']
        if int(id) not in list(catalogue['ID']):
            outfit['items'].remove(item)
print("items removed")

# remove the outfits with less than 4 items
drop_indeces = []
print("searching the outfits with less than 4 items")
for i in range(len(all_outfits)):
    if len(all_outfits[i]['items']) < 4:
        drop_indeces.append(i)
print("outfits with less than 4 items removed")
# remove the outfits with less than 4 items
all_outfits = [i for j, i in enumerate(all_outfits) if j not in drop_indeces]

# check that in each outfit there are 4 items of the categories we are interested in
print("checking that in each outfit there are 4 items of the categories we are interested in")
for outfit in all_outfits:
    categories = []
    new_outfit = []
    for item in outfit['items']:
        category = catalogue[catalogue['ID'] == int(item['item_id'])]['Semantic_category'].values[0]
        if category not in categories:
            categories.append(category)
            new_outfit.append(item)
    outfit['items'] = new_outfit # qui dovrebbe sovrascrivere e togliere item_id
print("checked")

# remove the outfits with less than 4 items
print("searching the outfits with less than 4 items")
drop_indeces = []
for i in range(len(all_outfits)):
    if len(all_outfits[i]['items']) < 4:
        drop_indeces.append(i)

all_outfits = [i for j, i in enumerate(all_outfits) if j not in drop_indeces]
print("outfits with less than 4 items removed")
# from all_outfits obtain just the rows
data = []
print("creating the dataframe with the outfits")
for outfit in all_outfits:
    data.append(outfit['items'])

# create the dataframe
df = pd.DataFrame(data, columns=['item1', 'item2', 'item3', 'item4'])

print("created the dataframe with the outfits")


