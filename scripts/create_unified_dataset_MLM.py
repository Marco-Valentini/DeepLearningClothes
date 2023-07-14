import json
import pandas as pd

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
    df.reset_index(drop=True, inplace=True)
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
new_list = []
for i,outfit in enumerate(all_outfits):
    new_outfit = []
    for item in outfit['items']:
        id = item['item_id']
        if int(id) in list(catalogue['ID']):
            new_outfit.append(id)
    new_list.append(new_outfit)
print("items removed")

# remove the outfits with less than 4 items
drop_indeces = []
print("searching the outfits with less than 4 items")
for i in range(len(new_list)):
    if len(new_list[i]) < 4:
        drop_indeces.append(i)
print("outfits with less than 4 items removed")
# remove the outfits with less than 4 items
new_list = [i for j, i in enumerate(new_list) if j not in drop_indeces]

# check that in each outfit there are 4 items of the categories we are interested in
print("checking that in each outfit there are 4 items of the categories we are interested in")
for i,outfit in enumerate(new_list):
    categories = []
    new_outfit = []
    for item in outfit:
        category = catalogue[catalogue['ID'] == int(item)]['Semantic_category'].values[0]
        if category not in categories:
            categories.append(category)
            new_outfit.append(item)
    new_list[i] = new_outfit
print("checked")

# remove the outfits with less than 4 items
print("searching the outfits with less than 4 items")
drop_indeces = []
for i in range(len(new_list)):
    if len(new_list[i]) < 4:
        drop_indeces.append(i)

new_list = [i for j, i in enumerate(new_list) if j not in drop_indeces]
print("outfits with less than 4 items removed")
print("creating the dataframe with the outfits")
data = new_list

# create the dataframe
df = pd.DataFrame(data, columns=['item_1', 'item_2', 'item_3', 'item_4'])

print("created the dataframe with the outfits")
# check that the outfits are already ordered
count_tops, count_bottoms, count_shoes, count_accessories = count_categories(df['item_1'])
print(f"Number of tops in column 1: {count_tops}")
print(f"Number of bottoms in column 1: {count_bottoms}")
print(f"Number of shoes in column 1: {count_shoes}")
print(f"Number of accessories in column 1: {count_accessories}")
count_tops, count_bottoms, count_shoes, count_accessories = count_categories(df['item_2'])
print(f"Number of tops in column 2: {count_tops}")
print(f"Number of bottoms in column 2: {count_bottoms}")
print(f"Number of shoes in column 2: {count_shoes}")
print(f"Number of accessories in column 2: {count_accessories}")
count_tops, count_bottoms, count_shoes, count_accessories = count_categories(df['item_3'])
print(f"Number of tops in column 3: {count_tops}")
print(f"Number of bottoms in column 3: {count_bottoms}")
print(f"Number of shoes in column 3: {count_shoes}")
print(f"Number of accessories in column 3: {count_accessories}")
count_tops, count_bottoms, count_shoes, count_accessories = count_categories(df['item_4'])
print(f"Number of tops in column 4: {count_tops}")
print(f"Number of bottoms in column 4: {count_bottoms}")
print(f"Number of shoes in column 4: {count_shoes}")
print(f"Number of accessories in column 4: {count_accessories}")

print("reordering the outfits")
df = reorder_outfits(df)

print("reordered the outfits")

print("saving the dataframe")
df.to_csv('../reduced_data/unified_dataset_MLM.csv', index=False)


