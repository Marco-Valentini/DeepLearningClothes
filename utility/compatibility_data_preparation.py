# Starting from txt files about compatible outfits (compatibility train, valid and test), obtain a dataframe containing
# outfit composed of 4 items belonging to the categories ['tops','bottoms','accessories','shoes'] and save them into a
# csv file containing the item IDs
# import required libraries
import pandas as pd
import numpy as np
import json


# define the required functions
def retrieve_item(set_id_n):
    if set_id_n is None:
        return None
    if set_id_n in ['0', '1']:
        return set_id_n
    # set id n must be a string
    set_id, item_idx = set_id_n.split('_')
    outfit = [outfit['items'] for outfit in set_json if outfit['set_id'] == set_id][0]
    item_id = [item['item_id'] for item in outfit if item['index'] == int(item_idx)][0]
    return item_id


def count_not_null(array):
    count = 0
    for el in array:
        if el is not None:
            count += 1
    return count


def is_in_categories(item_id):
    # TODO scrivi la doc
    if item_id is None:
        # if it is None (due to previous padding reasons)
        return None
    if item_id in ['0', '1']:
        return item_id
    if int(item_id) not in catalogue['ID'].values:
        return None
    else:
        return item_id


def remove_and_compact(df):
    df_tmp = df.drop(columns='compatibility')
    df_new = pd.DataFrame(np.zeros((df_tmp.shape[0], 4)), columns=df_tmp.columns[:4])
    compatibility = df['compatibility'].values
    for i in range(df_tmp.shape[0]):
        row = df_tmp.loc[i]
        filtered_row = [el for el in row.values if el is not None]
        if len(filtered_row) > 4:
            clothes = []
            categories = []
            for item in filtered_row:
                idx = list(catalogue['ID']).index(int(item))
                category = catalogue['Semantic_category'].values[idx]
                if category not in categories:
                    categories.append(category)
                    clothes.append(item)
            if len(clothes) < 4:
                filtered_row = [None, None, None, None]
            else:
                filtered_row = clothes
        df_new.loc[i] = filtered_row
    df_new.insert(0, column='compatibility', value=compatibility, allow_duplicates=True)
    df_new = df_new.dropna(axis=0)
    df_new.reset_index(inplace=True, drop=True)
    return df_new

# import the catalogue ID-Category
catalogue = pd.read_csv('../reduced_data/reduced_catalogue.csv')

for data_set in ['train', 'test', 'valid']:
    # read the data from the .txt file
    path = f'../dataset/disjoint/compatibility_{data_set}.txt'
    with open(path) as file:
        data = file.read()
    # obtain a list with one element for each line
    data = data.split('\n')
    # remove the double spaces
    data = [line.replace('  ', ' ') for line in data]
    # obtain an element for each word in each line
    data = [line.split(' ') for line in data]
    # convert into pandas DataFrame
    max_len = max([len(line) for line in data])  # max number of items in an outfit
    labels = ['compatibility'] + ['item_' + str(i) for i in range(1, max_len)]

    df = pd.DataFrame(data, columns=labels, index=[i for i in range(len(data))])

    with open(f'../dataset/disjoint/{data_set}.json') as file:
        set_json = json.load(file)

    indexes_to_drop = []
    for i in range(df.shape[0]):
        row = df.loc[i]
        count = count_not_null(row)
        if count < 5:
            indexes_to_drop.append(i)

    df.drop(index=indexes_to_drop, axis=0, inplace=True)
    df.reset_index(inplace=True, drop=True)

    # substitute the set_id_index with the item id which corresponds to
    df = df.applymap(retrieve_item)

    # check that there are no None value in item 1,2,3,4 (all the outfits have at least 4 items)
    if df['item_1'].isna().sum() == 0 and df['item_2'].isna().sum() == 0 and df['item_3'].isna().sum() == 0 and df[
        'item_4'].isna().sum() == 0:
        print('OK - no outfit with less than 4 items in the DataFrame')

    # set to None all the items not belonging to the allowed categories
    df = df.applymap(is_in_categories)  # apply this 'is_in_category filter'
    df.reset_index(inplace=True, drop=True)

    # remove the no more valid outfits
    indexes_to_drop = []
    for i in range(df.shape[0]):
        row = df.loc[i]
        count = count_not_null(row)
        if count < 5:
            indexes_to_drop.append(i)

    df.drop(index=indexes_to_drop, axis=0, inplace=True)
    df.reset_index(inplace=True, drop=True)

    # remove the None and compact
    df = remove_and_compact(df)
    # check if the classes are balanced or not
    value_counts = df['compatibility']
    print(f"Proportion of negative/positive samples {value_counts}")

    destination_path = f'../reduced_data/reduced_compatibility_{data_set}.csv'
    df.to_csv(destination_path, index=False)
