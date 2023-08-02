# Starting from txt files about compatible outfits (compatibility train, valid and test), obtain a dataframe containing
# outfit composed of 4 items belonging to the categories ['tops','bottoms','accessories','shoes'] and save them into a
# csv file containing the item IDs
# import required libraries
import os

import pandas as pd
import numpy as np
import json

# set the working directory to the path of the file
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# define the required functions
def retrieve_item(set_id_n):
    """
    Given a set_id_n, returns the item_id of the corresponding item in the catalogue
    :param set_id_n: set_id_n of the item
    :return: item_id of the item
    """
    if set_id_n is None:  # if it is None (due to previous padding reasons)
        return None
    if set_id_n in ['0', '1']:  # if it is a compatibility value
        return set_id_n
    set_id, item_idx = set_id_n.split('_')  # split the set_id_n into set_id and item_idx
    outfit = [outfit['items'] for outfit in set_json if outfit['set_id'] == set_id][0]  # retrieve the outfit
    item_id = [item['item_id'] for item in outfit if item['index'] == int(item_idx)][0]  # retrieve the item_id
    return item_id


def count_not_null(array):
    """
    Given an array, returns the number of not null elements
    :param array: array to be analyzed
    :return: number of not null elements
    """
    count = 0
    for el in array:
        if el is not None:
            count += 1
    return count


def is_in_categories(item_id):
    """
    Given an item_id, returns True if the item belongs to the categories ['tops','bottoms','accessories','shoes']
    :param item_id: item_id to be analyzed
    :return: True if the item belongs to the categories ['tops','bottoms','accessories','shoes'], False otherwise
    """
    if item_id is None:
        # if it is None (due to previous padding reasons)
        return None
    if item_id in ['0', '1']:  # if it is a compatibility value
        return item_id
    if int(item_id) not in catalogue['ID'].values:  # if the item_id is not in the catalogue
        return None
    else:
        return item_id


def remove_and_compact(df):
    """
    Given a dataframe, removes the rows containing None values and compact the dataframe.
    :param df: dataframe to be compacted.
    :return: compacted dataframe.
    """
    df_tmp = df.drop(columns='compatibility')  # remove the compatibility column
    df_new = pd.DataFrame(np.zeros((df_tmp.shape[0], 4)), columns=df_tmp.columns[:4])  # create a new dataframe
    compatibility = df['compatibility'].values  # retrieve the compatibility values
    for i in range(df_tmp.shape[0]):  # for each row
        row = df_tmp.loc[i]  # retrieve the row
        filtered_row = [el for el in row.values if el is not None]  # remove the None values
        # if there are more than 4 elements remove the elements belonging to the same category
        if len(filtered_row) >= 4:
            clothes = []
            categories = []
            for item in filtered_row:
                idx = list(catalogue['ID']).index(int(item))  # retrieve the index of the item in the catalogue
                category = catalogue['Semantic_category'].values[idx]  # retrieve the category of the item
                if category not in categories:  # if the category is not already in the list
                    categories.append(category)
                    clothes.append(item)
            if len(clothes) < 4:  # if after the process there are less than 4 elements
                filtered_row = [None, None, None, None]  # substitute with None values (will be dropped later)
            else:
                filtered_row = clothes
        else:
            filtered_row = [None, None, None, None]  # substitute with None values (will be dropped later)
        df_new.loc[i] = filtered_row  # insert the filtered row in the new dataframe
    # insert the compatibility column
    df_new.insert(0, column='compatibility', value=compatibility, allow_duplicates=True)
    df_new = df_new.dropna(axis=0)  # drop the rows containing None values
    df_new.reset_index(inplace=True, drop=True)  # reset the index
    return df_new


def count_categories(column):
    """
    Given a column of item_ids, returns the number of items belonging to each category
    :param column: column of item_ids
    :return: number of items belonging to each category
    """
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
            count_accessories += 1
    return count_tops, count_bottoms, count_shoes, count_accessories


# import the catalogue ID-Category
catalogue = pd.read_csv('../reduced_data/reduced_catalogue.csv')
for fold in ['nondisjoint', 'disjoint']:
    for data_set in ['train', 'test', 'valid']:  # for each data set apply the following operations
        print(f'Processing {data_set} of {fold}...')
        # read the data from the <train, test, valid>.txt file
        path = f'../dataset/{fold}/compatibility_{data_set}.txt'
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

        with open(f'../dataset/{fold}/{data_set}.json') as file:
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
        print('Substituting the set_id_index with the item id which corresponds to...')
        df = df.applymap(retrieve_item)

        # check that there are no None value in item 1,2,3,4 (all the outfits have at least 4 items)
        print('Checking that there are no None value in item 1,2,3,4 (all the outfits have at least 4 items)...')
        if df['item_1'].isna().sum() == 0 and \
                df['item_2'].isna().sum() == 0 and \
                df['item_3'].isna().sum() == 0 and \
                df['item_4'].isna().sum() == 0:
            print('OK - no outfit with less than 4 items in the DataFrame')

        # set to None all the items not belonging to the allowed categories
        print('Setting to None all the items not belonging to the allowed categories...')
        df = df.applymap(is_in_categories)  # apply this 'is_in_category filter'
        df.reset_index(inplace=True, drop=True)

        # remove the no more valid outfits
        print('Removing the no more valid outfits...')
        indexes_to_drop = []
        for i in range(df.shape[0]):
            row = df.loc[i]
            count = count_not_null(row)
            if count < 5:
                indexes_to_drop.append(i)

        df.drop(index=indexes_to_drop, axis=0, inplace=True)
        df.reset_index(inplace=True, drop=True)

        # remove the None and compact
        print('Removing the None and compacting the DataFrame...')
        df = remove_and_compact(df)
        # check if the classes are balanced or not
        print('Checking if the classes are balanced or not...')
        value_counts = df['compatibility'].value_counts()
        print(f"Proportion of negative/positive samples {value_counts}")
        # check the number of items for each column
        print(f"Phase  {data_set} data {fold}")
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

        # save the reduced dataset
        destination_path = f'../reduced_data/{fold}_reduced_compatibility_{data_set}.csv'
        df.to_csv(destination_path, index=False)

# join the disjoint and non-disjoint dataset in a single one
df_train_disjoint = pd.read_csv('../reduced_data/disjoint_reduced_compatibility_train.csv')
df_train_nondisjoint = pd.read_csv('../reduced_data/nondisjoint_reduced_compatibility_train.csv')
df_valid_disjoint = pd.read_csv('../reduced_data/disjoint_reduced_compatibility_valid.csv')
df_valid_nondisjoint = pd.read_csv('../reduced_data/nondisjoint_reduced_compatibility_valid.csv')
df_test_disjoint = pd.read_csv('../reduced_data/disjoint_reduced_compatibility_test.csv')
df_test_nondisjoint = pd.read_csv('../reduced_data/nondisjoint_reduced_compatibility_test.csv')
# concatenate all the dataframes
df = pd.concat([df_train_disjoint, df_train_nondisjoint, df_valid_disjoint, df_valid_nondisjoint, df_test_disjoint,
                df_test_nondisjoint], axis=0)
# remove the duplicates
df.drop_duplicates(inplace=True)
# reset the index
df.reset_index(inplace=True, drop=True)
# save the final datasets
df.to_csv('../reduced_data/reduced_compatibility.csv', index=False)
# delete the old datasets
os.remove('../reduced_data/disjoint_reduced_compatibility_train.csv')
os.remove('../reduced_data/nondisjoint_reduced_compatibility_train.csv')
os.remove('../reduced_data/disjoint_reduced_compatibility_valid.csv')
os.remove('../reduced_data/nondisjoint_reduced_compatibility_valid.csv')
os.remove('../reduced_data/disjoint_reduced_compatibility_test.csv')
os.remove('../reduced_data/nondisjoint_reduced_compatibility_test.csv')
