# ottenere gli item id degli outfit
import pandas as pd

# read the data from the .txt file
path = '../dataset/polyvore_outfits/disjoint/compatibility_test.txt'
with open(path) as file:
    data = file.read()
# obtain a list with one element for each line
data = data.split('\n')
# remove the double spaces
data = [line.replace('  ', ' ') for line in data]
# obtain an element for each word in each line
data = [line.split(' ') for line in data]
# convert into pandas DataFrame
labels = 'compatibility item_1 item_2 item_3 item_4 item_5 item_6 item_7 item_8 item_9 item_10 item_11 item_12 item_13 item_14'

df = pd.DataFrame(data,columns=labels.split(),index=[i for i in range(len(data))])
print(data)

import json
with open('./dataset/polyvore_outfits/disjoint/test.json') as file:
    test = json.load(file)
def retrieve_item(set_id_n):
    if set_id_n is None:
        return None
    if set_id_n in ['0','1']:
        return set_id_n
    # set id n must be a string
    set_id,item_idx = set_id_n.split('_')
    outfit = [outfit['items'] for outfit in test if outfit['set_id'] == set_id][0]
    item_id = [item['item_id'] for item in outfit if item['index'] == int(item_idx)]
    return item_id

def count_not_null(array):
    count = 0
    for el in array:
        if el is not None:
            count+=1
    return count

indeces_to_drop = []
for i in range(df.shape[0]):
    row = df.loc[i]
    count = count_not_null(row)
    if count < 5:
        indeces_to_drop.append(i)

df.drop(index=indeces_to_drop,axis=0,inplace=True)
df.reset_index()

