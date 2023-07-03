import pandas as pd
#TODO import embedding token MASK
import random

catalogue = pd.read_csv('../reduced_data/reduced_catalogue.csv')
MASK = None # TODO sostituire con relativo embedding #TODO decidere questo embedding
def mask_item(outfit,outfit_labels):
    # supponendo che outfit labels sia del tipo [211990161,183179503,190445143,211444470] cioè 4 ID di vestiti
    idx = random.randrange(0,4)
    masked_pos_start = idx * 512
    masked_pos_end = masked_pos_start + 511
    masked_range = [i for i in range(masked_pos_start,masked_pos_end+1)]
    label = list(catalogue['ID'].values).index(outfit_labels[idx])
    if random.uniform(0,1) < 0.8:
        outfit[masked_range] = MASK
    elif random.uniform(0,1) < 0.5:
        random_idx = random.randrange(0,catalogue['ID'].size)
        outfit[masked_range] = catalogue[random_idx]
    return outfit,masked_range,label
