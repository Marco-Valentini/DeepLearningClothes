# these we use all the pre-trained and fine-tuned models to obtain a practical demonstration of the results
# load the embeddings
import numpy as np
import pandas as pd
import json

import torch
from torch.utils.data import DataLoader
import os
from utility.create_tensor_dataset_for_BC_from_dataframe import create_tensor_dataset_for_BC_from_dataframe
from utility.dataset_augmentation import create_permutations_per_outfit, mask_one_item_per_time

np.random.seed(42)  # for reproducibility

# set the working directory to the path of the file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

device = torch.device("mps" if torch.has_mps else "cpu")  # use the mps device if available
print(f'Working on device: {device}')

with open("./reduced_data/IDs_list", "r") as fp:
    IDs = json.load(fp)
print("IDs loaded")
with open('./reduced_data/embeddings.npy', 'rb') as f:
    embeddings = np.load(f)

# load the catalogue
catalogue = pd.read_csv('./reduced_data/reduced_catalogue.csv')
# load the test set
mean = torch.load('./reduced_data/mean_fine_tuning.pth')
std = torch.load('./reduced_data/std_fine_tuning.pth')

# create the MASK
print('Creating the MASK token embeddings...')
CLS = np.random.randn(1, embeddings.shape[1])
MASK = np.random.randn(1, embeddings.shape[1])
print('Done!')

test_dataframe = pd.read_csv('./reduced_data/reduced_compatibility_test.csv')
test_dataframe.drop(columns='compatibility', inplace=True)

print("Creating the test set")
test_set = create_tensor_dataset_for_BC_from_dataframe(test_dataframe, embeddings, IDs, CLS).to(device)
# augment the validation set with all the permutations of the outfits
print('Augmenting the test set with all the permutations of the outfits...')
test_set = create_permutations_per_outfit(test_set, with_CLS=False)  # to remove CLS embeddings
#  remove the CLS
test_set = test_set[:, 1:, :]
# scale the validation set using z-score (layer+batch normalization) (using the mean and std of the training set)
print('Scaling the test set using z-score...')
test_set = (test_set - mean) / std
# repeat each row of the train dataframe 24 times (all the permutations of the outfit)
test_dataframe = pd.DataFrame(np.repeat(test_dataframe.values, 24, axis=0), columns=test_dataframe.columns)
# mask one item per time
print('Masking one item per time...')
MASK = torch.Tensor(MASK).unsqueeze(0).to(device)
test_set = test_set.to(device) # move the test set to the device
masked_outfit_test, masked_indexes_test, labels_test = mask_one_item_per_time(test_set, test_dataframe, MASK,
                                                                           input_contains_CLS=True, device=device,output_in_batch_first=True)

# remove the CLS
masked_outfit_test = masked_outfit_test[:, 1:, :]
# create the test set for the fill in the blank task
test_labels = torch.Tensor(labels_test).reshape(len(labels_test), 1)
masked_positions_tensor_test = torch.Tensor(masked_indexes_test).reshape(len(masked_indexes_test), 1)
test_labels = torch.concat((test_labels, masked_positions_tensor_test), dim=1)
test_dataset = torch.utils.data.TensorDataset(masked_outfit_test, test_labels)
testloader = DataLoader(test_dataset, batch_size=8, num_workers=0, shuffle=True)
print("Test set for fill in the blank fine tuning created")

# load the fine-tuned model

# display 5 inputs

# compute the predictions

# display 5 outputs

# compute the accuracy and other metrics
