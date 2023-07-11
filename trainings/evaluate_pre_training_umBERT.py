# load the model from the checkpoint
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from BERT_architecture.umBERT import umBERT
from utility.create_tensor_dataset_for_BC_from_dataframe import create_tensor_dataset_for_BC_from_dataframe
from utility.masking_input import masking_input
from utility.umBERT_trainer import umBERT_evaluator
from utility.dataset_augmentation import mask_one_item_per_time
from constants import MASK, CLS  # import the MASK and CLS tokens


# set the working directory to the path of the file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

np.random.seed(42)  # for reproducibility

# use GPU if available
device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
print('Using device:', device)

catalogue = pd.read_csv('../reduced_data/reduced_catalogue.csv')  # load the catalogue

# first step: load the embeddings of the dataset obtained by the fine-tuned model finetuned_fashion_resnet18.pth
with open("../reduced_data/IDs_list", "r") as fp:
    IDs = json.load(fp)
print("IDs loaded")

with open('../reduced_data/embeddings_512.npy', 'rb') as f:
    embeddings = np.load(f)

print("Embeddings loaded")

file_path = '../models/umBERT_pretrained_BERT_like.pth'
checkpoint = torch.load(file_path)

model = umBERT(catalogue_size=checkpoint['catalogue_size'], d_model=checkpoint['d_model'], num_encoders=checkpoint['num_encoders'], num_heads=checkpoint['num_heads'],
                dropout=checkpoint['dropout'], dim_feedforward=checkpoint['dim_feedforward'])
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

# import the validation set
test_dataframe = pd.read_csv('../reduced_data/reduced_compatibility_test.csv')
compatibility_test = test_dataframe['compatibility'].values
test_dataframe.drop(columns='compatibility', inplace=True)

# create the tensor dataset for the test set (which contains the CLS embedding)
print('Creating the tensor dataset for the test set...')
test_set = create_tensor_dataset_for_BC_from_dataframe(test_dataframe, embeddings, IDs, CLS)
# remove the CLS
test_set = test_set[1:, :, :]
mean = torch.load('../reduced_data/mean.pth') # this is computed during pre training on the train set
std = torch.load('../reduced_data/std.pth')
test_set = (test_set - mean) / std

# mask the input (using the MASK embedding)
print('Masking the input...')
test_set, masked_indexes_test, masked_labels_test = mask_one_item_per_time(test_set,
                                                                           test_dataframe,
                                                                           MASK,
                                                                           input_contains_CLS=False,
                                                                           device=device,
                                                                           output_in_batch_first=True)

# labels for BC are the same as the compatibility labels, labels for MLM are the masked labels
compatibility_test = compatibility_test.repeat(4)
BC_test_labels = torch.Tensor(compatibility_test).unsqueeze(1)
MLM_test_labels = torch.Tensor(masked_labels_test).unsqueeze(1)
masked_test_positions = torch.Tensor(masked_indexes_test).unsqueeze(1)
# concatenate the labels
test_labels = torch.concat((BC_test_labels,MLM_test_labels, masked_test_positions), dim=1)
# create a Tensor Dataset
test_set = torch.utils.data.TensorDataset(test_set, test_labels)

# create the dataloader for the test set
print('Creating the dataloader for the test set...')
testloader = DataLoader(test_set, batch_size=32, shuffle=True, num_workers=0)
print('Done!')

# evaluate the model on the test set
print('Start evaluating the model on the test set')
evaluator = umBERT_evaluator(model, device)
accuracy_MLM, accuracy_BC = evaluator.evaluate_BERT_like(testloader)
print(f'Accuracy on MLM task: {accuracy_MLM}')
print(f'Accuracy on BC task: {accuracy_BC}')
