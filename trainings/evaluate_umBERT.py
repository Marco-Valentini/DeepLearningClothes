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

with open('../reduced_data/embeddings.npy', 'rb') as f:
    embeddings = np.load(f)

print("Embeddings loaded")

# create MASK and CLS token embeddings as random tensors with the same shape of the embeddings
print('Creating the MASK and CLS token embeddings...')
CLS = np.random.randn(1, embeddings.shape[1])
MASK = np.random.randn(1, embeddings.shape[1])

model = umBERT(catalogue_size=catalogue['ID'].size, d_model=embeddings.shape[1], num_encoders=6, num_heads=8,
                dropout=0.2, dim_feedforward=None)
model.load_state_dict(torch.load('../models/umBERT_pretrained_jointly.pth'))
model.to(device)

# import the validation set
test_dataframe = pd.read_csv('../reduced_data/reduced_compatibility_test.csv')
compatibility_test = test_dataframe['compatibility'].values
test_dataframe.drop(columns='compatibility', inplace=True)

# create the tensor dataset for the test set (which contains the CLS embedding)
print('Creating the tensor dataset for the test set...')
test_set = create_tensor_dataset_for_BC_from_dataframe(test_dataframe, embeddings, IDs, CLS)
# mask the input (using the MASK embedding)
print('Masking the input...')
test_set, masked_indexes_test, masked_labels_test = masking_input(test_set, test_dataframe, MASK)
# labels for BC are the same as the compatibility labels, labels for MLM are the masked labels
BC_test_labels = torch.Tensor(compatibility_test).unsqueeze(1)
MLM_test_labels = torch.Tensor(masked_labels_test).unsqueeze(1)
masked_test_positions = torch.Tensor(masked_indexes_test).unsqueeze(1)
# concatenate the labels
test_labels = torch.concat((BC_test_labels, MLM_test_labels, masked_test_positions), dim=1)
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
