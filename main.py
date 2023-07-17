# these we use all the pre-trained and fine-tuned models to obtain a practical demonstration of the results
import numpy as np
import pandas as pd
import json
import torch
from torch.utils.data import DataLoader
import os
from BERT_architecture.umBERT import umBERT
from utility.create_tensor_dataset_for_BC_from_dataframe import create_tensor_dataset_for_BC_from_dataframe
from utility.dataset_augmentation import mask_one_item_per_time
from constants import generate_special_embeddings_randomly  # CLS is the embedding of the CLS token, MASK is the embedding of the MASK token
from utility.display import display_outfits, display_predictions
from utility.umBERT_trainer import umBERT_evaluator


# set the working directory to the path of the file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

device = torch.device("mps" if torch.has_mps else "cpu")  # use the mps device if available

# get the special embeddings
CLS, MASK = generate_special_embeddings_randomly()

print(f'Working on device: {device}')

with open("./reduced_data/IDs_list", "r") as fp:
    IDs = json.load(fp)
print("IDs loaded")
with open('reduced_data/embeddings_512_old.npy', 'rb') as f:
    embeddings = np.load(f)

# load the catalogue
catalogue = pd.read_csv('./reduced_data/reduced_catalogue.csv')
# load the test set
mean = torch.load('./reduced_data/mean_fine_tuning.pth')
std = torch.load('./reduced_data/std_fine_tuning.pth')

test_dataframe = pd.read_csv('./reduced_data/reduced_compatibility_test.csv')
# remove the outfits with compatibility 0 (not compatible) (we want only compatible outfits)
test_dataframe = test_dataframe[test_dataframe['compatibility'] == 1]
compatibility_test = test_dataframe['compatibility'].values
test_dataframe.drop(columns='compatibility', inplace=True)

print("Creating the test set")
test_set = create_tensor_dataset_for_BC_from_dataframe(test_dataframe, embeddings, IDs, CLS)
#  remove the CLS
test_set = test_set[1:, :, :]
# scale the validation set using z-score (layer+batch normalization) (using the mean and std of the training set)
print('Scaling the test set using z-score...')
test_set = (test_set - mean) / std
# mask one item per time
print('Masking one item per time...')
masked_outfit_test, masked_indexes_test, labels_test = mask_one_item_per_time(test_set,
                                                                              test_dataframe,
                                                                              MASK,
                                                                              input_contains_CLS=False,
                                                                              device=device,
                                                                              output_in_batch_first=True)

# create the test set for the fill in the blank task
compatibility_test = np.repeat(compatibility_test, 4)  # repeat the compatibility 4 times (one for each masked item)
compatibility_test = torch.Tensor(compatibility_test).unsqueeze(1)  # not useful for the fill in the blank task but it is needed for avoid errors with the evaluator
labels_test_tensor = torch.Tensor(labels_test).unsqueeze(1)
masked_positions_tensor_test = torch.Tensor(masked_indexes_test).unsqueeze(1)
labels_test_tensor = torch.concat((compatibility_test, labels_test_tensor, masked_positions_tensor_test), dim=1)
test_dataset = torch.utils.data.TensorDataset(masked_outfit_test, labels_test_tensor)
testloader = DataLoader(test_dataset, batch_size=8, num_workers=0, shuffle=True)
print("Test set for fill in the blank fine tuning created")

# load the fine-tuned model
file_path = './models/umBERT_pretrained_BERT_like.pth'
checkpoint = torch.load(file_path)

model = umBERT(catalogue_size=checkpoint['catalogue_size'], d_model=checkpoint['d_model'],
               num_encoders=checkpoint['num_encoders'], num_heads=checkpoint['num_heads'],
               dropout=checkpoint['dropout'], dim_feedforward=checkpoint['dim_feedforward'])
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

# display 4 inputs
display_outfits(test_dataframe, 4)

# compute the predictions
predictions = model.predict_MLM(masked_outfit_test, masked_positions_tensor_test, device)

# display 4 outputs
# I'm taking:
# the first prediction for the first outfit
# the second prediction for the second outfit
# the third prediction for the third outfit
# the fourth prediction for the fourth outfit
first_4_outfits_predicted = [0, 5, 10, 15]
items_predicted = predictions[first_4_outfits_predicted]
# print the positions of the first 4 masked items
for i in range(len(first_4_outfits_predicted)):
    print(f'The positions of the masked item in the outfit {i+1} is: {masked_indexes_test[first_4_outfits_predicted[i]]}')
display_predictions(items_predicted, catalogue)

# compute the accuracy and other metrics
evaluator = umBERT_evaluator(model, device)
accuracy = evaluator.test_MLM(testloader)
print(f'Accuracy on the test set: {accuracy}')
