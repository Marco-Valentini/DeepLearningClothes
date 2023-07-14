# in this main the umBERT2 model is trained with the best hyperparameters found by the Bayesian optimization.
from constants import get_special_embeddings
from utility.umBERT2_trainer import umBERT2_trainer
from BERT_architecture.umBERT2 import umBERT2
from utility.get_category_labels import get_category_labels
from utility.create_tensor_dataset_for_BC_from_dataframe import create_tensor_dataset_for_BC_from_dataframe
from utility.dataset_augmentation import mask_one_item_per_time
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, AdamW
from lion_pytorch import Lion
import torch
import numpy as np
import pandas as pd
import random
import os
import json

# set the working directory to the path of the file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# set the seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)

# import the MASK and CLS tokens
dim_embeddings = 64
CLS, MASK = get_special_embeddings(dim_embeddings)

# use GPU if available
device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
print('Device used: ', device)

# load the catalogue for each category (shoes, tops, accessories, bottoms)
catalogues = {}
for category in ['shoes', 'tops', 'accessories', 'bottoms']:
    catalogues[category] = pd.read_csv(f'../reduced_data/reduced_catalogue_{category}.csv')
    print(f'Catalogue {category} loaded') # each catalogue is organized in form ID-category

# first step: load the embeddings of the dataset obtained from fine-tuned model finetuned_fashion_resnet18
with open(f'../reduced_data/IDs_list', 'r') as fp:
    IDs = json.load(fp)
print(f'IDs {category} loaded')

with open(f'../reduced_data/embeddings_{dim_embeddings}.npy', 'rb') as f:
    embeddings = np.load(f)
print(f'Embeddings {category} loaded')

# create a dict of catalogue sizes for each category
catalogue_sizes = {}
for category in ['shoes', 'tops', 'accessories', 'bottoms']:
    catalogue_sizes[category] = catalogues[category]['ID'].size

# import the training set
train_dataframe = pd.read_csv('../reduced_data/reduced_compatibility_train2.csv') # this is an augmented version with samples added from the test set
train_dataframe.drop(columns=['compatibility'], inplace=True) # we don't need the compatibility column anymore

# create the labels for each item in the catalogue with respect to their position in their catalogue
# (e.g. the first item of the catalogue has label 0, the second item has label 1, etc.)
category_labels_train = get_category_labels(train_dataframe, catalogues) # TODO chiedi questo from each reduced catalogue, retrieve the label of the orginal catalogue

# create the tensor dataset for the training set (which contains the CLS embedding, we will remove later)
print('Creating the tensor dataset for the training set...')
training_set = create_tensor_dataset_for_BC_from_dataframe(train_dataframe, embeddings, IDs, CLS)
print('Scaling the training set using z-score...')
training_set = training_set[1:, :, :]  # remove CLS from the tensor
mean = training_set.mean(dim=0).mean(dim=0)
std = training_set.std(dim=0).std(dim=0)
training_set = (training_set - mean) / std
print('Training set scaled')
# mask the input (using the MASK embedding)
print('Masking the input...')
training_set, _, _ = mask_one_item_per_time(training_set, train_dataframe, MASK,input_contains_CLS=False, device=device, output_in_batch_first=True)
# after the masking the first dimension is the batch size thanks to the batch first flag

shoes_trainings_labels = torch.Tensor(category_labels_train['shoes']).unsqueeze(1)
tops_trainings_labels = torch.Tensor(category_labels_train['tops']).unsqueeze(1)
accessories_trainings_labels = torch.Tensor(category_labels_train['accessories']).unsqueeze(1)
bottoms_trainings_labels = torch.Tensor(category_labels_train['bottoms']).unsqueeze(1)

# concatenate the labels to the tensor
training_labels = torch.cat((shoes_trainings_labels,
                             tops_trainings_labels,
                             accessories_trainings_labels,
                             bottoms_trainings_labels), dim=1)
# TODO replicate the labels

# create a Tensor Dataset
training_set = torch.utils.data.TensorDataset(training_set, training_labels)

# create the dataloader for the training set
print('Creating the dataloader for the training set...')
trainloader = DataLoader(training_set, batch_size=32, shuffle=True, num_workers=0)

# import the validation set
validation_dataframe = pd.read_csv('../reduced_data/reduced_compatibility_validation.csv')
validation_dataframe.drop(columns=['compatibility'], inplace=True) # we don't need the compatibility column anymore

# create the labels for each item in the catalogue with respect to their position in their catalogue
# (e.g. the first item of the catalogue has label 0, the second item has label 1, etc.)
category_labels_validation = get_category_labels(validation_dataframe, catalogues)

# create the tensor dataset for the validation set (which contains the CLS embedding, we will remove later)
print('Creating the tensor dataset for the validation set...')
validation_set = create_tensor_dataset_for_BC_from_dataframe(validation_dataframe, embeddings, IDs, CLS)
print('Scaling the validation set using z-score...')
validation_set = validation_set[1:, :, :]  # remove CLS from the tensor
validation_set = (validation_set - mean) / std
print('Validation set scaled')
# mask the input (using the MASK embedding)
print('Masking the input...')
validation_set, _, _ = mask_one_item_per_time(validation_set, validation_dataframe, MASK,input_contains_CLS=False, device=device, output_in_batch_first=True)
# after the masking the first dimension is the batch size thanks to the batch first flag

shoes_validation_labels = torch.Tensor(category_labels_validation['shoes']).unsqueeze(1)
tops_validation_labels = torch.Tensor(category_labels_validation['tops']).unsqueeze(1)
accessories_validation_labels = torch.Tensor(category_labels_validation['accessories']).unsqueeze(1)
bottoms_validation_labels = torch.Tensor(category_labels_validation['bottoms']).unsqueeze(1)

# concatenate the labels to the tensor
validation_labels = torch.cat((shoes_validation_labels,tops_validation_labels,accessories_validation_labels,bottoms_validation_labels), dim=1)

# create a Tensor Dataset
validation_set = torch.utils.data.TensorDataset(validation_set, validation_labels)

# create the dataloader for the validation set
print('Creating the dataloader for the validation set...')
validationloader = DataLoader(validation_set, batch_size=32, shuffle=True, num_workers=0)

# create the dictionary containing the dataloaders for the training and validation set
dataloaders = {'train': trainloader, 'val': validationloader}

# define the model and load the weights
# load the checkpoint
checkpoint = torch.load(f'../models/umBERT2_pretrained.pth')
# load the model architecture
model = umBERT2(catalogue_sizes=catalogue_sizes, d_model=checkpoint['d_model'],num_encoders=checkpoint['num_encoders'], num_heads=checkpoint['num_heads'], dropout=checkpoint['dropout'])
# load the weights
model.load_state_dict(checkpoint['model_state_dict'])

criterion = CrossEntropyLoss()

optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.01,betas=(0.9, 0.999), eps=1e-08)

# fine-tune end-to-end the model
trainer = umBERT2_trainer(model=model,device=device,n_epochs=100,criterion=criterion,optimizer = optimizer)
best_acc = trainer.fine_tuning(dataloaders)

print(f'Best accuracy: {best_acc}')
# evaluate the model on the test set
test_dataframe = pd.read_csv('../reduced_data/reduced_compatibility_test.csv')
test_dataframe.drop(columns=['compatibility'], inplace=True) # we don't need the compatibility column anymore

# create the labels for each item in the catalogue with respect to their position in their catalogue
# (e.g. the first item of the catalogue has label 0, the second item has label 1, etc.)
category_labels_test = get_category_labels(test_dataframe, catalogues)

# create the tensor dataset for the test set (which contains the CLS embedding, we will remove later)
print('Creating the tensor dataset for the test set...')
test_set = create_tensor_dataset_for_BC_from_dataframe(test_dataframe, embeddings, IDs, CLS)
print('Scaling the test set using z-score...')
test_set = test_set[1:, :, :]  # remove CLS from the tensor
test_set = (test_set - mean) / std
print('Test set scaled')
# mask the input (using the MASK embedding)
print('Masking the input...')
test_set, _, _ = mask_one_item_per_time(test_set, test_dataframe, MASK,input_contains_CLS=False, device=device, output_in_batch_first=True)
# after the masking the first dimension is the batch size thanks to the batch first flag

shoes_test_labels = torch.Tensor(category_labels_test['shoes']).unsqueeze(1)
tops_test_labels = torch.Tensor(category_labels_test['tops']).unsqueeze(1)
accessories_test_labels = torch.Tensor(category_labels_test['accessories']).unsqueeze(1)
bottoms_test_labels = torch.Tensor(category_labels_test['bottoms']).unsqueeze(1)

# concatenate the labels to the tensor
test_labels = torch.cat((shoes_test_labels,tops_test_labels,accessories_test_labels,bottoms_test_labels), dim=1)

# create a Tensor Dataset
test_set = torch.utils.data.TensorDataset(test_set, test_labels)

# create the dataloader for the test set
print('Creating the dataloader for the test set...')
testloader = DataLoader(test_set, batch_size=32, shuffle=True, num_workers=0)

# evaluate the model on the test set
print('Evaluating the model on the test set...')
test_acc = trainer.evaluate_fine_tuning(testloader)