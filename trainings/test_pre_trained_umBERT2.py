import os
import torch
import random
import json

from sklearn.model_selection import train_test_split
from BERT_architecture.umBERT2 import umBERT2
from torch.utils.data import DataLoader
from utility.umBERT2_trainer import umBERT2_trainer
from constants import *

# set the seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
SEED = 42

dim_embeddings = 128

# set the working directory to the path of the file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# use GPU if available
device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
print('Device used: ', device)

# read the dataset
df = pd.read_csv('../reduced_data/reduced_compatibility.csv')

# load the IDs of the images
with open("../reduced_data/IDs_list", "r") as fp:
    IDs = json.load(fp)
# load the embeddings
with open(f'../reduced_data/embeddings_{str(dim_embeddings)}.npy', 'rb') as f:
    embeddings = np.load(f)

# generate the special embeddings CLS and MASK
create_CLS_modality = 'random'  # 'random', 'task_based'
create_MASK_modality = 'zeros'  # 'random', 'task_based', 'zeros'
print(f'Creating the special embeddings CLS and MASK using {create_CLS_modality} modality for CLS and '
      f'{create_MASK_modality} modality for MASK...')

if create_CLS_modality == 'random':
    CLS, _ = generate_special_embeddings_randomly(dim_embeddings)
elif create_CLS_modality == 'task_based':
    CLS = task_based_cls_embedding(dim_embeddings, df, embeddings, IDs)
else:
    raise ValueError('The modality for the creation of the CLS embedding is not valid.')

if create_MASK_modality == 'random':
    _, MASK = generate_special_embeddings_randomly(dim_embeddings)
elif create_MASK_modality == 'task_based':
    MASK = task_based_mask_embedding(embeddings)
elif create_MASK_modality == 'zeros':
    MASK = initialize_mask_embedding_zeros(dim_embeddings)
else:
    raise ValueError('The modality for the creation of the MASK embedding is not valid.')

# split the dataset in train, valid and test set (80%, 10%, 10%) in a stratified way on the compatibility column
compatibility = df['compatibility'].values
df = df.drop(columns=['compatibility'])
df_train, df_test, compatibility_train, compatibility_test = train_test_split(df, compatibility, test_size=0.2,
                                                                              stratify=compatibility,
                                                                              random_state=42,
                                                                              shuffle=True)
df_valid, df_test, compatibility_valid, compatibility_test = train_test_split(df_test, compatibility_test,
                                                                              test_size=0.5,
                                                                              stratify=compatibility_test,
                                                                              random_state=42,
                                                                              shuffle=True)
# reset the index of the dataframes
df_train = df_train.reset_index(drop=True)
df_valid = df_valid.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# load mean and std of the trainings set
mean = torch.load('../reduced_data/mean.pth')
std = torch.load('../reduced_data/std.pth')

# load the pre-trained model
checkpoint = torch.load(f'../models/umBERT2_pre_trained_{dim_embeddings}.pth')

model = umBERT2(d_model=checkpoint['d_model'],
                num_encoders=checkpoint['num_encoders'],
                num_heads=checkpoint['num_heads'],
                dropout=checkpoint['dropout'],
                dim_feedforward=checkpoint['dim_feedforward'])
# load the model weights
model.load_state_dict(checkpoint['model_state_dict'])

# prepare the first 10 sample to feed the model with
embeddings_samples = np.zeros((5, df_test.shape[0], embeddings.shape[1]))

# shuffle the outfits in the df_train
df_test = df_test.sample(frac=1).reset_index(drop=True)
for i in range(embeddings_samples.shape[1]):  # for each outfit
    for j in range(embeddings_samples.shape[0]):  # for each item in the outfit
        if j == 0:
            # aggiungi CLS
            embeddings_samples[j, i, :] = CLS
        else:
            ID = df_test.iloc[i, j - 1]
            index_item = IDs.index(ID)
            embedding = embeddings[index_item]
            embeddings_samples[j, i, :] = embedding
embeddings_samples = torch.Tensor(embeddings_samples).to(device)
embeddings_samples = embeddings_samples.transpose(0, 1)  # shape: (10, 5, 128)

labels_CLF = torch.tensor(compatibility_test).to(device)  # compatibility labels of the first 10 outfits
labels_CLF_one_hot = torch.nn.functional.one_hot(labels_CLF, num_classes=2).to(device)

IDs_train = torch.tensor(df_test.values).to(device)  # IDs of the first 10 outfits
labels_shoes = torch.tensor(df_test.iloc[:, 0].values).to(device)  # IDs of the shoes of the first 10 outfits
labels_tops = torch.tensor(df_test.iloc[:, 1].values).to(device)  # IDs of the tops of the first 10 outfits
labels_acc = torch.tensor(df_test.iloc[:, 2].values).to(device)  # IDs of the accessories of the first 10 outfits
labels_bottoms = torch.tensor(df_test.iloc[:, 3].values).to(device)  # IDs of the bottoms of the first 10 outfits

trainer = umBERT2_trainer(model=model,  optimizer=None, criterion=None, device=device, n_epochs=None)

# feed the model with the first 10 samples
model.to(device)
model.eval()
with torch.no_grad():
    dict_outputs = model(embeddings_samples)

    dict_inputs = {
        'clf': labels_CLF_one_hot,
        'shoes': embeddings_samples[:, 1, :],
        'tops': embeddings_samples[:, 2, :],
        'accessories': embeddings_samples[:, 3, :],
        'bottoms': embeddings_samples[:, 4, :]
    }

    # update the accuracy of the classification task
    pred_labels_CLF = torch.max((model.softmax(dict_outputs['clf'], dim=1)), dim=1).indices
    pred_labels_shoes = trainer.find_closest_embeddings(dict_outputs['shoes'])
    pred_labels_tops = trainer.find_closest_embeddings(dict_outputs['tops'])
    pred_labels_acc = trainer.find_closest_embeddings(dict_outputs['accessories'])
    pred_labels_bottoms = trainer.find_closest_embeddings(dict_outputs['bottoms'])

    # update the accuracy of the classification task
    accuracy_CLF = torch.sum(pred_labels_CLF == labels_CLF) / len(labels_CLF)
    # update the accuracy of the MLM task
    accuracy_shoes = torch.sum(pred_labels_shoes == labels_shoes) / len(labels_shoes)
    accuracy_tops = torch.sum(pred_labels_tops == labels_tops) / len(labels_tops)
    accuracy_acc = torch.sum(pred_labels_acc == labels_acc) / len(labels_acc)
    accuracy_bottoms = torch.sum(pred_labels_bottoms == labels_bottoms) / len(labels_bottoms)

    print(f'Accuracy CLF: {accuracy_CLF}')
    print(f'Accuracy shoes: {accuracy_shoes}')
    print(f'Accuracy tops: {accuracy_tops}')
    print(f'Accuracy acc: {accuracy_acc}')
    print(f'Accuracy bottoms: {accuracy_bottoms}')
