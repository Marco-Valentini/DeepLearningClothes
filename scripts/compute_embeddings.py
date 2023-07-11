# obtain the embeddings of the dataset using the fine-tuned model finetuned_fashion_resnet18.pth
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utility.custom_image_dataset import CustomImageDataset
from utility.image_to_embedding import image_to_embedding
from torchvision.models import resnet18
from torchvision.transforms import transforms

# use GPU if available
device = torch.device("mps" if torch.has_mps else "cpu")
print("Device used: ", device)

# load the model finetuned_fashion_resnet18.pth
print("Loading the model finetuned_fashion_resnet18.pth")
fashion_resnet18 = resnet18()
num_ftrs = fashion_resnet18.fc.in_features  # get the number of input features for the last fully connected layer
fashion_resnet18.fc = nn.Linear(num_ftrs,
                                4)  # modify the last fully connected layer to have the desired number of classes
fashion_resnet18.load_state_dict(torch.load(
    '../models/finetuned_fashion_resnet18.pth'))  # load the weights of the model finetuned_fashion_resnet18.pth
fashion_resnet18.eval()  # set the model to evaluation mode
fashion_resnet18.to(device)  # set the model to run on the device

print("Model loaded")

print("Loading the image dataset")

# load the dataset (just for now, we will use the test dataset)
data_transform = transforms.Compose([  # define the transformations to be applied to the images
    transforms.Resize(256),  # resize the image to 256x256
    transforms.CenterCrop(224),  # crop the image to 224x224
    transforms.ToTensor(),  # convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalize the image to the ImageNet mean and standard deviation
])

data_dir = '../dataset_catalogue'  # define the directory of the dataset
image_dataset = CustomImageDataset(root_dir=data_dir, data_transform=data_transform)  # create the dataset

dataloaders = DataLoader(image_dataset, batch_size=8, shuffle=False, num_workers=0)  # create the dataloader
print("Dataset loaded")

print("Computing the embeddings of the dataset")
# get the embeddings of the dataset, the labels and the ids
embeddings, labels, IDs = image_to_embedding(dataloaders, fashion_resnet18, device)
print("Embeddings computed")

print("Saving the embeddings IDs")
with open("../reduced_data/IDs_list", "w") as fp:
    json.dump(IDs, fp)
print("IDs saved")

with open('../reduced_data/embeddings_512.npy', 'wb') as f:
    np.save(f, embeddings)
print("Embeddings saved")
