import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
from torchvision.transforms import transforms

from image_to_embedding import image_to_embedding

# use GPU if available
device = torch.device("mps" if torch.has_mps else "cpu")

# first step: obtain the embeddings of the dataset using the fine-tuned model finetuned_fashion_resnet18.pth

# load the model finetuned_fashion_resnet18.pth
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features  # get the number of input features for the last fully connected layer
model.fc = nn.Linear(num_ftrs, 4)  # modify the last fully connected layer to have the desired number of classes
model.load_state_dict(torch.load('./models/finetuned_fashion_resnet18.pth'))  # load the weights of the model finetuned_fashion_resnet18.pth
model.eval()  # set the model to evaluation mode
model.to(device)  # set the model to run on the device

# load the dataset (just for now, we will use the test dataset)
data_transform = transforms.Compose([  # define the transformations to be applied to the images
    transforms.Resize((224, 224)),  # resize the image to 224x224
    transforms.ToTensor()  # convert the image to a tensor
])
data_dir = './dataset_cnn_fine_tuning'
test_dir = data_dir + '/test'
image_dataset = ImageFolder(test_dir, transform=data_transform)  # create the dataset

dataloaders = DataLoader(image_dataset, batch_size=32, shuffle=False, num_workers=0)  # create the dataloader

# get the embeddings of the dataset, the labels and the ids
embeddings, labels, paths = image_to_embedding(dataloaders, model, device)
# show the embeddings
print(embeddings)
print(embeddings.shape)
print(paths)

# second step: define the bert-like model and train it on the embeddings
