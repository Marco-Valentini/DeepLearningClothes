# obtain the embeddings of the dataset using the fine-tuned model finetuned_fashion_resnet18_128.pth
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from utility.custom_image_dataset import CustomImageDataset
from utility.resnet_image_to_embedding import resnet_image_to_embedding
from torchvision.models import resnet18
from models.resnet18_modified import Resnet18Modified
from torchvision.transforms import transforms

# use GPU if available
device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
print("Device used: ", device)
# define the size of the embeddings
dim_embeddings = 128
# load the checkpoint
checkpoint = torch.load(f'../checkpoints/finetuned_fashion_resnet18_{dim_embeddings}.pth')
# load the model finetuned_fashion_resnet18
print(f"Loading the model finetuned_fashion_resnet18_{dim_embeddings}.pth")
fashion_resnet18 = resnet18()
fashion_resnet18 = Resnet18Modified(fashion_resnet18, dim_embeddings=checkpoint['dim_embeddings'], num_classes=4)

# load the weights of the model finetuned_fashion_resnet18
fashion_resnet18.load_state_dict(checkpoint['state_dict'])

fashion_resnet18.eval()  # set the model to evaluation mode
fashion_resnet18.to(device)  # set the model to run on the device

print("Model loaded")

print("Loading the image dataset")

# load the dataset (just for now, we will use the test dataset)
data_transform = transforms.Compose([  # define the transformations to be applied to the images
    transforms.Resize(256),  # resize the image to 256x256
    transforms.ToTensor(),  # convert the image to a tensor
])

data_dir = '../dataset_catalogue'  # define the directory of the dataset
image_dataset = CustomImageDataset(root_dir=data_dir, data_transform=data_transform)  # create the dataset

dataloaders = DataLoader(image_dataset, batch_size=256, num_workers=0)  # create the dataloader
print("Dataset loaded")

print("Computing the embeddings of the dataset")
# get the embeddings of the dataset, the labels and the ids
embeddings, labels, IDs = resnet_image_to_embedding(dataloaders, fashion_resnet18, device)
print("Embeddings computed")

print("Saving the embeddings IDs")
with open("../reduced_data/resnet_IDs_list", "w") as fp:
    json.dump(IDs, fp)
print("IDs saved")

with open(f'../reduced_data/embeddings_{dim_embeddings}.npy', 'wb') as f:
    np.save(f, embeddings)
print("Embeddings saved")
