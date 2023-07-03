# Description: this file contains the function to convert the images to embeddings using the pretrained model and the
# dataloader of the dataset and the device to run the model on.
import numpy as np
import torch
from torch import nn


class Identity(nn.Module):
    """
    This class is used to replace the last fully connected layer of the model with an identity layer
    """
    def __init__(self):
        """
        Constructor of the class
        """
        super(Identity, self).__init__()

    def forward(self, x):
        """
        Forward pass of the class (does nothing)
        :param x: the input.
        :return: the input.
        """
        return x

def image_to_embedding(dataloader, model, device):
    """
    given a dataloader, a model and a device, return the embeddings of the dataset and the labels
    :param dataloader: the dataloader of the dataset
    :param model: cnn resnet18 model pretrained and finetuned with fashion dataset
    :param device: the device to run the model on
    :return: the embeddings of the dataset the labels of the dataset and the image paths of the dataset
    """
    # get the embeddings
    embeddings = []
    # keep track of the images paths
    paths = []
    model.fc = Identity()
    with torch.no_grad():  # do not calculate the gradients
        for inputs, labels in dataloader:  # iterate over the data
            inputs = inputs.to(device)  # move the input images to the device
            labels = labels.to(device)  # move the labels to the device
            outputs = model(inputs)  # forward pass
            embeddings.append(outputs.cpu().numpy())  # append the embeddings to the list
            # Retrieve the image paths from the dataloader dataset
            batch_ids = dataloader.dataset.samples  # get the image paths
            paths.extend([path for path, _ in batch_ids])  # append the image paths to the list
    embeddings = np.concatenate(embeddings)  # concatenate the embeddings

    return embeddings, labels, paths
