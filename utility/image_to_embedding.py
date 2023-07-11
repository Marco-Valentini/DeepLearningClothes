# Description: this file contains the function to convert the images to embeddings using the pretrained model and the
import numpy as np
import torch
from utility.identity import Identity
from utility.resnet18_modified import Resnet18Modified


def image_to_embedding(dataloader, model: Resnet18Modified, device):
    """
    given a dataloader, a model and a device, return the embeddings of the dataset and the labels
    :param dataloader: the dataloader of the dataset
    :param model: cnn resnet18 model pretrained and finetuned with fashion dataset
    :param device: the device to run the model on
    :return: the embeddings of the dataset the labels of the dataset and the image paths of the dataset
    """
    # get the embeddings
    embeddings = []  # list to store the embeddings
    model.fc2 = Identity()  # replace the last fully connected layer with an identity layer
    with torch.no_grad():  # do not calculate the gradients
        for inputs, labels in dataloader:  # iterate over the data
            inputs = inputs.to(device)  # move the input images to the device
            labels = labels.to(device)  # move the labels to the device
            outputs = model(inputs)  # forward pass
            embeddings.append(outputs.cpu().numpy())  # append the embeddings to the list

    embeddings = np.concatenate(embeddings)  # concatenate the embeddings
    # Retrieve the image paths from the dataloader dataset
    ids = dataloader.dataset.image_files  # get the image paths
    # get the ids of the images
    ids = [int(image_name.split('.')[0]) for image_name in ids]
    return embeddings, labels, ids
