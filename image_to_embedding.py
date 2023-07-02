# Description: this file contains the function to convert the images to embeddings using the pretrained model and the
# dataloader
import numpy as np
import torch


def image_to_embedding(dataloader, model, device):
    """
    given a dataloader, a model and a device, return the embeddings of the dataset and the labels
    :param dataloader: the dataloader of the dataset
    :param model: cnn resnet18 model pretrained and finetuned with fashion dataset
    :param device: the device to run the model on
    :return:
    """
    # get the embeddings
    embeddings = []
    with torch.no_grad():  # do not calculate the gradients
        for inputs, labels in dataloader:  # iterate over the data
            inputs = inputs.to(device)  # move the input images to the device
            labels = labels.to(device)  # move the labels to the device
            outputs = model(inputs)  # forward pass
            embeddings.append(outputs.cpu().numpy())  # append the embeddings to the list
    embeddings = np.concatenate(embeddings)  # concatenate the embeddings

    return embeddings, labels
