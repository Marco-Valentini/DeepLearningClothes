import numpy as np
import torch
from tqdm import tqdm
def image_to_embedding_VAE(dataloader, model, device):
    """
    given a dataloader, a model and a device, return the embeddings of the dataset and the labels
    :param dataloader: the dataloader of the dataset
    :param model: cnn resnet18 model pretrained and finetuned with fashion dataset
    :param device: the device to run the model on
    :return: the embeddings of the dataset the labels of the dataset and the image paths of the dataset
    """
    # get the embeddings
    embeddings = []  # list to store the embeddings
    with torch.no_grad():  # do not calculate the gradients
        for inputs, _ in tqdm(dataloader):  # iterate over the data
            inputs = inputs.to(device)  # move the input images to the device
            _,outputs,_,_ = model(inputs)  # forward pass
            embeddings.append(outputs.cpu().numpy())  # append the embeddings to the list

    embeddings = np.concatenate(embeddings)  # concatenate the embeddings
    # Retrieve the image paths from the dataloader dataset
    ids = dataloader.dataset.image_files  # get the image paths
    # get the ids of the images
    ids = [int(image_name.split('.')[0]) for image_name in ids]
    return embeddings, ids