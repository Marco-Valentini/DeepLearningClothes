import numpy as np
import pandas as pd

API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjMTY5ZDBlZC1kY2QzLTQzNDYtYjc0OS02YzkzM2M3YjIyOTAifQ=='


np.random.seed(42)  # for reproducibility


def generate_special_embeddings_randomly(emb_dimension):
    """
    This function returns the special embeddings CLS and MASK.
    :param emb_dimension: the dimension of the embeddings
    :return: CLS and MASK embeddings
    """
    CLS = np.random.randn(1, emb_dimension)
    MASK = np.random.randn(1, emb_dimension)
    return CLS, MASK


def initialize_mask_embedding_zeros(emb_dimension):
    """
    This function returns the special embedding MASK.
    :param emb_dimension: the dimension of the embeddings
    :return: MASK embeddings
    """
    MASK = np.zeros((1, emb_dimension))
    return MASK


def task_based_mask_embedding(embeddings):
    """
    This function returns the special embedding MASK based on the task (embedding reconstruction).
    It first normalizes the embeddings to have zero mean and unary standard deviation
     and then computes the mean embedding of all the images embeddings.
    :param embeddings: embeddings of all the images
    :return: MASK embeddings
    """
    # normalize the embeddings
    embeddings = (embeddings - np.mean(embeddings, axis=0)) / np.std(embeddings, axis=0)
    MASK = np.mean(embeddings, axis=0).reshape(1, -1)
    return MASK


def task_based_cls_embedding(emb_dimension, compatibility_outfits: pd.DataFrame, embeddings, IDs):
    """
    This function returns the special embedding CLS based on the task (binary classification).
    It first normalizes the embeddings to have zero mean and unary standard deviation
    For each of the two classes,
    1) the sequence-level representation is obtained by applying a pooling operation (e.g. mean pooling or max pooling)
    on the image embeddings within each sequence. This will result in a single embedding vector for each sequence.
    2) Calculate the average embedding vector for each class by taking the average of the embeddings
    at the sequence level of that class.
    3) Concatenate the mean embeddings for the two classes into a single CLS embedding tensor.
    :param emb_dimension: the dimension of the embeddings
    :param compatibility_outfits: dataframe of outfits with their compatibility label,
    each row is composed of a compatibility label followed by a sequence of 4 images IDs
    :param embeddings: embeddings of all the images
    :param IDs: IDs of all the images (in the same order as the embeddings)
    :return: CLS embeddings of dimension emb_dimension
    """
    # normalize the embeddings
    embeddings = (embeddings - np.mean(embeddings, axis=0)) / np.std(embeddings, axis=0)
    # 1) the sequence-level representation is obtained by applying a pooling operation (e.g. mean pooling or max pooling)
    # on the image embeddings within each sequence. This will result in a single embedding vector of dimension emb_dim/2
    # for each sequence.
    outfits_embeddings = np.zeros((compatibility_outfits.shape[0], emb_dimension // 2))  # 2 is the pooling dimension
    compatibility_labels = np.zeros(compatibility_outfits.shape[0])
    for i in range(compatibility_outfits.shape[0]):  # for each outfit
        outfit = compatibility_outfits.iloc[i]  # get the outfit
        compatibility_labels[i] = outfit[0]  # get the compatibility label
        outfit_embeddings = np.zeros((4, emb_dimension))  # 4 is the sequence length
        for j in range(4):  # for each image in the outfit
            outfit_embeddings[j] = embeddings[IDs.index(outfit[j + 1])]  # get the embedding of the image
        outfits_embeddings[i] = np.mean(outfit_embeddings, axis=0)[:emb_dimension // 2]  # apply pooling

    # 2) Calculate the average embedding vector for each class by taking the average or median of the embeddings
    # at the sequence level of that class.
    mean_embeddings = np.zeros((2, outfits_embeddings.shape[1]))  # 2 is the number of classes
    for i in range(2):  # for each class
        mean_embeddings[i] = np.mean(outfits_embeddings[compatibility_labels == i], axis=0)  # apply pooling
    # 3) Concatenate the mean embeddings for the two classes into a single CLS embedding tensor.
    CLS = np.concatenate((mean_embeddings[0], mean_embeddings[1])).reshape(1, -1)  # concatenate the two classes
    return CLS
