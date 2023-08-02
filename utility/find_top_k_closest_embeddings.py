from torch.nn import CosineSimilarity
import torch
import numpy as np
import pandas as pd
import json


def find_top_k_closest_embeddings(recons_embeddings, embeddings_dict, masked_positions, shoes_IDs, tops_IDs,
                                  accessories_IDs, bottoms_IDs, device=torch.device('cpu'), topk=10):
    """
    given the reconstructed embeddings, the embeddings of the catalogue, the masked positions, the IDs of the shoes,
    tops, accessories and bottoms, find the top k closest embeddings in the catalogue for each masked position
    :param recons_embeddings: the reconstructed embeddings
    :param embeddings_dict: the embeddings of the catalogue
    :param masked_positions: the masked positions
    :param shoes_IDs: the IDs of the shoes
    :param tops_IDs: the IDs of the tops
    :param accessories_IDs: the IDs of the accessories
    :param bottoms_IDs: the IDs of the bottoms
    :param device: the device to run the model on
    :param topk: the number of closest embeddings to retrieve
    :return: closest_embeddings: the IDs of the closest embeddings
    """
    embeddings_shoes = embeddings_dict['shoes']
    embeddings_tops = embeddings_dict['tops']
    embeddings_accessories = embeddings_dict['accessories']
    embeddings_bottoms = embeddings_dict['bottoms']
    closest_embeddings = []
    cosine_similarity = CosineSimilarity(dim=1)
    for i, pos in enumerate(masked_positions):
        if pos == 0:  # shoes
            # compute the cosine similarity between the reconstructed embedding and the embeddings of the catalogue
            similarities = cosine_similarity(recons_embeddings[i, :], torch.Tensor(embeddings_shoes).to(device))
            idx = torch.topk(similarities, k=topk).indices
            idx = idx.tolist()
            closest = [shoes_IDs[j] for j in idx]
        elif pos == 1:  # tops
            # compute the cosine similarity between the reconstructed embedding and the embeddings of the catalogue
            similarities = cosine_similarity(recons_embeddings[i, :], torch.Tensor(embeddings_tops).to(device))
            idx = torch.topk(similarities, k=topk).indices
            idx = idx.tolist()
            closest = [tops_IDs[j] for j in idx]
        elif pos == 2:  # accessories
            # compute the cosine similarity between the reconstructed embedding and the embeddings of the catalogue
            similarities = cosine_similarity(recons_embeddings[i, :], torch.Tensor(embeddings_accessories).to(device))
            idx = torch.topk(similarities, k=topk).indices
            idx = idx.tolist()
            closest = [accessories_IDs[j] for j in idx]
        elif pos == 3:  # bottoms
            # compute the cosine similarity between the reconstructed embedding and the embeddings of the catalogue
            similarities = cosine_similarity(recons_embeddings[i, :], torch.Tensor(embeddings_bottoms).to(device))
            idx = torch.topk(similarities, k=topk).indices
            idx = idx.tolist()
            closest = [bottoms_IDs[j] for j in idx]
        # append the closest embedding to the list
        closest_embeddings.append(closest)
    return closest_embeddings
