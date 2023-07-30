import torch


def find_closest_embeddings(recons_embeddings, embeddings, set_ids):
    """
    Find the closest embeddings in the catalogue to the reconstructed embeddings
    :param recons_embeddings: the reconstructed embeddings (tensor) (shape: (batch_size, embedding_size))
    :param embeddings: the embeddings of the catalogue (tensor) (shape: (num_embeddings, embedding_size))
    :param set_ids: the IDs of the items in the catalogue (list)
    :return: the closest embeddings (tensor)
    """
    closest_embeddings = []
    for i in range(recons_embeddings.shape[0]):  # for each reconstructed embedding in the batch
        # compute the euclidean distances between the reconstructed embedding and the embeddings of the catalogue
        distances = torch.cdist(recons_embeddings[i, :].unsqueeze(0), embeddings)
        # find the index of the closest embedding
        idx = torch.min(distances, dim=1).indices
        # append the closest embedding to the list
        closest_embeddings.append(list(set_ids)[idx.item()])  # retrieve the internal ID of the item
    return closest_embeddings


def find_top_k_closest_embeddings(recons_embeddings, masked_positions, shoes_emebeddings, shoes_idx, tops_embeddings,
                                  tops_idx, accessories_embeddings, accessories_idx, bottoms_embeddings, bottoms_idx,
                                  topk=10):
    """
    Find the top k closest embeddings in the catalogue to the reconstructed embeddings for each masked position.
    :param recons_embeddings: the reconstructed embeddings (tensor) (shape: (batch_size, embedding_size))
    :param masked_positions: the positions of the masked items (list)
    :param shoes_emebeddings: the embeddings of the shoes in the catalogue (tensor) (shape: (num_shoes, embedding_size))
    :param shoes_idx: the IDs of the shoes in the catalogue (list)
    :param tops_embeddings: the embeddings of the tops in the catalogue (tensor) (shape: (num_tops, embedding_size))
    :param tops_idx: the IDs of the tops in the catalogue (list)
    :param accessories_embeddings: the embeddings of the accessories in the catalogue (tensor)
    (shape: (num_accessories, embedding_size))
    :param accessories_idx: the IDs of the accessories in the catalogue (list)
    :param bottoms_embeddings: the embeddings of the bottoms in the catalogue (tensor)
    :param bottoms_idx: the IDs of the bottoms in the catalogue (list)
    :param topk: the number of closest embeddings to retrieve (int)
    :return: the top k closest embeddings (list)
    """
    closest_embeddings = []
    for i, pos in enumerate(masked_positions):
        if pos == 0:
            distances = torch.cdist(recons_embeddings[i, :].unsqueeze(0), shoes_emebeddings)
            idx = torch.topk(distances, topk, largest=False, dim=1).indices
            idx = idx.tolist()
            closest_embeddings.append([list(shoes_idx)[i] for i in idx[0]])
        elif pos == 1:
            distances = torch.cdist(recons_embeddings[i, :].unsqueeze(0), tops_embeddings)
            idx = torch.topk(distances, topk, largest=False, dim=1).indices
            idx = idx.tolist()
            closest_embeddings.append([list(tops_idx)[i] for i in idx[0]])
        elif pos == 2:
            distances = torch.cdist(recons_embeddings[i, :].unsqueeze(0), accessories_embeddings)
            idx = torch.topk(distances, topk, largest=False, dim=1).indices
            idx = idx.tolist()
            closest_embeddings.append([list(accessories_idx)[i] for i in idx[0]])
        elif pos == 3:
            distances = torch.cdist(recons_embeddings[i, :].unsqueeze(0), bottoms_embeddings)
            idx = torch.topk(distances, topk, largest=False, dim=1).indices
            idx = idx.tolist()
            closest_embeddings.append([list(bottoms_idx)[i] for i in idx[0]])
    return closest_embeddings
