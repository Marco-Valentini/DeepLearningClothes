import torch

def find_closest_embeddings(recons_embeddings, embeddings, set_ids):
    """
    Find the closest embeddings in the catalogue to the reconstructed embeddings
    :param recons_embeddings: the reconstructed embeddings (tensor) (shape: (batch_size, embedding_size))
    :return: the closest embeddings (tensor)
    """
    # TODO valutare se invece si usa una euclidean distance cosa succede
    # embeddings = torch.from_numpy(embeddings).to(device)  # convert to tensor
    # with open('./reduced_data/resnet_IDs_list') as f:
    #     resnet_IDs_list = json.load(f)
    closest_embeddings = []
    for i in range(recons_embeddings.shape[0]):  # for each reconstructed embedding in the batch
        # compute the cosine similarity between the reconstructed embedding and the embeddings of the catalogue
        distances = torch.cdist(recons_embeddings[i, :].unsqueeze(0), embeddings)
        # find the index of the closest embedding
        idx = torch.min(distances, dim=1).indices
        # append the closest embedding to the list
        closest_embeddings.append(list(set_ids)[idx.item()])  # retrieve the internal ID of the item
    return closest_embeddings

def find_top_k_closest_embeddings(recons_embeddings, masked_positions,shoes_emebeddings, shoes_idx,tops_embeddings, tops_idx, accessories_embeddings, accessories_idx, bottoms_embeddings, bottoms_idx, topk=10):
    closest_embeddings = []
    for i, pos in enumerate(masked_positions):
        if pos == 0:
            distances = torch.cdist(recons_embeddings[i, :].unsqueeze(0),shoes_emebeddings)
            idx = torch.topk(distances, topk, largest=False, dim=1).indices
            idx = idx.tolist()
            closest_embeddings.append([list(shoes_idx)[i] for i in idx[0]])
        elif pos == 1:
            distances = torch.cdist(recons_embeddings[i, :].unsqueeze(0),tops_embeddings)
            idx = torch.topk(distances, topk, largest=False, dim=1).indices
            idx = idx.tolist()
            closest_embeddings.append([list(tops_idx)[i] for i in idx[0]])
        elif pos == 2:
            distances = torch.cdist(recons_embeddings[i, :].unsqueeze(0),accessories_embeddings)
            idx = torch.topk(distances, topk, largest=False, dim=1).indices
            idx = idx.tolist()
            closest_embeddings.append([list(accessories_idx)[i] for i in idx[0]])
        elif pos == 3:
            distances = torch.cdist(recons_embeddings[i, :].unsqueeze(0),bottoms_embeddings)
            idx = torch.topk(distances, topk, largest=False, dim=1).indices
            idx = idx.tolist()
            closest_embeddings.append([list(bottoms_idx)[i] for i in idx[0]])
    return closest_embeddings

