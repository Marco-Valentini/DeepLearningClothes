import torch

def find_closest_embeddings(recons_embeddings, embeddings, set_ids):
    """
    Find the closest embeddings in the catalogue to the reconstructed embeddings
    :param recons_embeddings: the reconstructed embeddings (tensor) (shape: (batch_size, embedding_size))
    :return: the closest embeddings (tensor)
    """
    # TODO valutare se invece si usa una euclidean distance cosa succede
    # embeddings = torch.from_numpy(embeddings).to(device)  # convert to tensor
    # with open('./reduced_data/IDs_list') as f:
    #     IDs_list = json.load(f)
    closest_embeddings = []
    for i in range(recons_embeddings.shape[0]):  # for each reconstructed embedding in the batch
        # compute the cosine similarity between the reconstructed embedding and the embeddings of the catalogue
        distances = torch.cdist(recons_embeddings[i, :].unsqueeze(0), embeddings)
        # find the index of the closest embedding
        idx = torch.min(distances, dim=1).indices
        # append the closest embedding to the list
        closest_embeddings.append(list(set_ids)[idx.item()])  # retrieve the internal ID of the item
    return closest_embeddings

