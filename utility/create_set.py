import torch
import json

def create_set(json_path,embedding_size):
    """
    This function takes as input a json file containing the outfits and the embedding size.
    It returns a tensor of shape (n_outfits, 4, embedding_size) containing the embeddings of the items in the outfit.
    :param json_path: the path of the json file containing the outfits
    :param embedding_size: the embedding size of the items
    :return: a tensor of shape (n_outfits, 4, embedding_size) containing the embeddings of the items in the outfit
    """
    with open(json_path) as json_file:
        data = json.load(json_file)
    n_outfits = len(data)
    with open('../embedding/catalogue_embeddings') as json_file:
        catalogue_embedding =json.load(json_file)
    dataset = torch.zeros((n_outfits, 4, embedding_size))  # (seq_length, batch_size, feature_dim)
    for i, outfit in enumerate(data):
        clothes_in_list = outfit['item_id']
        dataset[i][0] = catalogue_embedding[clothes_in_list[0]]
        dataset[i][1] = catalogue_embedding[clothes_in_list[1]]
        dataset[i][2] = catalogue_embedding[clothes_in_list[2]]
        dataset[i][3] = catalogue_embedding[clothes_in_list[3]]
    return dataset

# se facciamo con dataframe
# df = pd.read_csv('../embedding/catalogue_embeddings')
# df[clothes_in_list[0]]



