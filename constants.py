import numpy as np

np.random.seed(42)  # for reproducibility


def get_special_embeddings(emb_dimension):
    """
    This function returns the special embeddings CLS and MASK.
    :param emb_dimension: the dimension of the embeddings
    :return: CLS and MASK embeddings
    """
    CLS = np.random.randn(1, emb_dimension)
    MASK = np.random.randn(1, emb_dimension)
    return CLS, MASK


API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjMTY5ZDBlZC1kY2QzLTQzNDYtYjc0OS02YzkzM2M3YjIyOTAifQ=='
