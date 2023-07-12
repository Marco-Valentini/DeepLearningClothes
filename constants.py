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
