from torch import nn


class Identity(nn.Module):
    """
    This class is used to replace the last fully connected layer of the model with an identity layer
    """

    def __init__(self):
        """
        Constructor of the class
        """
        super(Identity, self).__init__()

    def forward(self, x):
        """
        Forward pass of the class (does nothing)
        :param x: the input.
        :return: the input.
        """
        return x
