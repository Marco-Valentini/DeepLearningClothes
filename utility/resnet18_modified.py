from torch import nn


class Resnet18Modified(nn.Module):
    """
    This class modifies the Resnet18 model by replacing the last fully connected layer with a new fully connected layer
    with the given number of output features (the desired size of the embeddings) and add a final fully connected layer
    with the given number of classes to classify (bottoms, tops, shoes, accessories).
    """
    def __init__(self, model, dim_embeddings, num_classes):
        super(Resnet18Modified, self).__init__()
        self.model = model
        self.model.fc = nn.Linear(model.fc.in_features, dim_embeddings)
        self.fc2 = nn.Linear(dim_embeddings, num_classes)

    def forward(self, x):
        """
        This function performs a forward pass on the given input. It first passes the input through the modified Resnet18
        model and then passes the output of the model through the final fully connected layer.
        :param x: the input to the model (images)
        :return: the output of the model (logits)
        """
        x = self.model(x)
        x = self.fc2(x)
        return x