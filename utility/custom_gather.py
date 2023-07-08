import torch


def custom_gather(outputs, masked_positions, device):
    """
    This function takes as input the output of the BERT-like model and the masked positions and returns the embedding of the masked positions
    :param outputs: the output of the BERT-like model
    :param masked_positions: the positions of the masked items
    :param device: the device on which the model is running (cpu or gpu)
    :return: the embedding of the masked positions
    """
    # convert masked_positions to numpy array of integers
    masked_positions = masked_positions.cpu().numpy().astype(int)
    # create a tensor of zeros of shape (batch_size,embedding_size)
    masked = torch.zeros(outputs.shape[0], 1, outputs.shape[2]).to(device)

    for i in range(outputs.shape[0]):
        # retrieve the final embedding to give in input to the MLM head
        masked[i, 0, :] = outputs[i, masked_positions[i], :]
    return masked
