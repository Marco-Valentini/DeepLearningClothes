import torch
def custom_gather(outputs,masked_positions,device):
    masked = torch.zeros(outputs.shape[0],outputs.shape[2]).to(device) # create a tensor of zeros of shape (batch_size,embedding_size)
    for i in range(outputs.shape[0]):
        masked[i,:] = outputs[i,masked_positions[i],:] # retrieve the final embedding to give in input to the MLM head
    return masked