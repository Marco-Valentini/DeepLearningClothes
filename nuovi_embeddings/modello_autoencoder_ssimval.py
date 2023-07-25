# import the required libraries to define an autoencoder model as seen in the repository https://github.com/VainF/pytorch-msssim/tree/master
import matplotlib.pyplot as plt
from datetime import datetime
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function

# define a loss based on ssimval
class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        return 100*( 1 - super(SSIM_Loss, self).forward(img1, img2) )

class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        ctx.save_for_backward(inputs, inputs.new_ones(1) * bound)
        return inputs.clamp(min=bound)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, bound = ctx.saved_tensors

        pass_through_1 = (inputs >= bound)
        pass_through_2 = (grad_output < 0)

        pass_through = (pass_through_1 | pass_through_2)
        return pass_through.type(grad_output.dtype) * grad_output, None


class GDN(nn.Module):
    def __init__(self,
                 num_features,
                 inverse=False,
                 gamma_init=.1,
                 beta_bound=1e-6,
                 gamma_bound=0.0,
                 reparam_offset=2 ** -18,
                 ):
        super(GDN, self).__init__()
        self._inverse = inverse
        self.num_features = num_features
        self.reparam_offset = reparam_offset
        self.pedestal = self.reparam_offset ** 2

        beta_init = torch.sqrt(torch.ones(num_features, dtype=torch.float) + self.pedestal)
        gama_init = torch.sqrt(torch.full((num_features, num_features), fill_value=gamma_init, dtype=torch.float)
                               * torch.eye(num_features, dtype=torch.float) + self.pedestal)

        self.beta = nn.Parameter(beta_init)
        self.gamma = nn.Parameter(gama_init)

        self.beta_bound = (beta_bound + self.pedestal) ** 0.5
        self.gamma_bound = (gamma_bound + self.pedestal) ** 0.5

    def _reparam(self, var, bound):
        var = LowerBound.apply(var, bound)
        return (var ** 2) - self.pedestal

    def forward(self, x):
        gamma = self._reparam(self.gamma, self.gamma_bound).view(self.num_features, self.num_features, 1,
                                                                 1)  # expand to (C, C, 1, 1)
        beta = self._reparam(self.beta, self.beta_bound)
        norm_pool = F.conv2d(x ** 2, gamma, bias=beta, stride=1, padding=0)
        norm_pool = torch.sqrt(norm_pool)

        if self._inverse:
            norm_pool = x * norm_pool
        else:
            norm_pool = x / norm_pool
        return norm_pool


# https://arxiv.org/pdf/1611.01704.pdf
# A simplfied version without quantization
class AutoEncoder(nn.Module):
    def __init__(self, C=128, M=128, in_chan=3, out_chan=3):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(C=C, M=M, in_chan=in_chan)
        self.decoder = Decoder(C=C, M=M, out_chan=out_chan)
        # add linear layers to reduce the dimensionality of the hidden representation
        self.ffnn1 = nn.Linear(128*8*8, 1024)
        self.ffn2 = nn.Linear(1024, 128*8*8)

    def forward(self, x, **kargs):
        code = self.encoder(x)  # shape (batch_size, 128, 8, 8)
        code = code.reshape(code.shape[0], -1)  # flatten the hidden representation
        code = self.ffnn1(code)  # the hidden representation we will return
        out = self.ffn2(code)
        out = out.reshape(out.shape[0], 128, 8, 8)
        out = self.decoder(out)
        return out, code  # return the reconstructed image and the hidden representation


class Encoder(nn.Module):
    """ Encoder
    """

    def __init__(self, C=32, M=128, in_chan=3):
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=M, kernel_size=5, stride=2, padding=2, bias=False),
            GDN(M),

            nn.Conv2d(in_channels=M, out_channels=M, kernel_size=5, stride=2, padding=2, bias=False),
            GDN(M),

            nn.Conv2d(in_channels=M, out_channels=M, kernel_size=5, stride=2, padding=2, bias=False),
            GDN(M),

            nn.Conv2d(in_channels=M, out_channels=C, kernel_size=5, stride=2, padding=2, bias=False)
        )

    def forward(self, x):
        return self.enc(x)


class Decoder(nn.Module):
    """ Decoder
    """

    def __init__(self, C=32, M=128, out_chan=3):
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(in_channels=C, out_channels=M, kernel_size=5, stride=2, padding=2, output_padding=1,
                               bias=False),
            GDN(M, inverse=True),

            nn.ConvTranspose2d(in_channels=M, out_channels=M, kernel_size=5, stride=2, padding=2, output_padding=1,
                               bias=False),
            GDN(M, inverse=True),

            nn.ConvTranspose2d(in_channels=M, out_channels=M, kernel_size=5, stride=2, padding=2, output_padding=1,
                               bias=False),
            GDN(M, inverse=True),

            nn.ConvTranspose2d(in_channels=M, out_channels=out_chan, kernel_size=5, stride=2, padding=2,
                               output_padding=1, bias=False),
        )

    def forward(self, q):
        return torch.sigmoid(self.dec(q))

# freeze the given number of layers of the given model
def freeze(model, n=None):
    """
    This function freezes the given number of layers of the given model in order to fine-tune it.
    :param model: the model to be frozen
    :param n: the number of layers to be frozen, if None, all the layers are frozen
    :return: the frozen model
    #TODO non usata si può togliere
    """
    if n is None:
        for param in model.parameters():
            param.requires_grad = False
    else:
        count = 0
        for param in model.parameters():
            if count < n:
                param.requires_grad = False
            count += 1

# fine-tune the model
def fine_tune_model(model, freezer, optimizer, dataloaders, device, criterion, n_layers_to_freeze=0,num_epochs=20):
    """
    This function fine-tunes the given model using the given optimizer and loss function for the given number of epochs.
    :param model: the model to be fine-tuned
    :param optimizer: the optimizer to be used
    :param criterion: the loss function to be used
    :param dataloaders: the data loaders to be used
    :param device: the device to be used
    :param num_epochs: the number of epochs to train the model
    :return: the fine-tuned model
    # TODO cambia il nome qua
    """

    # freeze parameters of the pre-trained model
    # To freeze a specific number of layers, pass the number as the second argument
    freezer(model, n=n_layers_to_freeze)

    model = model.to(device)  # move the model to the device

    train_loss = []  # keep track of the loss of the training phase
    val_loss = []  # keep track of the loss of the validation phase
    valid_loss_min = np.Inf  # track change in validation loss
    best_model = model  # keep track of the best model
    early_stopping = 0  # keep track of the number of epochs without improvement
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:  # train and validate the model
            print(f'Fine-tuning the VAE Epoch: {epoch + 1}/{num_epochs} | Phase: {phase}')
            if phase == 'train':
                model.train()  # set model to training mode
                print('Training...')
            else:
                model.eval()  # set model to evaluate mode
                print('Validating...')

            running_loss = 0.0  # keep track of the loss

            for inputs, labels in tqdm(dataloaders[phase]):  # iterate over the data
                inputs = inputs.to(device)  # move the input images to the device

                optimizer.zero_grad()  # zero the parameter gradients

                with torch.set_grad_enabled(phase == 'train'):  # only calculate the gradients if training
                    outputs,_ = model(inputs)  # forward pass
                    ssm_loss = criterion(outputs, inputs)
                    loss = ssm_loss  # calculate the loss

                    if phase == 'train':  # backward pass + optimize only if training
                        loss.backward()  # calculate the gradients
                        optimizer.step()  # update the weights

                running_loss += loss.item() * inputs.size(0)  # update the loss

            epoch_loss = running_loss / len(dataloaders[phase])  # calculate the average loss

            print(f'{phase} Total Loss: {epoch_loss}')

            if phase == 'train':
                train_loss.append(epoch_loss)
            else:
                val_loss.append(epoch_loss)
                if epoch_loss <= valid_loss_min:
                    print(f'Validation loss decreased ({valid_loss_min:.6f} --> {epoch_loss:.6f}).  Saving model ...')
                    checkpoint = {'model_state_dict': model.state_dict()}
                    now = datetime.now()
                    dt_string = now.strftime("%Y_%m_%d")
                    torch.save(checkpoint, f'./{dt_string}_trained_fashion_VAE_con_linear_layers.pth')
                    valid_loss_min = epoch_loss
                    best_model = model
                    early_stopping = 0
                else:
                    early_stopping += 1
        if early_stopping == 5:
            print('Early stopping!')
            break
    plt.plot(train_loss, label='train_total')
    plt.plot(val_loss, label='val_total')
    plt.legend()
    plt.title('Overall loss in training the VAE')
    plt.show()
    return best_model


# test the model
def test_model(model, dataloader, device, criterion):
    """
    This function tests the given model using the given data loaders.
    :param model: the model to be tested
    :param dataloader: the data loaders to be used
    :param device: the device to be used
    :return: None
    """
    model = model.to(device)  # move the model to the device
    model.eval()  # set model to evaluate mode
    with torch.no_grad():  # disable gradient calculation
        total_error_ssm = 0
        for inputs, labels in dataloader:  # iterate over the data
            inputs = inputs.to(device)  # move the input images to the device

            outputs, embedding = model(inputs)  # forward pass
            ssm = criterion(outputs, inputs)  # calculate the error


            total_error_ssm += ssm.item()*inputs.size(0)  # update the total number of images

        print(f'Reconstruction Loss of test: {total_error_ssm/len(dataloader.dataset)}')

