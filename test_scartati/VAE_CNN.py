import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

from tqdm import tqdm
# define a loss based on ssimval
class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        return 100*( 1 - super(SSIM_Loss, self).forward(img1, img2) )


class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class BasicBlockEnc(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes * stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes / stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet18Enc(nn.Module):

    def __init__(self, num_Blocks=None, z_dim=10, nc=3):
        super().__init__()
        if num_Blocks is None:
            num_Blocks = [2, 2, 2, 2]
        self.in_planes = 64
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)  # this is the UPSAMPLING part
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2)
        self.linear = nn.Linear(512, 2 * z_dim)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        mu = x[:, :self.z_dim]
        logvar = x[:, self.z_dim:]
        return mu, logvar


class ResNet18Dec(nn.Module):

    def __init__(self, num_Blocks=None, z_dim=10, nc=3):
        super().__init__()
        if num_Blocks is None:
            num_Blocks = [2, 2, 2, 2]
        self.in_planes = 512

        self.linear = nn.Linear(z_dim, 512)

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=4)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = torch.sigmoid(self.conv1(x))
        x = x.view(x.size(0), 3, 64, 64)
        return x


class VAE(nn.Module):

    def __init__(self, z_dim):
        super().__init__()
        self.encoder = ResNet18Enc(z_dim=z_dim)
        self.decoder = ResNet18Dec(z_dim=z_dim)
        self.dim_embeddings = z_dim

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x = self.decoder(z)
        return x, z, mean, logvar

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2)  # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std)
        return epsilon * std + mean




# freeze the given number of layers of the given model
def freeze(model, n=None):
    """
    this function freezes the given number of layers of the given model in order to fine-tune it.
    :param model: the model to be frozen
    :param n: the number of layers to be frozen, if None, all the layers are frozen
    :return: the frozen model
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
    """

    # freeze parameters of the pre-trained model
    # To freeze a specific number of layers, pass the number as the second argument
    freezer(model, n=n_layers_to_freeze)

    model = model.to(device)  # move the model to the device

    train_loss = []  # keep track of the loss of the training phase
    val_loss = []  # keep track of the loss of the validation phase
    train_ssm_loss = []  # keep track of the reconstruction loss of the training phase
    val_ssm_loss = []  # keep track of the reconstruction loss of the validation phase
    train_kld_loss = []  # keep track of the KL divergence loss of the training phase
    val_kld_loss = []  # keep track of the KL divergence loss of the validation phase
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
            running_loss_ssm = 0.0 # keep track of the reconstruction loss
            running_loss_kld = 0.0 # keep track of the KL divergence loss

            for inputs, labels in tqdm(dataloaders[phase]):  # iterate over the data
                inputs = inputs.to(device)  # move the input images to the device

                optimizer.zero_grad()  # zero the parameter gradients

                with torch.set_grad_enabled(phase == 'train'):  # only calculate the gradients if training
                    outputs,latent,mu,log_var = model(inputs)  # forward pass
                    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
                    ssm_loss = criterion(outputs, inputs)
                    kld_weight = 0.005
                    loss = ssm_loss + kld_weight*kld_loss  # calculate the loss

                    if phase == 'train':  # backward pass + optimize only if training
                        loss.backward()  # calculate the gradients
                        optimizer.step()  # update the weights

                running_loss += loss.item() * inputs.size(0)  # update the loss
                running_loss_ssm += ssm_loss.item() * inputs.size(0)  # update the reconstruction loss
                running_loss_kld += kld_loss.item() * inputs.size(0)  # update the KL divergence loss

            epoch_loss = running_loss / len(dataloaders[phase].dataset)  # calculate the average loss
            epoch_loss_ssm = running_loss_ssm / len(dataloaders[phase].dataset)  # calculate the average reconstruction loss
            epoch_loss_kld = running_loss_kld / len(dataloaders[phase].dataset)  # calculate the average KL divergence loss

            print(f'{phase} Total Loss: {epoch_loss}')
            print(f'{phase} Loss of reconstruction: {epoch_loss_ssm}')
            print(f'{phase} KLD Loss: {epoch_loss_kld}')

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_ssm_loss.append(epoch_loss_ssm)
                train_kld_loss.append(epoch_loss_kld)
            else:
                val_loss.append(epoch_loss)
                val_ssm_loss.append(epoch_loss_ssm)
                val_kld_loss.append(epoch_loss_kld)
                if epoch_loss <= valid_loss_min:
                    print(f'Validation loss decreased ({valid_loss_min:.6f} --> {epoch_loss:.6f}).  Saving model ...')
                    checkpoint = {'model_state_dict': model.state_dict(),
                                  'dim_embeddings': model.dim_embeddings}
                    now = datetime.now()
                    dt_string = now.strftime("%Y_%m_%d")
                    torch.save(checkpoint, f'./{dt_string}_trained_fashion_VAE_resnet18_{model.dim_embeddings}_with_ssmLoss.pth')
                    valid_loss_min = epoch_loss
                    best_model = model
                else:
                    early_stopping += 1
        if early_stopping == 5:
            print('Early stopping!')
            break
    plt.plot(train_loss, label='train_total')
    plt.plot(val_loss, label='val_total')
    plt.plot(train_ssm_loss, label='train_ssimval')
    plt.plot(val_ssm_loss, label='val_ssimmval')
    plt.plot(train_kld_loss, label='train_kld')
    plt.plot(val_kld_loss, label='val_kld')
    plt.legend()
    plt.title('Overall loss in training the VAE')
    plt.show()
    return best_model


# test the model
def test_model(model, dataloader, device):
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
        total_kld = 0
        for inputs, labels in dataloader:  # iterate over the data
            inputs = inputs.to(device)  # move the input images to the device

            kld_weight = 0.005
            outputs, latent, mu, log_var = model(inputs)  # forward pass
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
            ssm = criterion(outputs, inputs)  # calculate the error
            kld_ = kld_weight * kld_loss

            total_error_ssm += ssm.item()*inputs.size(0)  # update the total number of images
            total_kld += kld_.item()*inputs.size(0)

        print(f'Reconstruction Loss of test: {total_error_ssm/len(dataloader.dataset)}')
        print(f'KLD Loss of test: {total_kld/len(dataloader.dataset)}')