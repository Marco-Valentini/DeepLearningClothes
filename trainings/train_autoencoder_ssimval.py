import os

# import required libraries to train an autoencoder model (used later to compute image embeddings)
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from models.model_autoencoder_ssimval import AutoEncoder, fine_tune_model, test_model, freeze, SSIM_Loss
import torch

# use GPU if available (mps is mac M1 pro GPU)
device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
# device = torch.device('cpu')
print(f"Working on device {device}")

# set the working directory to the path of the file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# define the transformations to be applied to the images, the employed model doesn't require normalization
data_transforms = {
    'train': transforms.Compose(
        [  # For training, the data gets transformed by undergoing augmentation.
            transforms.RandomCrop(size=128, pad_if_needed=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ]),
    'val': transforms.Compose([
        transforms.Resize(size=128),  # Just normalization for validation, no augmentation.
        transforms.ToTensor()
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=128),  # Just Resizing for testing, no normalization and no augmentation.
        transforms.Resize(size=128),
        transforms.ToTensor()
    ])
}

# Set the paths to your training and validation data folders
data_dir = '../Images_categorized'
train_dir = data_dir + '/train'
val_dir = data_dir + '/val'
test_dir = data_dir + '/test'

# Create the ImageFolder datasets for training and validation
print("Loading the datasets")
image_datasets = {
    'train': ImageFolder(train_dir, transform=data_transforms['train']),
    'val': ImageFolder(val_dir, transform=data_transforms['val']),
    'test': ImageFolder(test_dir, transform=data_transforms['test'])
}
print("Datasets loaded")

# Create data loaders for training and validation
print("Creating the dataloaders")
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=128, shuffle=True, num_workers=0),
    'val': DataLoader(image_datasets['val'], batch_size=128, shuffle=False, num_workers=0),
    'test': DataLoader(image_datasets['test'], batch_size=128, shuffle=False, num_workers=0)
}
print("Dataloaders created")
# define the model, the optimizer and a criterion (loss function)
model = AutoEncoder(C=128, M=128, in_chan=3, out_chan=3).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

criterion = SSIM_Loss()

num_epochs = 50  # number of epochs to train the model

# train the model
model = fine_tune_model(model=model, freezer=freeze, optimizer=optimizer, dataloaders=dataloaders, device=device,
                        criterion=criterion, n_layers_to_freeze=0, num_epochs=num_epochs)

# test the model
test_model(model, dataloaders['test'], device, criterion)
