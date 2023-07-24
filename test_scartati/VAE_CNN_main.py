import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from VAE_CNN import VAE, fine_tune_model, test_model, SSIM_Loss, freeze
data_transforms = {
    'train': transforms.Compose(
        [  # For training, the data gets transformed by undergoing augmentation and normalization.
            transforms.Resize(64),  # takes a crop of an image at various scales between 0.01 to 0.8 times
            # the size of the image and resizes it to given number
            # transforms.RandomHorizontalFlip(),
            # randomly flip image horizontally, it is a common technique in computer
            # vision to augment the size of your data set. Firstly, it increases the number of times the network gets to
            # see the same thing, and secondly it adds rotational invariance to your networks learning.
            transforms.ToTensor(),  # convert the image to a tensor
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalize the image to the ImageNet mean and standard deviation
        ]),
    'val': transforms.Compose([  # Just normalization for validation, no augmentation.
        transforms.Resize(64),  # resize the image to 256x256
        #transforms.CenterCrop(64),  # crop the image to 224x224
        transforms.ToTensor(),  # convert the image to a tensor
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalize the image to the ImageNet mean and standard deviation
    ]),
    'test': transforms.Compose([  # Just Resizing for testing, no normalization and no augmentation.
        transforms.Resize(64),  # resize the image to 256x256
        # transforms.CenterCrop(64),  # crop the image to 224x224
        transforms.ToTensor(),  # convert the image to a tensor
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalize the image to the ImageNet mean and standard deviation
    ])
}

# Set the paths to your training and validation data folders
data_dir = '../dataset_cnn_fine_tuning'
train_dir = data_dir + '/train'
val_dir = data_dir + '/val'
test_dir = data_dir + '/test'

# Create the ImageFolder datasets for training and validation
image_datasets = {
    'train': ImageFolder(train_dir, transform=data_transforms['train']),
    'val': ImageFolder(val_dir, transform=data_transforms['val']),
    'test': ImageFolder(test_dir, transform=data_transforms['test'])
}

# Create data loaders for training and validation
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=0),
    'val': DataLoader(image_datasets['val'], batch_size=32, shuffle=False, num_workers=0),
    'test': DataLoader(image_datasets['test'], batch_size=32, shuffle=False, num_workers=0)
}

print(len(dataloaders['train']))


dim_embeddings = 64  # dimension of the embeddings to be learned

# Load the pre-trained ResNet18 model and modify the last fully connected layer
# train the VAE based on ResNet18
# define the model and the optimizer
model = VAE(z_dim=dim_embeddings)
# TODO capisci optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
# TODO capisci loss function
criterion = SSIM_Loss()

# modify the model architecture to output embeddings of the given dimension
# and classify the embeddings into the given number of classes
num_epochs = 50  # number of epochs to train the model

device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
# device = torch.device('cpu')
print(f"Working on device {device}")

# fine-tune the model
model = fine_tune_model(model=model, freezer=freeze, optimizer=optimizer, dataloaders=dataloaders, device=device,criterion=criterion, n_layers_to_freeze=0,  num_epochs=num_epochs)

# test the model
test_model(model, dataloaders['test'], device)
