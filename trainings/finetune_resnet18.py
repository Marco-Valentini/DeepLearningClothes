import os
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights, resnet18
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms

# set the working directory to the path of the file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

device = torch.device("mps" if torch.has_mps else "cpu")  # use the mps device if available

# This code expects data to be in same format as Imagenet.
# Thus, the dataset has three folds, named 'train', 'val' and 'test'.
# The 'train' folder contains training set and 'val' folder contains validation set on which accuracy is measured.
# The 'test' folder is used for testing the pre-trained model.

# The structure within 'train', 'val' and 'test' folders will be the same.
# They both contain one folder per class. All the images of that class are inside the folder named by class name.

# So, the structure looks like this :
# |- data_cnn_fine_tuning
#      |- train
#            |- accessories
#                 |- accessory_image_1
#                 |- accessory_image_2
#                        .....
#            |- bottoms
#                 |- bottom_image_1
#                 |- bottom_image_2
#                        .....
#            |- shoes
#                 |- shoes_image_1
#                 |- shoes_image_2
#                        .....
#            |- tops
#                 |- top_image_1
#                 |- top_image_2
#                        .....
#      |- val
#            |- accessories
#            |- bottoms
#            |- shoes
#            |- tops
#      |- test
#            |- accessories
#            |- bottoms
#            |- shoes
#            |- tops

# data loading and shuffling/augmentation/normalization.
# Normalization is a common technique in computer vision which helps the network to converge faster.
# The mean and standard deviation values are taken from Imagenet dataset.

data_transforms = {
    'train': transforms.Compose(
        [  # For training, the data gets transformed by undergoing augmentation and normalization.
            transforms.RandomResizedCrop(224),  # takes a crop of an image at various scales between 0.01 to 0.8 times
            # the size of the image and resizes it to given number
            transforms.RandomHorizontalFlip(),
            # randomly flip image horizontally, it is a common technique in computer
            # vision to augment the size of your data set. Firstly, it increases the number of times the network gets to
            # see the same thing, and secondly it adds rotational invariance to your networks learning.
            transforms.ToTensor(),  # convert the image to a tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalize the image to the ImageNet mean and standard deviation
        ]),
    'val': transforms.Compose([  # Just normalization for validation, no augmentation.
        transforms.Resize(256),  # resize the image to 256x256
        transforms.CenterCrop(224),  # crop the image to 224x224
        transforms.ToTensor(),  # convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalize the image to the ImageNet mean and standard deviation
    ]),
    'test': transforms.Compose([  # Just Resizing for testing, no normalization and no augmentation.
        transforms.Resize(256),  # resize the image to 256x256
        transforms.CenterCrop(224),  # crop the image to 224x224
        transforms.ToTensor(),  # convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalize the image to the ImageNet mean and standard deviation
    ])
}

# Set the paths to your training and validation data folders
data_dir = '../dataset_cnn_fine_tuning'
train_dir = data_dir + '/train'
val_dir = data_dir + '/val'

# Create the ImageFolder datasets for training and validation
image_datasets = {
    'train': ImageFolder(train_dir, transform=data_transforms['train']),
    'val': ImageFolder(val_dir, transform=data_transforms['val']),
    'test': ImageFolder(val_dir, transform=data_transforms['test'])
}

# Create data loaders for training and validation
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=0),
    'val': DataLoader(image_datasets['val'], batch_size=32, shuffle=False, num_workers=0),
    'test': DataLoader(image_datasets['test'], batch_size=32, shuffle=False, num_workers=0)
}

# Load the pre-trained ResNet18 model and modify the last fully connected layer
weights = ResNet18_Weights.IMAGENET1K_V1  # use the weights trained on ImageNet
model = resnet18(weights=weights)  # load the model


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


# Define the loss function and optimizer to be used
criterion = nn.CrossEntropyLoss()  # loss function (categorical cross-entropy)
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)  # optimizer (stochastic gradient descent)
num_epochs = 20  # number of epochs to train the model
num_classes = 4  # number of classes in the dataset (4 in our case)


# fine-tune the model
def fine_tune_model(model, freezer, optimizer, criterion, dataloaders, num_classes, device, num_epochs=20):
    """
    This function fine-tunes the given model using the given optimizer and loss function for the given number of epochs.
    :param model: the model to be fine-tuned
    :param optimizer: the optimizer to be used
    :param criterion: the loss function to be used
    :param dataloaders: the data loaders to be used
    :param num_classes: the number of classes in the dataset
    :param device: the device to be used
    :param num_epochs: the number of epochs to train the model
    :return: the fine-tuned model
    """
    # Get the number of input features for the last fully connected layer
    num_features = model.fc.in_features

    # Modify the last fully connected layer to have the desired number of classes
    model.fc = torch.nn.Linear(num_features, num_classes)

    # freeze parameters of the pre-trained model
    # To freeze a specific number of layers, pass the number as the second argument
    freezer(model, n=25)

    model = model.to(device)  # move the model to the device

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:  # train and validate the model
            print(f'Epoch: {epoch + 1}/{num_epochs} | Phase: {phase}')
            if phase == 'train':
                model.train()  # set model to training mode
            else:
                model.eval()  # set model to evaluate mode

            running_loss = 0.0  # keep track of the loss
            correct = 0  # keep track of the number of correct predictions

            for inputs, labels in dataloaders[phase]:  # iterate over the data
                inputs = inputs.to(device)  # move the input images to the device
                labels = labels.to(device)  # move the labels to the device

                optimizer.zero_grad()  # zero the parameter gradients

                with torch.set_grad_enabled(phase == 'train'):  # only calculate the gradients if training
                    outputs = model(inputs)  # forward pass
                    loss = criterion(outputs, labels)  # calculate the loss

                    if phase == 'train':  # backward pass + optimize only if training
                        loss.backward()  # calculate the gradients
                        optimizer.step()  # update the weights

                running_loss += loss.item() * inputs.size(0)  # update the loss
                _, preds = torch.max(outputs, 1)  # get the predicted classes
                correct += torch.sum(preds == labels.data)  # update the number of correct predictions

            epoch_loss = running_loss / len(image_datasets[phase])  # calculate the average loss
            epoch_acc = correct / len(image_datasets[phase])  # calculate the accuracy

            # compute  precision, recall and F1 score
            precision, recall, f1_score, _ = precision_recall_fscore_support(labels.data.cpu().numpy(), preds.cpu().numpy(), average='macro')
            print(f'{phase} Loss: {epoch_loss}, Accuracy: {epoch_acc}, Precision: {precision}, Recall: {recall}, F1_score: {f1_score}')
    return model


# test the model
def test_model(model, dataloader, device):
    """
    This function tests the given model using the given data loaders.
    :param model: the model to be tested
    :param dataloaders: the data loaders to be used
    :param device: the device to be used
    :return: None
    """
    model.eval()  # set model to evaluate mode
    with torch.no_grad():  # disable gradient calculation
        correct = 0
        total = 0
        for inputs, labels in dataloader:  # iterate over the data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)  # forward pass
            _, preds = torch.max(outputs, 1)  # get the predicted classes

            total += labels.size(0)  # update the total number of images
            correct += (preds == labels).sum().item()  # update the number of correct predictions
            precision, recall, f1_score, _ = precision_recall_fscore_support(labels.data.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=0)

        print(f'Accuracy: {correct / total}, Precision: {precision}, Recall: {recall}, F1_score: {f1_score}')


# fine-tune the model
model = fine_tune_model(model, freeze, optimizer, criterion, dataloaders, num_classes, device, num_epochs)

# test the model
test_model(model, dataloaders['test'], device)

# save the model
torch.save(model.state_dict(), '../models/finetuned_fashion_resnet18.pth')
