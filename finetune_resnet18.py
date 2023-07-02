import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights, resnet18
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms

# Define the data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),  # crop the image to 224x224
        transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
        transforms.ToTensor(),  # convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalize the image
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),  # resize the image to 256x256
        transforms.CenterCrop(224),  # crop the image to 224x224
        transforms.ToTensor(),  # convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalize the image
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),  # resize the image to 256x256
        transforms.CenterCrop(224),  # crop the image to 224x224
        transforms.ToTensor()  # convert the image to a tensor
    ])
}

# Set the paths to your training and validation data folders
data_dir = './dataset_cnn_fine_tuning'
train_dir = data_dir + '/train'
val_dir = data_dir + '/val'

# Create the ImageFolder datasets for training and validation
image_datasets = {
    'train': ImageFolder(train_dir, transform=data_transforms['train']),
    'val': ImageFolder(val_dir, transform=data_transforms['val']),
    'test': ImageFolder(val_dir, transform=data_transforms['test'])
}

# Create data loaders for training and ivalidation
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=0),
    'val': DataLoader(image_datasets['val'], batch_size=32, shuffle=False, num_workers=0),
    'test': DataLoader(image_datasets['test'], batch_size=32, shuffle=False, num_workers=0)
}

# Load the pre-trained ResNet18 model and modify the last fully connected layer
weights = ResNet18_Weights.IMAGENET1K_V1   # use the weights trained on ImageNet
model = resnet18(weights=weights)  # load the model

# Freeze all the parameters of the pre-trained model
for param in model.parameters():
    param.requires_grad = False

# Freeze the first quarter parameters of the pre-trained model
# for param in model.parameters()[:int(len(model.parameters()) / 4)]:
#     param.requires_grad = False



# Get the number of input features for the last fully connected layer
num_features = model.fc.in_features

# Modify the last fully connected layer to have the desired number of classes
num_classes = 4
model.fc = torch.nn.Linear(num_features, num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # loss function
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)  # optimizer

# Fine-tune the model
device = torch.device("mps" if torch.has_mps else "cpu")  # use the mps device if available
model = model.to(device)  # move the model to the device

epochs = 20
for epoch in range(epochs):
    for phase in ['train', 'val']:  # train and validate the model
        if phase == 'train':
            model.train()  # set model to training mode
        else:
            model.eval()   # set model to evaluate mode

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
                    optimizer.step()   # update the weights

            running_loss += loss.item() * inputs.size(0)  # update the loss
            _, preds = torch.max(outputs, 1)  # get the predicted classes
            correct += torch.sum(preds == labels.data)  # update the number of correct predictions

        epoch_loss = running_loss / len(image_datasets[phase])  # calculate the average loss
        epoch_acc = correct / len(image_datasets[phase])  # calculate the accuracy

        print(f'{phase} Loss: {epoch_loss}, Accuracy: {epoch_acc}')

# test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()

    print(f'Accuracy: {correct / total}')

# save the model
torch.save(model.state_dict(), 'models/finetuned_fashion_resnet18.pth')



