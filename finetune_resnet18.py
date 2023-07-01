import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights, resnet18
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms

# Define the data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}

# Set the paths to your training and validation data folders
data_dir = './dataset_cnn_fine_tuning'
train_dir = data_dir + '/train'
val_dir = data_dir + '/val'

# Create the ImageFolder datasets for training and validation
image_datasets = {
    'train': ImageFolder(train_dir, transform=data_transforms['train']),
    'val': ImageFolder(val_dir, transform=data_transforms['val'])
}

# Create data loaders for training and ivalidation
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=0),
    'val': DataLoader(image_datasets['val'], batch_size=32, shuffle=False, num_workers=0)
}

# Load the pre-trained ResNet18 model and modify the last fully connected layer
weights = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=weights)

# Freeze all the parameters of the pre-trained model
for param in model.parameters():
    param.requires_grad = False

# Get the number of input features for the last fully connected layer
num_features = model.fc.in_features

# Modify the last fully connected layer to have the desired number of classes
num_classes = 4
model.fc = torch.nn.Linear(num_features, num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

# Fine-tune the model
device = torch.device("mps" if torch.has_mps else "cpu")
model = model.to(device)

epochs = 20
for epoch in range(epochs):
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        correct = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(image_datasets[phase])
        # epoch_acc = correct.double() / len(image_datasets[phase])

        print(f'{phase} Loss: {epoch_loss}')

# save the model
torch.save(model.state_dict(), 'finetuned_fashion_rsnet18.pth')

