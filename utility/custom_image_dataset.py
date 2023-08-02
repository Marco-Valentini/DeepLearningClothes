import os

from PIL import Image
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    """
    Custom dataset class for images in a folder with no labels
    """

    def __init__(self, root_dir, data_transform=None):
        """
        This method is called when you instantiate the class
        :param root_dir: The root directory of the dataset
        :param data_transform: The data transformation to apply to the images (e.g. resize, crop, etc.) if needed
        """
        self.root_dir = root_dir
        self.image_files = os.listdir(root_dir)
        self.data_transform = data_transform

    def __len__(self):
        """
        This method is called when you do len(dataset)
        :return: The number of images in the dataset
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        This method is called when you do dataset[idx]
        :param idx: The index of the image to retrieve
        :return: The image and its index
        """
        image_path = os.path.join(self.root_dir, self.image_files[idx])  # Get the path of the image
        image = Image.open(image_path)  # Open the image
        # Preprocess the image if needed
        if self.data_transform is not None:  # If a transformation is provided
            image = self.data_transform(image)  # Apply the transformation
        # Return the image and its index
        return image, idx
