import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import random_split
from utils import get_basis_vec

class CustomImageDataset(Dataset):
    def __init__(self, dataset, transform1, transform2 = None):
        """
        Args:
            root_dir (string): Directory with all the images, organized by class subdirectories.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataset = dataset
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        """Return the total number of images."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Return a sample image and its label."""
        img_path = self.dataset[idx][0]
        label = self.dataset[idx][1]
        
        # Load image
        image = Image.open(img_path).convert("RGB")  # Ensure the image is RGB
        image1 = self.transform1(image)
        
        if self.transform2:
            image2 = self.transform2(image)
            return image1, image2, label

        return image1, label


def get_dataset(root_dir, train_pct):
    print(f"Domain name: {root_dir.split('/')[-1]}")
    dataset = []
    classes = sorted(os.listdir(root_dir))
    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}

    for class_name in classes:
        class_folder = os.path.join(root_dir, class_name)
        for img_name in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_name)
            label = class_to_idx[class_name]
            dataset.append((img_path, label))
    
    train_size = int(train_pct * len(dataset))  # train_pct% for training
    test_size = len(dataset) - train_size  # rest for testing

    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    return train_dataset, test_dataset
