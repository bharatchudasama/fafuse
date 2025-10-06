import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class SkinClassificationDataset(Dataset):
    """
    Dataloader for the ISIC 2018 skin lesion classification task.
    Reads images and their corresponding labels from a CSV file.
    """
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels_df = pd.read_csv(csv_file)
        self.transform = transform
        
        # Determine the base directory from the csv_file name (train, val, or test)
        if 'train' in csv_file:
            self.image_dir = 'data/classification/ISIC2018_Task3_Training_Input/'
        elif 'val' in csv_file:
            self.image_dir = 'data/classification/ISIC2018_Task3_Validation_Input/'
        elif 'test' in csv_file:
            self.image_dir = 'data/classification/ISIC2018_Task3_Test_Input/'
        else:
            # Fallback to a default directory if naming convention is different
            self.image_dir = 'data/classification/ISIC2018_Task3_Training_Input/'

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_dir, self.labels_df.iloc[idx, 0])
        
        try:
            image = Image.open(img_name).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Cannot find image file at {img_name}")
            # Return a placeholder tensor if an image is missing, or raise an error
            return torch.zeros((3, 224, 224)), -1

        label = int(self.labels_df.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label

def get_transforms():
    """
    Defines and returns the data augmentation and normalization pipelines.
    These are standard transforms for image classification tasks.
    """
    # Define transforms for the training set (includes data augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Define transforms for the validation/testing set (only resizing and normalization)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform
