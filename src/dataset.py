import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from .config import (
    INPUT_IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    BATCH_SIZE
)


class FERDataset(Dataset):
    def __init__(self, dataframe, usage, transform=None):
        self.data = dataframe[dataframe["Usage"] == usage].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label = int(row["emotion"])

        pixels = np.array(row["pixels"].split(), dtype=np.uint8)
        image = pixels.reshape(48, 48)

        image = Image.fromarray(image).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(
            INPUT_IMAGE_SIZE,
            scale=(0.8, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize(
            (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_fer2013_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"FER2013 CSV not found at: {csv_path}")
    return pd.read_csv(csv_path)


def create_datasets(csv_path):
    df = load_fer2013_csv(csv_path)

    train_dataset = FERDataset(
        df,
        usage="Training",
        transform=get_train_transform()
    )

    val_dataset = FERDataset(
        df,
        usage="PublicTest",
        transform=get_eval_transform()
    )

    test_dataset = FERDataset(
        df,
        usage="PrivateTest",
        transform=get_eval_transform()
    )

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(train_dataset, val_dataset, test_dataset):
    class_counts = train_dataset.data["emotion"].value_counts().sort_index().values
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)

    train_labels = train_dataset.data["emotion"].values
    sample_weights = [class_weights[label] for label in train_labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return train_loader, val_loader, test_loader

