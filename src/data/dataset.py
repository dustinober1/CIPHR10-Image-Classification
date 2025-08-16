"""
Data loading and preprocessing utilities for CIFAR-10 dataset.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from typing import Tuple, Optional
import os


class CIFAR10DataModule:
    """
    Data module for CIFAR-10 dataset handling.
    """
    
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 32,
        val_split: float = 0.1,
        num_workers: int = 4,
        pin_memory: bool = True,
        download: bool = True
    ):
        """
        Initialize CIFAR-10 data module.
        
        Args:
            data_dir: Directory to store/load data
            batch_size: Batch size for data loaders
            val_split: Fraction of test set to use for validation
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory for GPU training
            download: Whether to download data if not present
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.download = download
        
        # CIFAR-10 classes
        self.classes = (
            'plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        )
        
        # Initialize transforms
        self.train_transforms = self._get_train_transforms()
        self.test_transforms = self._get_test_transforms()
        
        # Initialize datasets
        self._setup_datasets()
    
    def _get_train_transforms(self) -> transforms.Compose:
        """Get training data transformations with augmentation."""
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ])
    
    def _get_test_transforms(self) -> transforms.Compose:
        """Get test/validation data transformations."""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ])
    
    def _setup_datasets(self):
        """Setup train, validation, and test datasets."""
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load training dataset
        self.train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=self.download,
            transform=self.train_transforms
        )
        
        # Load test dataset
        test_dataset_full = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=self.download,
            transform=self.test_transforms
        )
        
        # Split test set into validation and test sets
        val_size = int(len(test_dataset_full) * self.val_split)
        test_size = len(test_dataset_full) - val_size
        
        self.val_dataset, self.test_dataset = random_split(
            test_dataset_full, [val_size, test_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        print(f"Dataset sizes:")
        print(f"  Train: {len(self.train_dataset)}")
        print(f"  Validation: {len(self.val_dataset)}")
        print(f"  Test: {len(self.test_dataset)}")
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get data loaders for train, validation, and test sets.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        return train_loader, val_loader, test_loader
    
    def get_sample_batch(self, dataset_type: str = "train") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample batch from the specified dataset.
        
        Args:
            dataset_type: One of 'train', 'val', or 'test'
            
        Returns:
            Tuple of (images, labels)
        """
        if dataset_type == "train":
            loader = DataLoader(self.train_dataset, batch_size=8, shuffle=True)
        elif dataset_type == "val":
            loader = DataLoader(self.val_dataset, batch_size=8, shuffle=True)
        elif dataset_type == "test":
            loader = DataLoader(self.test_dataset, batch_size=8, shuffle=True)
        else:
            raise ValueError("dataset_type must be one of 'train', 'val', or 'test'")
        
        return next(iter(loader))


def get_cifar10_stats():
    """
    Calculate and return CIFAR-10 dataset statistics.
    
    Returns:
        Dictionary with mean and std for each channel
    """
    # These are the commonly used CIFAR-10 statistics
    return {
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2023, 0.1994, 0.2010]
    }


def create_data_loaders(
    data_dir: str = "./data",
    batch_size: int = 32,
    val_split: float = 0.1,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convenience function to create CIFAR-10 data loaders.
    
    Args:
        data_dir: Directory to store/load data
        batch_size: Batch size for data loaders
        val_split: Fraction of test set to use for validation
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_module = CIFAR10DataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        val_split=val_split,
        num_workers=num_workers
    )
    
    return data_module.get_dataloaders()