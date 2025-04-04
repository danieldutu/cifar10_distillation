import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, List


def load_cifar10(batch_size: int = 128, num_workers: int = 2) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Load CIFAR-10 dataset with simple normalization (division by 255) and minimal augmentation.
    
    Args:
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple containing:
            - train_loader: DataLoader for training data
            - test_loader: DataLoader for test data
            - classes: List of class names
    """
    # Define transforms - simply using ToTensor() which divides by 255
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Converts images to [0,1] range by dividing by 255
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),  # Converts images to [0,1] range by dividing by 255
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform_train
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform_test
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    # Class names
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    return train_loader, test_loader, classes


def visualize_samples(data_loader: DataLoader, classes: List[str], num_samples: int = 10) -> None:
    """
    Visualize samples from the dataset.
    
    Args:
        data_loader: DataLoader to take samples from
        classes: List of class names
        num_samples: Number of samples to display
    """
    import matplotlib.pyplot as plt
    
    # Get a batch
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    
    # Convert tensor to numpy for visualization
    images = images.numpy()
    
    # Images are already in [0,1] range, just need to transpose for proper display
    images = np.transpose(images, (0, 2, 3, 1))
    
    # Plot
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        ax = axes[i]
        ax.imshow(images[i])
        ax.set_title(f"{classes[labels[i]]}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    
def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 