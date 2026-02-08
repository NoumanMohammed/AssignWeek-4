"""
Convolutional Neural Network for CIFAR-10 Image Classification
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

# ============================================================================
# COMMIT: Loaded and preprocessed CIFAR-10 dataset
# ============================================================================

# Define transformations: convert to tensor and normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Create data loaders for batching
train_loader = torch.utils.data.DataLoader(
    trainset,
    batch_size=32,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    testset,
    batch_size=32,
    shuffle=False
)

print("CIFAR-10 dataset loaded successfully!")
print(f"Training samples: {len(trainset)}")
print(f"Test samples: {len(testset)}")
