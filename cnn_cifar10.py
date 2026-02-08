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

# ============================================================================
# COMMIT: Implemented CNN for CIFAR-10 classification
# ============================================================================

# Define a simple CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layer: 3 input channels (RGB), 32 output channels
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # Max pooling layer: reduces spatial dimensions by half
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layer: 32*16*16 -> 10 classes
        self.fc1 = nn.Linear(32 * 16 * 16, 10)

    def forward(self, x):
        # Apply conv + ReLU + pooling
        x = self.pool(torch.relu(self.conv1(x)))
        # Flatten for fully connected layer
        x = x.view(-1, 32 * 16 * 16)
        # Output layer
        x = self.fc1(x)
        return x

# Initialize model, loss function, and optimizer
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
print("\nTraining CNN on CIFAR-10...")
for epoch in range(5):
    running_loss = 0.0
    for images, labels in train_loader:
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/5, Loss: {avg_loss:.4f}")

print("Training complete!")