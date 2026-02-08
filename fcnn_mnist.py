"""
Fully Connected Neural Network for MNIST Digit Classification
"""

# ============================================================================
# COMMIT: Loaded and preprocessed MNIST dataset
# ============================================================================


import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# Define transformations: convert to tensor and normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
trainset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

testset = torchvision.datasets.MNIST(
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

# Display a sample image
image, label = trainset[0]
plt.imshow(image.squeeze(), cmap='gray')
plt.title(f"Sample Image - Label {label}")
plt.axis("off")
plt.savefig('mnist_sample.png')
print(f"Sample image saved. Label: {label}")



# ============================================================================
# COMMIT: Implemented FCNN for MNIST classification
# ============================================================================

# Define a simple fully connected network
class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Input: 784, Hidden: 128
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)  # Output: 10 classes

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model, loss function, and optimizer
model = FCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
print("\nTraining FCNN on MNIST...")
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