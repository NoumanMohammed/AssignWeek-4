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

# Training loop with metrics tracking
print("\nTraining CNN on CIFAR-10...")
train_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(5):
    # Training phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    train_losses.append(avg_loss)
    train_accuracies.append(train_acc)
    
    # Evaluation phase
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_acc = 100 * correct / total
    test_accuracies.append(test_acc)
    
    print(f"Epoch {epoch+1}/5, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

print("Training complete!")

# ============================================================================
# Generate Output Images for Presentation
# ============================================================================

import matplotlib.pyplot as plt

# 1. Accuracy Curve Graph
plt.figure(figsize=(10, 6))
epochs_range = range(1, 6)
plt.plot(epochs_range, train_accuracies, 'b-o', label='Training Accuracy', linewidth=2)
plt.plot(epochs_range, test_accuracies, 'r-s', label='Test Accuracy', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('CNN Training and Test Accuracy on CIFAR-10', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cifar10_accuracy_curve.png', dpi=300)
print("Saved: cifar10_accuracy_curve.png")
plt.close()

# 2. Loss Curve Graph
plt.figure(figsize=(10, 6))
plt.plot(epochs_range, train_losses, 'g-o', label='Training Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('CNN Training Loss on CIFAR-10', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cifar10_loss_curve.png', dpi=300)
print("Saved: cifar10_loss_curve.png")
plt.close()

# 3. Training Metrics Table
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('tight')
ax.axis('off')

table_data = [
    ['Epoch', 'Train Loss', 'Train Accuracy', 'Test Accuracy'],
]
for i in range(5):
    table_data.append([
        f'{i+1}',
        f'{train_losses[i]:.4f}',
        f'{train_accuracies[i]:.2f}%',
        f'{test_accuracies[i]:.2f}%'
    ])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.15, 0.25, 0.3, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)

# Style header row
for i in range(4):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, 6):
    for j in range(4):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#E7E6E6')

plt.title('CIFAR-10 CNN Training Metrics Summary', fontsize=14, fontweight='bold', pad=20)
plt.savefig('cifar10_metrics_table.png', dpi=300, bbox_inches='tight')
print("Saved: cifar10_metrics_table.png")
plt.close()

print("\n✅ All output images generated for presentation!")