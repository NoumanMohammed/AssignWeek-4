"""
Fully Connected Neural Network for MNIST Digit Classification
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ============================================================================
# COMMIT: Loaded and preprocessed MNIST dataset
# ============================================================================

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

# Training loop with metrics tracking
print("\nTraining FCNN on MNIST...")
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

# 1. Accuracy Curve Graph
plt.figure(figsize=(10, 6))
epochs_range = range(1, 6)
plt.plot(epochs_range, train_accuracies, 'b-o', label='Training Accuracy', linewidth=2)
plt.plot(epochs_range, test_accuracies, 'r-s', label='Test Accuracy', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('FCNN Training and Test Accuracy on MNIST', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('mnist_accuracy_curve.png', dpi=300)
print("Saved: mnist_accuracy_curve.png")
plt.close()

# 2. Loss Curve Graph
plt.figure(figsize=(10, 6))
plt.plot(epochs_range, train_losses, 'g-o', label='Training Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('FCNN Training Loss on MNIST', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('mnist_loss_curve.png', dpi=300)
print("Saved: mnist_loss_curve.png")
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

plt.title('MNIST FCNN Training Metrics Summary', fontsize=14, fontweight='bold', pad=20)
plt.savefig('mnist_metrics_table.png', dpi=300, bbox_inches='tight')
print("Saved: mnist_metrics_table.png")
plt.close()

print("\n✅ All output images generated for presentation!")