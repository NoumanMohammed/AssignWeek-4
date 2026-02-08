"""
Autograd Exploration with PyTorch
Demonstrates automatic gradient computation
"""

import torch

# Define tensors with requires_grad=True to track computation
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# Define an operation: z = x^2 + y^3
z = x ** 2 + y ** 3

# Compute gradients automatically
z.backward()

# Print gradients
print(f"Gradient of x: {x.grad}")  # Should be 2 * x = 4.0
print(f"Gradient of y: {y.grad}")  # Should be 3 * y^2 = 27.0