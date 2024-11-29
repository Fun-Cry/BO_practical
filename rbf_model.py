import torch
import torch.nn as nn
import torch.optim as optim
from utils.random_function import random_function
import numpy as np

class RBFNet(nn.Module):
    def __init__(self, num_centers, input_dim, gamma=1.0):
        super(RBFNet, self).__init__()
        self.num_centers = num_centers
        self.input_dim = input_dim
        self.gamma = gamma

        # Centers of the RBFs (learnable parameters)
        self.centers = nn.Parameter(torch.randn(num_centers, input_dim))
        # Weights for the output layer
        self.weights = nn.Parameter(torch.randn(num_centers, 1))

    def rbf(self, x, center):
        # Compute the RBF activation (Gaussian)
        return torch.exp(-self.gamma * torch.sum((x - center) ** 2, dim=-1))

    def forward(self, x):
        # Compute RBF activations for each center
        activations = torch.stack([self.rbf(x, c) for c in self.centers], dim=1)
        # Compute weighted sum of RBF activations
        return activations @ self.weights

# Example Usage
num_samples = 100
input_dim = 1
num_centers = 10
d = 5
D = 10
func = random_function(D, d)


# Generate some synthetic data
X = np.random.rand(num_samples, D)  # Random inputs in high-dimensional space
y = np.array([func(x) for x in X])  # Evaluate the random function

# Convert to PyTorch tensors
# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y, dtype=torch.float).view(-1, 1)  # Ensure y is 2D

# Initialize the RBF network
rbf_net = RBFNet(num_centers=num_centers, input_dim=input_dim, gamma=10.0)

# Define the optimizer and loss function
optimizer = optim.Adam(rbf_net.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Train the RBF network
epochs = 500
for epoch in range(epochs):
    optimizer.zero_grad()
    predictions = rbf_net(X)
    loss = loss_fn(predictions, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Test the trained model
with torch.no_grad():
    predictions = rbf_net(X)

#print(predictions)
