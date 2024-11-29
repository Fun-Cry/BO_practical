import torch
import gpytorch
from utils.random_function import random_function
import numpy as np

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )  # Radial Basis Function (RBF) kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# Data generation
D = 10
d = 5
num_samples = 100
func = random_function(D, d)

# Convert NumPy arrays to PyTorch tensors
X = np.random.rand(num_samples, D)  # Random inputs in high-dimensional space
y = np.array([func(x) for x in X])  # Evaluate the random function

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y, dtype=torch.float)

# GPyTorch setup
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(X, y, likelihood)

# Optimizer and training setup
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

model.train()
likelihood.train()

# Training loop
epochs = 500
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X)
    loss = -mll(output, y)  # Fix: Use 'y' instead of 'Y'
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Switch to evaluation mode
model.eval()
likelihood.eval()

# Test inputs (convert to PyTorch tensor)
# Generate test inputs with the same dimensionality as training inputs
num_test_samples = 100
test_x = torch.rand(num_test_samples, D)  # Random test points in R^D

# Make predictions
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(test_x))

# Extract predictive mean and confidence intervals
predicted_mean = observed_pred.mean
predicted_std = observed_pred.stddev
lower, upper = observed_pred.confidence_region()

print("Predicted Mean:", predicted_mean)

