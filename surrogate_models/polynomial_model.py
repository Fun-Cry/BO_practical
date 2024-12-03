import torch
import numpy as np
from utils.random_function import random_function
from sklearn.preprocessing import PolynomialFeatures

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

# Polynomial feature expansion
poly_degree = 2  # Degree of the polynomial
poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
X_poly = torch.tensor(poly.fit_transform(X.numpy()), dtype=torch.float)

# Define a simple linear regression model
class PolynomialRegression(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)

# Instantiate the model
input_dim = X_poly.shape[1]
model = PolynomialRegression(input_dim)

# Loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Training loop
epochs = 500
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_poly)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Switch to evaluation mode
model.eval()

# Generate test inputs with the same dimensionality as training inputs
num_test_samples = 100
test_X = np.random.rand(num_test_samples, D)
test_X_poly = torch.tensor(poly.transform(test_X), dtype=torch.float)

# Make predictions
with torch.no_grad():
    predicted_mean = model(test_X_poly)

print("Predicted Mean:", predicted_mean)
