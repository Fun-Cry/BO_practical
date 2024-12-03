import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.random_function import random_function

class RBFNetwork:
    def __init__(self, num_centers, input_dim, gamma=1.0, learning_rate=0.01):
        """
        Initializes the RBF network.

        Args:
            num_centers (int): Number of RBF centers.
            input_dim (int): Dimensionality of the input data.
            gamma (float): Parameter controlling the width of the RBFs.
            learning_rate (float): Learning rate for the optimizer.
        """
        self.model = RBFNet(num_centers, input_dim, gamma)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def fit(self, X, y, epochs=500, verbose=True):
        """
        Train the RBF network.

        Args:
            X (np.ndarray): Input data of shape (num_samples, input_dim).
            y (np.ndarray): Target data of shape (num_samples,).
            epochs (int): Number of training epochs.
            verbose (bool): Print loss every 50 epochs if True.
        """
        X_tensor = torch.tensor(X, dtype=torch.float)
        y_tensor = torch.tensor(y, dtype=torch.float).view(-1, 1)  # Ensure y is 2D

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            predictions = self.model(X_tensor)
            loss = self.loss_fn(predictions, y_tensor)
            loss.backward()
            self.optimizer.step()

            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    def predict(self, X, requires_grad=False):
        """
        Predict using the trained RBF network.

        Args:
            X (np.ndarray): Input data of shape (num_samples, input_dim).
            requires_grad (bool): Whether to compute gradients for predictions.

        Returns:
            torch.Tensor: Predictions as a differentiable tensor.
        """
        X_tensor = torch.tensor(X, dtype=torch.float, requires_grad=requires_grad)

        self.model.eval()
        with torch.set_grad_enabled(requires_grad):
            predictions = self.model(X_tensor)

        return predictions


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


# Example usage
if __name__ == "__main__":
    num_samples = 100
    input_dim = 10
    num_centers = 10
    d = 5
    D = 10
    func = random_function(D, d)

    # Generate some synthetic data
    X = np.random.rand(num_samples, D)
    y = np.array([func(x) for x in X])

    # Instantiate and train the RBF model
    rbf_model = RBFNetwork(num_centers=num_centers, input_dim=input_dim, gamma=10.0, learning_rate=0.01)
    rbf_model.fit(X, y, epochs=500)

    # Predict on the training data
    predictions = rbf_model.predict(X)

    print("Predictions:", predictions)
