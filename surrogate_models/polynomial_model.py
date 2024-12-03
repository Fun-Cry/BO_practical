import torch
import numpy as np
from utils.random_function import random_function
from sklearn.preprocessing import PolynomialFeatures

class PolynomialRegressionModel:
    def __init__(self, input_dim, poly_degree=2, learning_rate=0.1):
        self.poly_degree = poly_degree
        self.poly = PolynomialFeatures(degree=self.poly_degree, include_bias=False)
        self.model = torch.nn.Linear(input_dim, 1)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def fit(self, X, y, epochs=500, verbose=True):
        """
        Train the polynomial regression model.

        Args:
            X (np.ndarray): Input data of shape (num_samples, input_dim).
            y (np.ndarray): Target data of shape (num_samples,).
            epochs (int): Number of training epochs.
            verbose (bool): Print loss every 50 epochs if True.
        """
        X_poly = self._transform_to_poly(X)
        X_tensor = torch.tensor(X_poly, dtype=torch.float)
        y_tensor = torch.tensor(y, dtype=torch.float)

        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            predictions = self.model(X_tensor).squeeze(-1)
            loss = self.criterion(predictions, y_tensor)
            loss.backward()
            self.optimizer.step()

            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    def predict(self, X, requires_grad=False):
        """
        Predict using the trained model.

        Args:
            X (np.ndarray): Input data of shape (num_samples, input_dim).
            requires_grad (bool): Whether to compute gradients for predictions.

        Returns:
            torch.Tensor: Predictions as a differentiable tensor.
        """
        X_poly = self._transform_to_poly(X)
        X_tensor = torch.tensor(X_poly, dtype=torch.float, requires_grad=requires_grad)

        self.model.eval()
        with torch.set_grad_enabled(requires_grad):
            predictions = self.model(X_tensor).squeeze(-1)

        return predictions

    def _transform_to_poly(self, X):
        """
        Transform input data to polynomial features.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Polynomial transformed input data.
        """
        return self.poly.fit_transform(X)

if __name__ == "__main__":
    # Example usage
    D = 10  # Input dimensionality
    d = 5  # True function dimensionality
    num_samples = 100
    func = random_function(D, d)

    # Generate random data
    X = np.random.rand(num_samples, D)
    y = np.array([func(x) for x in X])

    # Instantiate and train the model
    model = PolynomialRegressionModel(input_dim=X.shape[1], poly_degree=2, learning_rate=0.1)
    model.fit(X, y, epochs=500)

    # Predict on test data
    num_test_samples = 100
    test_X = np.random.rand(num_test_samples, D)

    # Obtain predictions with differentiability
    predicted_mean = model.predict(test_X, requires_grad=True)

    print("Predicted Mean:", predicted_mean)
