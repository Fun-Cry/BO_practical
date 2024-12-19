import torch
import numpy as np
from utils.random_function import random_function

class DifferentiablePolynomialFeatures(torch.nn.Module):
    def __init__(self, degree):
        super().__init__()
        self.degree = degree

    def forward(self, X):
        poly_features = [X]
        for deg in range(2, self.degree + 1):
            poly_features.append(X ** deg)
        return torch.cat(poly_features, dim=-1)


class PolynomialRegressionModel:
    def __init__(self, input_dim, poly_degree=2, learning_rate=0.1, epochs=1000):
        self.poly_degree = poly_degree
        self.poly_transform = DifferentiablePolynomialFeatures(degree=self.poly_degree)
        self._poly_input_dim = input_dim * self.poly_degree  # Approximation for transformed dim

        # Initialize model
        self.model = torch.nn.Linear(self._poly_input_dim, 1)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.epochs = epochs

    def fit(self, X, y, verbose=True):
        # Convert data to tensors
        X_tensor = torch.tensor(X, dtype=torch.float)
        y_tensor = torch.tensor(y, dtype=torch.float).unsqueeze(-1)

        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()

            # Apply differentiable polynomial transformation
            X_poly = self.poly_transform(X_tensor)
            predictions = self.model(X_poly)

            # Compute loss and update
            loss = self.criterion(predictions, y_tensor)
            loss.backward()
            self.optimizer.step()

            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")

    def predict(self, X, requires_grad=True):
        # Convert data to tensors
        X_tensor = torch.tensor(X, dtype=torch.float, requires_grad=requires_grad)

        # Apply differentiable polynomial transformation
        X_poly = self.poly_transform(X_tensor)

        # Perform prediction
        self.model.eval()
        predictions = self.model(X_poly).squeeze(-1)

        return predictions

if __name__ == "__main__":
    # Example usage
    D = 10  # Input dimensionality
    d = 5  # True function dimensionality
    num_samples = 100
    func = random_function(D, d)

    # Generate random data
    X = np.random.rand(num_samples, D)
    y = np.array([func(x) for x in X]) # type: ignore

    # Instantiate and train the model
    model = PolynomialRegressionModel(input_dim=X.shape[1], poly_degree=2, learning_rate=0.1)
    model.fit(X, y, epochs=500)

    # Predict on test data
    num_test_samples = 100
    test_X = np.random.rand(num_test_samples, D)

    # Obtain predictions with differentiability
    predicted_mean = model.predict(test_X, requires_grad=True)

    print("Predicted Mean:", predicted_mean)
