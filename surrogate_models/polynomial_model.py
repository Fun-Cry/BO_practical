import torch
import torch.nn as nn

class PolynomialRegressionModel(nn.Module):
    def __init__(self, poly_degree=2, learning_rate=1e-2, epochs=500, input_dim=None):
        super().__init__()
        assert input_dim is not None, "input_dim must be specified for PolynomialRegressionModel"
        self.poly_degree = poly_degree
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.input_dim = input_dim
        self.trained = False

    def _poly_features(self, X):
        # X: N x D
        # Construct polynomial features up to self.poly_degree
        # For simplicity, we just include terms x^1, x^2, ..., x^poly_degree for each dimension.
        # No cross terms.
        N, D = X.shape
        # features = [X^(d) for d in range(1, poly_degree+1)]
        # Stack them along last dimension
        feats = [X**d for d in range(1, self.poly_degree+1)]
        return torch.cat(feats, dim=1)  # N x (D * poly_degree)

    def fit(self, X, y):
        # X: N x D (numpy)
        # y: N x 1 (numpy)
        X_t = torch.tensor(X, dtype=torch.float)
        y_t = torch.tensor(y, dtype=torch.float).view(-1, 1)

        # Build polynomial features
        Phi = self._poly_features(X_t)  # N x (D * poly_degree)

        # Solve for weights in least squares sense: w = (Phi^T Phi)^{-1} Phi^T y
        # Closed form solution:
        A = Phi.T @ Phi
        b = Phi.T @ y_t
        w = torch.linalg.solve(A, b)  # (D*poly_degree) x 1

        # Register weight
        # No gradient needed for weights after fitting.
        self.register_buffer('weights', w)
        self.trained = True

    def predict(self, X):
        # X: M x D (torch.Tensor)
        if not self.trained:
            raise RuntimeError("Model not trained yet.")
        Phi = self._poly_features(X)
        return Phi @ self.weights
