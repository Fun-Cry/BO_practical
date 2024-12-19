import torch
import torch.nn as nn

class RBFNetwork(nn.Module):
    def __init__(self, num_centers=10, gamma=10.0, learning_rate=1e-2, epochs=500, input_dim=None):
        super().__init__()
        assert input_dim is not None, "input_dim must be specified for RBFNetwork"
        self.input_dim = input_dim
        self.num_centers = num_centers
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.trained = False

    def _rbf_features(self, X, centers):
        # X: N x D, centers: C x D
        # Phi_ij = exp(-gamma * ||x_i - c_j||^2)
        X = X.unsqueeze(1)     # N x 1 x D
        C = centers.unsqueeze(0)  # 1 x C x D
        dist_sq = torch.sum((X - C)**2, dim=-1)  # N x C
        return torch.exp(-self.gamma * dist_sq)   # N x C

    def fit(self, X, y):
        # Randomly pick centers from data or use k-means for center initialization
        # For simplicity, pick random subset as centers
        # Then solve a linear least squares for output weights
        X_t = torch.tensor(X, dtype=torch.float)
        y_t = torch.tensor(y, dtype=torch.float).view(-1, 1)

        # Select centers randomly from data
        idx = torch.randperm(X_t.size(0))[:self.num_centers]
        centers = X_t[idx]

        # Compute RBF features
        Phi = self._rbf_features(X_t, centers)  # N x C

        # Solve linear system: w = (Phi^T Phi)^{-1} Phi^T y
        A = Phi.T @ Phi
        b = Phi.T @ y_t
        w = torch.linalg.solve(A, b)  # C x 1

        # Register buffers (no further training)
        self.register_buffer('centers', centers)
        self.register_buffer('weights', w)
        self.trained = True

    def predict(self, X):
        # X: M x D (torch.Tensor)
        if not self.trained:
            raise RuntimeError("Model not trained yet.")
        Phi = self._rbf_features(X, self.centers)
        return Phi @ self.weights
