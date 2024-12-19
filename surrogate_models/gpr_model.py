import torch
import torch.nn as nn

class GPRSurrogate(nn.Module):
    def __init__(self, learning_rate=1e-2, epochs=500, noise=1e-6):
        super().__init__()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.noise = noise
        self.trained = False

    def rbf_kernel(self, X1, X2, lengthscale=1.0):
        # RBF kernel: K(x,x') = exp(-||x - x'||^2 / (2 * lengthscale^2))
        # X1: N x D, X2: M x D
        # returns N x M kernel matrix
        X1 = X1.unsqueeze(1)  # N x 1 x D
        X2 = X2.unsqueeze(0)  # 1 x M x D
        dist_sq = torch.sum((X1 - X2)**2, dim=2)
        return torch.exp(-dist_sq / (2.0 * lengthscale**2))

    def fit(self, X, y):
        """
        X: N x D (numpy array)
        y: N x 1 (numpy array)
        We'll do a simple GP fit with a fixed kernel and no hyperparameter optimization.
        """
        # Convert to torch
        X_t = torch.tensor(X, dtype=torch.float)
        y_t = torch.tensor(y, dtype=torch.float).view(-1, 1)

        # Compute kernel matrix
        K = self.rbf_kernel(X_t, X_t)
        K = K + self.noise * torch.eye(len(X_t))

        # Compute alpha = K^{-1} y using Cholesky
        L = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(y_t, L)

        # Store training data and alpha as parameters or buffers
        self.register_buffer('X_train', X_t)
        self.register_buffer('alpha', alpha)
        self.trained = True

    def predict(self, X):
        # X: M x D (torch.Tensor)
        # Predictive mean: k(X, X_train)*alpha
        if not self.trained:
            raise RuntimeError("Model not trained yet.")

        K_star = self.rbf_kernel(X, self.X_train)
        pred = K_star @ self.alpha
        return pred
