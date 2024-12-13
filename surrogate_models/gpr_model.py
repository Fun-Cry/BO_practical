import torch
import gpytorch
import numpy as np
from utils.random_function import random_function

class GaussianProcessModel:
    def __init__(self, input_dim, learning_rate=0.1):
        """
        Initializes the Gaussian Process model.

        Args:
            input_dim (int): Dimensionality of the input data.
            learning_rate (float): Learning rate for the optimizer.
        """
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = None
        self.learning_rate = learning_rate
        self.optimizer = None
        self.mll = None

    def fit(self, X, y, epochs=500, verbose=True):
        """
        Train the Gaussian Process model.

        Args:
            X (np.ndarray): Input data of shape (num_samples, input_dim).
            y (np.ndarray): Target data of shape (num_samples,).
            epochs (int): Number of training epochs.
            verbose (bool): Print loss every 50 epochs if True.
        """
        X_tensor = torch.tensor(X, dtype=torch.float)
        y_tensor = torch.tensor(y, dtype=torch.float)

        # Initialize the GP model
        self.model = ExactGPModel(X_tensor, y_tensor, self.likelihood)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        self.model.train()
        self.likelihood.train()

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            output = self.model(X_tensor)
            loss = -self.mll(output, y_tensor)  # Marginal log likelihood
            loss.backward()
            self.optimizer.step()

            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    def predict(self, X, return_std=False, return_confidence_region=False):
        """
        Predict using the trained Gaussian Process model.

        Args:
            X (np.ndarray): Input data of shape (num_samples, input_dim).
            return_std (bool): Whether to return the predictive standard deviation.
            return_confidence_region (bool): Whether to return confidence intervals.

        Returns:
            dict: A dictionary containing the predictions and optionally the uncertainty measures.
        """
        if self.model is None:
            raise ValueError("The model must be fitted before making predictions.")

        X_tensor = torch.tensor(X, dtype=torch.float)

        self.model.eval()
        self.likelihood.eval()
        
        observed_pred = self.likelihood(self.model(X_tensor))
        result = observed_pred.mean
        
        return result



class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

if __name__ == "__main__":
    # Example usage
    D = 10  # Input dimensionality
    d = 5  # True function dimensionality
    num_samples = 100
    func = random_function(D, d)

    # Generate random data
    X = np.random.rand(num_samples, D)
    y = np.array([func(x) for x in X])

    # Instantiate and train the GP model
    gp_model = GaussianProcessModel(input_dim=D, learning_rate=0.1)
    gp_model.fit(X, y, epochs=500)

    # Predict on test data
    num_test_samples = 100
    test_X = np.random.rand(num_test_samples, D)

    # Obtain predictions
    predictions = gp_model.predict(test_X, return_std=True, return_confidence_region=True)

    print("Predicted Mean:", predictions["mean"])
    print("Predicted Standard Deviation:", predictions["std"])
    print("Confidence Region (Lower, Upper):", predictions["confidence_region"])
