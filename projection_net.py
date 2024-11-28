import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class OrthogonalMatrix(nn.Module):
    """
    Creates a differentiable orthogonal matrix using a modified Gram-Schmidt process
    """
    def __init__(self, dim):
        super().__init__()
        # Initialize random matrix
        self.weight = nn.Parameter(torch.randn(dim, dim))
    
    def forward(self):
        # Modified Gram-Schmidt process
        x = self.weight
        q = torch.zeros_like(x)
        
        # First vector is just normalized
        q[:, 0] = x[:, 0] / torch.norm(x[:, 0])
        
        # Rest of the vectors
        for i in range(1, x.shape[1]):
            v = x[:, i]
            # Subtract projections onto previous vectors
            for j in range(i):
                v = v - torch.dot(v, q[:, j]) * q[:, j]
            # Normalize
            q[:, i] = v / torch.norm(v)
            
        return q

class ProjectionNetwork(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.U = OrthogonalMatrix(dim)
        # Initialize D with values between 0 and 1
        self.D = nn.Parameter(torch.rand(dim))
        
    def get_projection_matrix(self):
        U = self.U()
        # Ensure D values are between 0 and 1 using sigmoid
        D = torch.sigmoid(self.D)
        # Create diagonal matrix
        D_matrix = torch.diag(D)
        # Compute U * D * U^T
        P = U @ D_matrix @ U.t()
        return P
    
    def forward(self, x):
        P = self.get_projection_matrix()
        return P @ x

class EffectiveDimensionalityLearner:
    def __init__(self, dim, target_function, learning_rate=0.001):
        """
        Initialize the learner
        Args:
            dim: dimension of the search space
            target_function: function to analyze for effective dimensionality
            learning_rate: learning rate for optimization
        """
        self.dim = dim
        self.target_function = target_function
        self.network = ProjectionNetwork(dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
    def generate_samples(self, num_samples):
        """Generate random samples from the search space"""
        return torch.randn(num_samples, self.dim)
    
    def compute_loss(self, x, k=1.0):
        """
        Compute loss based on difference between f(Px) and f(x + k(I-P)x)
        Args:
            x: input samples
            k: scaling factor for orthogonal complement
        """
        P = self.network.get_projection_matrix()
        I = torch.eye(self.dim)
        
        # Compute projections
        Px = P @ x.t()
        orthogonal_component = k * ((I - P) @ x.t())
        
        # Compute function values
        f_px = torch.tensor([self.target_function(px.numpy()) for px in Px.t()])
        f_full = torch.tensor([self.target_function((px + oc).numpy()) 
                             for px, oc in zip(Px.t(), orthogonal_component.t())])
        
        # Compute loss as mean squared difference
        loss = torch.mean((f_px - f_full) ** 2)
        
        # Add regularization to encourage sparsity in D
        D = torch.sigmoid(self.network.D)
        sparsity_regularization = 0.1 * torch.mean(D)
        
        return loss + sparsity_regularization
    
    def train(self, num_epochs=1000, batch_size=32):
        """Train the network"""
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            
            # Generate batch of samples
            x = self.generate_samples(batch_size)
            
            # Compute and backpropagate loss
            loss = self.compute_loss(x)
            loss.backward()
            self.optimizer.step()
            
            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
    
    def get_effective_dimensions(self, threshold=0.5):
        """Get the effective dimensions based on trained network"""
        D = torch.sigmoid(self.network.D)
        effective_dims = torch.where(D > threshold)[0]
        return effective_dims.numpy()

# Example usage
def example_target_function(x):
    """Example target function that only depends on first two dimensions"""
    return x[0]**2 + x[1]**2

if __name__ == "__main__":
    # Initialize learner
    dim = 10  # Total dimensions
    learner = EffectiveDimensionalityLearner(dim, example_target_function)
    
    # Train the network
    learner.train(num_epochs=1000, batch_size=32)
    
    # Get effective dimensions
    effective_dims = learner.get_effective_dimensions()
    print(f"Effective dimensions: {effective_dims}")
    
    # Get final projection matrix
    P = learner.network.get_projection_matrix()
    print(f"Final projection matrix diagonal: {torch.diag(P)}")