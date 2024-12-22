import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import orthogonal
import numpy as np

class OrthogonalParameterWithSoftRounding(nn.Module):
    def __init__(self, D):
        super(OrthogonalParameterWithSoftRounding, self).__init__()
        # Create a linear layer to apply orthogonal parametrization
        self.U_layer = nn.Linear(D, D, bias=False)
        self.U_layer = orthogonal(self.U_layer)
        self.D = D
        # Initialize a D-dimensional array for the diagonal
        self.diag = nn.Parameter(torch.randn(D))

    def forward(self, x):
        # Apply sigmoid to get values between 0 and 1
        diag_soft = torch.sigmoid(self.diag)

        # Increase sharpness for more aggressive rounding
        # First stage of sharpening
        sharpness1 = 50.0
        diag_sharp = torch.sigmoid(sharpness1 * (diag_soft - 0.5))

        # Second stage of sharpening (apply again to push even closer to 0 or 1)
        sharpness2 = 50.0
        diag_rounded = torch.sigmoid(sharpness2 * (diag_sharp - 0.5))

        # Construct the diagonal matrix A
        A = torch.diag(diag_rounded)

        # Get the orthogonal matrix U
        U = self.U_layer.weight

        # Compute UAU*
        UA = U @ A
        UAU_star = UA @ U.T

        # Apply the projection matrix UAU* to the input vector
        return x @ UAU_star.T
    
    def get_basis(self):
        # Hard threshold: strictly binarize based on the current value
        diag_binary = (torch.sigmoid(self.diag) > 0.6).float()
        
        # Find indices of non-zero columns
        non_zero_indices = torch.nonzero(diag_binary).squeeze()
        if non_zero_indices.numel() < 2:  # if fewer than two non-zero columns
            # Sort the diagonal in descending order
            sorted_indices = torch.argsort(self.diag, descending=True)
            # Select the top two indices
            top_two_indices = sorted_indices[:2]
            non_zero_indices = top_two_indices

        # Return only the non-zero columns of U
        return self.U_layer.weight[:, non_zero_indices]


class ProjectionNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ProjectionNetwork, self).__init__()
        self.projection_layer = OrthogonalParameterWithSoftRounding(input_dim)

    def forward(self, x):
        return self.projection_layer(x)
    
    def complement_project(self, x):
        return x - self.projection_layer(x)
    
    def get_basis(self):
        return self.projection_layer.get_basis()

    def lower_to_higher(self, x):
        # Get the basis vectors
        basis = self.get_basis().detach().numpy()
        if len(x.shape) == 1:
            x = np.expand_dims(x, 0)
        # Check if the input matches the basis dimension
        if x.shape[1] != basis.shape[1]:
            raise ValueError(f"Input dimension {x.shape[1]} does not match basis dimension {basis.shape[1]}.")

        # Transform the lower-dimensional input to the problem space
        return x @ basis.T
    
    def higher_to_lower(self, x):
        # Get the basis vectors
        basis = self.get_basis().detach().numpy()
        # Check if the input matches the basis dimension
        if len(x.shape) == 1:
            x = np.expand_dims(x, 0)
        if x.shape[1] != basis.shape[0]:
            raise ValueError(f"Input dimension {x.shape[1]} does not match basis dimension {basis.shape[1]}.")

        # Transform the lower-dimensional input to the problem space
        return x @ basis
    


if __name__ == "__main__":
    # Example usage
    D = 4  # Dimension of the matrix and vector
    model = ProjectionNetwork(D)

    # Example input vector in the original space
    x = torch.randn(1, D)

    # Forward pass
    output = model(x)
    print("Input:", x)
    print("Output:", output)

    # Example lower-dimensional input vector
    basis = model.get_basis()
    lower_dim_vector = torch.randn(1, basis.shape[1])

    # Expand to problem space
    expanded_vector = model.expand_to_problem_space(lower_dim_vector)
    print("Lower-dimensional vector:", lower_dim_vector)
    print("Expanded vector:", expanded_vector)
