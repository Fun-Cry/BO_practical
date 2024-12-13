import torch
import torch.nn as nn

class OrthogonalParameterWithSoftRounding(nn.Module):
    def __init__(self, D):
        super(OrthogonalParameterWithSoftRounding, self).__init__()
        # Initialize an orthogonal matrix U
        self.U = nn.Parameter(torch.eye(D))  # Start with an identity matrix
        # print(self.U)
        self.D = D
        # Initialize a D-dimensional array for the diagonal
        self.diag = nn.Parameter(torch.randn(D))

    def forward(self, x):
        # Apply sigmoid to make values between 0 and 1
        diag_soft = torch.sigmoid(self.diag)

        # Optional: Make the soft rounding sharper by scaling sigmoid
        sharpness = 10  # Increase for more aggressive rounding
        diag_rounded = torch.sigmoid(sharpness * (diag_soft - 0.5))

        # Construct the diagonal matrix A
        A = torch.diag(diag_rounded)

        # Compute the matrix multiplication UAU*
        UA = self.U @ A
        UAU_star = UA @ self.U.T  # Equivalent to UAU*

        # Apply the projection matrix UAU* to the input vector
        return x @ UAU_star.T
    
    def get_basis(self):
        # Round the diagonal to strictly 0 or 1
        diag_binary = (torch.sigmoid(self.diag) > 0.5).float()
        
        # Find indices of non-zero columns
        non_zero_indices = torch.nonzero(diag_binary).squeeze()
        if non_zero_indices.numel() < 2:  # Fewer than two non-zero columns
            # Sort the diagonal in descending order
            sorted_indices = torch.argsort(self.diag, descending=True)
            
            # Select the top two indices with the largest diagonal values
            top_two_indices = sorted_indices[:2]
            
            # Use the top two indices as the basis
            non_zero_indices = top_two_indices
        # Return only the non-zero columns of U
        return self.U[:, non_zero_indices]


# Example neural network with the custom layer
class ProjectionNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ProjectionNetwork, self).__init__()
        self.projection_layer = OrthogonalParameterWithSoftRounding(input_dim)

    def forward(self, x):
        return self.projection_layer(x)
    
    def get_basis(self):
        return self.projection_layer.get_basis()


if __name__ == "__main__":
    # Example usage
    D = 4  # Dimension of the matrix and vector
    model = ProjectionNetwork(D)

    # Example input vector
    x = torch.randn(1, D)

    # Forward pass
    output = model(x)
    print("Input:", x)
    print("Output:", output)
