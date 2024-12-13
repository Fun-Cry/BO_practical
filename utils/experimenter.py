from utils.random_function import random_function
from pyDOE import lhs
import numpy as np
from utils.projection_net import ProjectionNetwork
import torch

class Experimenter:
    def __init__(self,
                 dim_total,
                 dim_effect,
                 surrogate_model,
                 num_DoE,
                 num_iters,
                 num_samples=64,  # Default to one sample per iteration
                 num_epochs=500    # Default to one epoch per iteration
                 ):
        self.dim_total = dim_total
        self.dim_effect = dim_effect
        self.equation, self.onb, self.function = random_function(dim_total, dim_effect)
        self.surrogate_model = surrogate_model
        self.num_DoE = num_DoE
        self.num_iters = num_iters
        self.num_samples = num_samples
        self.num_epochs = num_epochs
        
        self.pnet = ProjectionNetwork(input_dim=dim_total)
        self.pnet_optimizer = torch.optim.Adam(self.pnet.parameters())
        self.criterion = torch.nn.MSELoss()


    def initialize_surrogate(self):
        """
        Initializes the surrogate model by generating a design of experiments (DoE)
        and fitting the surrogate model to the observations.
        """
        # Generate LHS samples in the unit hypercube
        samples = lhs(self.dim_total, self.num_DoE)
        
        # Convert LHS samples to the range [0, 1]
        samples = np.clip(samples, 0, 1)

        # Evaluate the function for each sample
        observations = np.array([self.function(sample) for sample in samples])

        # Fit the surrogate model using the samples and observations
        self.surrogate_model.fit(samples, observations)

        # Store the initial DoE samples and observations for potential reuse
        self.samples = samples
        self.observations = observations

    def iterate(self):
        """
        Perform one iteration of training the projection network.

        Steps:
        1. Randomly generate new points in the input space.
        2. Evaluate the surrogate model at the generated points.
        3. Use the results to train the projection network (pnet) for a specified number of epochs.
        """
        for _ in range(self.num_epochs):
            # Step 1: Randomly generate new points in the input space
            random_points = np.random.rand(self.num_samples, self.dim_total)  # `num_samples` points in [0, 1]^dim_total

            # Step 2: Evaluate the surrogate model at the random points
            surrogate_values = self.surrogate_model.predict(random_points)  # Surrogate model predictions

            # Step 3: Train the projection network
            self.pnet_optimizer.zero_grad()

            # Convert random_points to PyTorch tensor
            random_points_tensor = torch.tensor(random_points, dtype=torch.float)

            # print(random_points_tensor.shape)
            # Forward pass through the projection network
            embeddings = self.pnet(random_points_tensor)
            pnet_outputs = self.surrogate_model.predict(embeddings)
            

            # assert pnet_outputs_tensor.shape == surrogate_values_tensor.shape, f"p: {pnet_outputs_tensor.shape}, s: {surrogate_values_tensor.shape}"
            # Loss is the squared error between the projection outputs and the surrogate model values
            loss = self.criterion(pnet_outputs, surrogate_values)
            # print(pnet_outputs.shape, surrogate_values_tensor.shape)

            # Backward pass and optimization step
            loss.backward()
            self.pnet_optimizer.step()

            # Print loss (optional)
            print(f"Epoch training loss: {loss.item():.4f}")

    def train(self):
        for _ in range(self.num_iters):
            self.iterate()
            
    def principal_angle(self):
        found_basis = self.pnet.get_basis()
        true_basis = torch.tensor(self.onb, dtype=torch.float)
        
        # Compute the cross-correlation matrix
        C = found_basis.T @ true_basis
        
        # Compute the singular values of the cross-correlation matrix
        # This gives the cosines of the principal angles
        S = torch.linalg.svdvals(C)
        
        # Take arccos of singular values, clamp to avoid numerical instability
        principal_angles_rad = torch.acos(torch.clamp(S, -1, 1))
        
        # Convert to degrees and return the maximum principal angle
        return torch.rad2deg(torch.max(principal_angles_rad)).item()
        
