from utils.random_function import random_function
from pyDOE import lhs
import numpy as np
from utils.projection_net import ProjectionNetwork
import torch

class Experimenter:
    def __init__(self,
                 dim_total,
                 surrogate_model,
                 num_DoE,
                 num_iters,
                 dim_effect=None,
                 num_samples=1000,  # Default to one sample per iteration
                 num_epochs=100,    # Default to one epoch per iteration
                 lr=1e-2,
                 toy=True,
                 function=None
                 ):
        self.toy = toy
        self.dim_total = dim_total
        if self.toy:
            assert dim_effect is not None, 'have to specify effective dimension when doing toy experiment'
            self.dim_effect = dim_effect
            self.equation, self.onb, self.function = random_function(dim_total, dim_effect)
        else:
            assert function is not None, 'have to initialize function if not doing toy experiment'
            self.function = function

        self.surrogate_model = surrogate_model
        self.num_DoE = num_DoE
        self.num_iters = num_iters
        self.num_samples = num_samples
        self.num_epochs = num_epochs

        self.pnet = ProjectionNetwork(input_dim=dim_total)
        self.pnet_optimizer = torch.optim.Adam(self.pnet.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()

    def initialize_surrogate(self):
        """
        Initializes the surrogate model by generating a design of experiments (DoE)
        and fitting the surrogate model to the observations.
        """
        # Generate LHS samples in the unit hypercube
        samples = lhs(self.dim_total, self.num_DoE)

        # Convert LHS samples to the range [0, 1]
        samples = np.clip(samples, 0, 1)  # type: ignore

        # Evaluate the function for each sample
        observations = np.array([self.function(sample) for sample in samples])

        # Fit the surrogate model using the samples and observations
        self.surrogate_model.fit(samples, observations)

        # Freeze surrogate model parameters after training
        for param in self.surrogate_model.parameters():
            param.requires_grad = False

        # Store the initial DoE samples and observations for potential reuse
        self.samples = samples
        self.observations = observations

    def iterate(self):
        """
        Perform one iteration of training the projection network without batching.

        Steps:
        1. Randomly generate new points in the input space.
        2. Evaluate the surrogate model at the generated points.
        3. Train the projection network (pnet) for a specified number of epochs.
        """
        # Step 1: Randomly generate new points in the input space
        random_points = np.random.rand(self.num_samples, self.dim_total)  # `num_samples` points in [0, 1]^dim_total

        # Convert all points to tensor
        random_points_tensor = torch.tensor(random_points, dtype=torch.float)
        rand_k = torch.rand(len(random_points), 1, dtype=torch.float)  # Shape (num_samples, 1)

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0  # To track the loss

            # Compute embeddings and complements for all points
            embeddings = self.pnet(random_points_tensor)
            complements = self.pnet.complement_project(random_points_tensor)

            embeddings_plus_complements = embeddings + complements * rand_k

            # Step 2: Evaluate the surrogate model at the random points
            embeddings_values = self.surrogate_model.predict(embeddings)
            complements_values = self.surrogate_model.predict(embeddings_plus_complements)

            # Compute loss for all points
            loss = self.criterion(embeddings_values, complements_values)
            epoch_loss = loss.item()

            # Backward pass and optimization step
            self.pnet_optimizer.zero_grad()
            loss.backward()
            # for name, param in self.pnet.named_parameters():
            #     if param.grad is not None:
            #         print(f"Grad {name}: {param.grad.norm():.4f}")
            #     else:
            #         print(f"No grad for {name}")

            self.pnet_optimizer.step()

            # Print loss for the epoch
            if (epoch + 1) % 50 == 0 or epoch == self.num_epochs - 1:
                print(f"Epoch {epoch + 1}/{self.num_epochs} training loss: {epoch_loss:.4f}")

    def train(self):
        for _ in range(self.num_iters):
            self.iterate()

    def principal_angle(self):
        found_basis = self.pnet.get_basis()
        found_dim = found_basis.shape[1]

        if not self.toy:
            return None, found_dim
        true_basis = torch.tensor(self.onb, dtype=torch.float)

        # Compute the cross-correlation matrix
        C = found_basis.T @ true_basis

        # Compute the singular values of the cross-correlation matrix
        # This gives the cosines of the principal angles
        S = torch.linalg.svdvals(C)
        # print(S)

        # Take arccos of singular values, clamp to avoid numerical instability
        principal_angles_rad = torch.acos(torch.clamp(S, -1, 1))

        # Convert to degrees and return the maximum principal angle
        return torch.rad2deg(torch.mean(principal_angles_rad)).item(), found_dim
