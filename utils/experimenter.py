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
        self.toy=toy
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
        samples = np.clip(samples, 0, 1) # type: ignore

        # Evaluate the function for each sample
        observations = np.array([self.function(sample) for sample in samples])

        # Fit the surrogate model using the samples and observations
        self.surrogate_model.fit(samples, observations)

        # Store the initial DoE samples and observations for potential reuse
        self.samples = samples
        self.observations = observations

    def iterate(self):
        """
        Perform one iteration of training the projection network with batching.

        Steps:
        1. Randomly generate new points in the input space.
        2. Evaluate the surrogate model at the generated points.
        3. Train the projection network (pnet) for a specified number of epochs using batches.
        """
        # Step 1: Randomly generate new points in the input space
        random_points = np.random.rand(self.num_samples, self.dim_total)  # `num_samples` points in [0, 1]^dim_total

        # Define batch size
        batch_size = 256  # Assume self.batch_size is defined elsewhere
        num_batches = int(np.ceil(self.num_samples / batch_size))

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0  # To track the loss across batches

            # Shuffle points for each epoch (optional)
            indices = np.random.permutation(self.num_samples)
            random_points_shuffled = random_points[indices]

            for i in range(num_batches):
                # Extract batch
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, self.num_samples)
                batch_points = random_points_shuffled[start_idx:end_idx]

                # Convert batch to tensor
                batch_points_tensor = torch.tensor(batch_points, dtype=torch.float)

                # Compute embeddings and complements for the batch
                embeddings = self.pnet(batch_points_tensor)
                complements = self.pnet.complement_project(batch_points_tensor)

                rand_k = torch.rand(len(batch_points), 1, dtype=torch.float)  # Shape (batch_size, 1)
                embeddings_plus_complements = embeddings + complements * rand_k

                # Step 2: Evaluate the surrogate model at the random points in the batch
                embeddings_values = self.surrogate_model.predict(embeddings)
                complements_values = self.surrogate_model.predict(embeddings_plus_complements)

                # Compute loss for the batch
                loss = self.criterion(embeddings_values, complements_values)
                epoch_loss += loss.item()

                # Backward pass and optimization step
                self.pnet_optimizer.zero_grad()
                loss.backward()
                self.pnet_optimizer.step()

            # Print loss for the epoch
            print(f"Epoch {epoch + 1}/{self.num_epochs} training loss: {epoch_loss / num_batches:.4f}")


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
        
        # Take arccos of singular values, clamp to avoid numerical instability
        principal_angles_rad = torch.acos(torch.clamp(S, -1, 1))
        
        # Convert to degrees and return the maximum principal angle
        return torch.rad2deg(torch.max(principal_angles_rad)).item(), found_dim
        
