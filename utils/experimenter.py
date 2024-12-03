from utils.random_function import random_function
from pyDOE import lhs
import numpy as np
from projection_net import ProjectionNetwork
import torch

class Experimenter:
    def __init__(self,
                 dim_total,
                 dim_effect,
                 surrogate_model,
                 num_DoE,
                 num_iters
                 ):
        self.dim_total = dim_total
        self.dim_effect = dim_effect
        self.equation, self.function = random_function(dim_total, dim_effect)
        self.surrogate_model = surrogate_model
        self.num_DoE = num_DoE
        self.num_iters = num_iters
        
        self.pnet = ProjectionNetwork(input_dim=dim_total)
    
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
        1. Randomly generate a new point in the input space.
        2. Evaluate the surrogate model at the generated point.
        3. Use the result to train the projection network (pnet).
        """
        # Step 1: Randomly generate a new point in the input space
        random_point = np.random.rand(1, self.dim_total)  # A single random point in [0, 1]^dim_total

        # Step 2: Evaluate the surrogate model at the random point
        surrogate_value = self.surrogate_model.predict(random_point)  # Surrogate model prediction

        # Step 3: Train the projection network
        self.pnet_optimizer.zero_grad()

        # Convert random_point to PyTorch tensor
        random_point_tensor = torch.tensor(random_point, dtype=torch.float)

        # Forward pass through the projection network
        pnet_output = self.pnet(random_point_tensor)

        # Loss is the squared error between the projection output and the surrogate model value
        surrogate_value_tensor = torch.tensor(surrogate_value, dtype=torch.float).view(-1, 1)
        loss = torch.mean((pnet_output - surrogate_value_tensor) ** 2)

        # Backward pass and optimization step
        loss.backward()
        self.pnet_optimizer.step()

        # Print loss (optional)
        print(f"Iteration training loss: {loss.item():.4f}")
        
    def train(self):
        for _ in range(self.num_iters):
            self.iterate()
        
    