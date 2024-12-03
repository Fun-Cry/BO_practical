# test_experimenter.py

import numpy as np
from surrogate_model.gpr_model import GaussianProcessModel
from utils.experimenter import Experimenter

if __name__ == "__main__":
    # Set parameters
    dim_total = 10  # Total input dimensions
    dim_effect = 5  # Effective function dimensions
    num_DoE = 50    # Number of initial design of experiments samples
    num_iters = 10  # Number of iterations for the experimenter
    num_samples = 5  # Number of samples to train on per iteration
    num_epochs = 3   # Number of epochs for each training iteration

    # Instantiate the surrogate model
    surrogate_model = GaussianProcessModel(input_dim=dim_total, learning_rate=0.01)

    # Create the Experimenter instance
    experimenter = Experimenter(
        dim_total=dim_total,
        dim_effect=dim_effect,
        surrogate_model=surrogate_model,
        num_DoE=num_DoE,
        num_iters=num_iters,
        num_samples=num_samples,
        num_epochs=num_epochs
    )

    # Step 1: Initialize the surrogate model
    print("Initializing the surrogate model with Design of Experiments (DoE)...")
    experimenter.initialize_surrogate()
    print("Surrogate model initialization complete.")

    # Step 2: Train the projection network with iterative updates
    print("Training the projection network...")
    experimenter.train()
    print("Training complete.")
