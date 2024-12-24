from experiment import Experimenter
import numpy as np
import itertools
from surrogate_models.gpr_model import GPRSurrogate
from surrogate_models.polynomial_model import PolynomialRegressionModel
from surrogate_models.rbf_model import RBFNetwork
from utils.experimenter import Experimenter
import random
import cocoex

# Initialize a suite
suite_name = "bbob" 
suite = cocoex.Suite("bbob", "", "")

desired_dimension = 40

# Select a specific problem
selected_problem = None

for problem in suite:
    if problem.dimension == desired_dimension:
        selected_problem = problem
        print(f"Selected Problem ID: {problem.id}, Dimension: {problem.dimension}")
        break  # Stop after finding the first match

dim_total = 20
dim_effect = 5
surrogate_model_class = GPRSurrogate()

experimenter = Experimenter(
        dim_total=problem.dimension,
        dim_effect=dim_effect,
        surrogate_model=surrogate_model_class, # type: ignore
        num_DoE=100,
        num_iters=10,
        num_samples=1000,
        num_epochs=1000,
        lr=1e-3, 
        toy=False,
        function=selected_problem
    )

    # Run the experiment
experimenter.initialize_surrogate()
experimenter.train()

principal_angle, found_dim = experimenter.principal_angle()