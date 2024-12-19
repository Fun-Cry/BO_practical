from experiment import Experimenter
import numpy as np
import itertools
from surrogate_models.gpr_model import GaussianProcessModel
from surrogate_models.polynomial_model import PolynomialRegressionModel
from surrogate_models.rbf_model import RBFNetwork
from utils.experimenter import Experimenter
import random
import cocoex

# Initialize a suite
suite_name = "bbob"  # You can also use "bbob-biobj" for multi-objective problems
suite = cocoex.Suite(suite_name, "", "")

problem_index = 23
problem = suite.get_problem(problem_index)

dim_total = 20
dim_effect = 5
surrogate_model_class = GaussianProcessModel(input_dim=problem.dimension)

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
        function=problem
    )

    # Run the experiment
experimenter.initialize_surrogate()
experimenter.train()

principal_angle, found_dim = experimenter.principal_angle()