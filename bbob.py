from ioh import Experiment, ProblemClass
from utils.PBO import PBO
import numpy as np
import random
import torch

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# Experimental setup
functions = [1, 8, 12, 15, 21]  # Function IDs for BBOB
instances = [0, 1, 2]           # Instance IDs
dimensions = [2, 10, 40]   # Problem dimensions
repetitions = 5                 # Number of repetitions per setup

# Initialize the experiment
experiment = Experiment(
    algorithm=PBO(),
    fids=functions,
    iids=instances,
    dims=dimensions,
    problem_class=ProblemClass.BBOB,
    reps=repetitions
)

# Run the experiment
experiment.run()
