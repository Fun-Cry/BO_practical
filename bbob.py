from ioh import Experiment, ProblemClass
from utils.PBO import PBO

# Experimental setup
functions = [1, 8, 12, 15, 21]  # Function IDs for BBOB
instances = [1, 2, 3]           # Instance IDs
dimensions = [2, 10, 40, 100]   # Problem dimensions
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
