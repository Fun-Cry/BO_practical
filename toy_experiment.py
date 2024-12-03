from utils.random_function import random_function
from utils.projection_net import ProjectionNetwork
import random
from pyDOE import lhs

dims = [10, 100, 1000]

for dim in dims:
    dim_effect = random.randint(1, int(dim / 2))
    func = random_function(dim, dim_effect)
    