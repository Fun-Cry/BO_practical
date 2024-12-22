import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.optimize import minimize
from utils.experimenter import Experimenter
from surrogate_models.gpr_model import GPRSurrogate
from scipy.stats import norm 

class PBO:
    def __init__(self, budget_multiplier=5):
        self.budget_multiplier = budget_multiplier

    def train_pnet(self, problem):
        surrogate_model = GPRSurrogate()
        experiment_config = {
            'num_DoE': problem.meta_data.n_variables * 5,
            'num_iters': 10,
            'num_samples': 100,
            'num_epochs': 200,
            'lr': 1e-2
        }
        experimenter = Experimenter(
            dim_total=problem.meta_data.n_variables,
            toy=False,
            function=problem,
            surrogate_model=surrogate_model,
            **experiment_config
        )
        experimenter.initialize_surrogate()
        experimenter.train()
        _, found_dim = experimenter.principal_angle()
        DoE = (experimenter.samples, experimenter.observations)
        return found_dim, DoE, experimenter.pnet

    def __call__(self, problem):
        found_dim, DoE, pnet = self.train_pnet(problem)
        X_high, y = DoE
        X_low = pnet.higher_to_lower(X_high)

        kernel = Matern(nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)
        gp.fit(X_low, y)

        def acquisition(x, gp, y_min):
            mean, std = gp.predict(x.reshape(1, -1), return_std=True)
            z = (y_min - mean) / std
            ei = (y_min - mean) * norm.cdf(z) + std * norm.pdf(z)
            return -ei

        budget = problem.meta_data.n_variables * self.budget_multiplier
        for _ in range(budget):
            res = minimize(acquisition, x0=np.zeros(found_dim), args=(gp, y.min()),
                           bounds=[(-1, 1)] * found_dim)
            x_next_low = res.x
            x_next_high = pnet.lower_to_higher(x_next_low)
            y_next = problem(x_next_high)
            X_low = np.vstack((X_low, x_next_low))
            y = np.append(y, y_next)
            gp.fit(X_low, y)
