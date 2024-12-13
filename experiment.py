# test_experimenter_comprehensive.py
import numpy as np
import itertools
from surrogate_models.gpr_model import GaussianProcessModel
from surrogate_models.polynomial_model import PolynomialRegressionModel
from surrogate_models.rbf_model import RBFNetwork
from utils.experimenter import Experimenter

def run_experiment(
    dim_total_list,
    dim_effect_list,
    surrogate_models,
    num_runs=5  # Number of independent runs for each configuration
):
    # Store results
    results = []

    # Grid search over parameters
    for dim_total, dim_effect, surrogate_model_class in itertools.product(
        dim_total_list, 
        dim_effect_list, 
        surrogate_models
    ):
        # Ensure effective dimensions are less than total dimensions
        print(dim_total, dim_effect)
        if dim_effect >= dim_total:
            continue

        for run in range(num_runs):
            # Instantiate the surrogate model
            if surrogate_model_class == GaussianProcessModel:
                surrogate_model = surrogate_model_class(input_dim=dim_total, learning_rate=0.01)
            elif surrogate_model_class == PolynomialRegressionModel:
                surrogate_model = surrogate_model_class(input_dim=dim_total, poly_degree=2, learning_rate=0.1)
            elif surrogate_model_class == RBFNetwork:
                surrogate_model = surrogate_model_class(num_centers=10, input_dim=dim_total, gamma=10.0, learning_rate=0.01)

            # Create the Experimenter instance
            experimenter = Experimenter(
                dim_total=dim_total,
                dim_effect=dim_effect,
                surrogate_model=surrogate_model,
                num_DoE=50,
                num_iters=10,
                num_samples=5,
                num_epochs=3
            )

            # Run the experiment
            experimenter.initialize_surrogate()
            experimenter.train()

            # Calculate principal angle
            principal_angle = experimenter.principal_angle()

            # Store results
            results.append({
                'dim_total': dim_total,
                'dim_effect': dim_effect,
                'surrogate_model': surrogate_model_class.__name__,
                'run': run,
                'principal_angle': principal_angle
            })

            print(f"Experiment: total_dim={dim_total}, effect_dim={dim_effect}, "
                  f"model={surrogate_model_class.__name__}, run={run}, "
                  f"principal_angle={principal_angle:.4f}")

    return results

def analyze_results(results):
    # Convert results to a pandas DataFrame for easier analysis
    import pandas as pd
    df = pd.DataFrame(results)

    # Aggregate results
    aggregated = df.groupby(['dim_total', 'dim_effect', 'surrogate_model'])['principal_angle'].agg(['mean', 'std'])
    
    # Plot or further analyze the results
    import matplotlib.pyplot as plt
    
    # Example visualization
    plt.figure(figsize=(12, 6))
    for model in df['surrogate_model'].unique():
        model_data = df[df['surrogate_model'] == model]
        plt.scatter(
            model_data['dim_effect'], 
            model_data['principal_angle'], 
            label=model, 
            alpha=0.6
        )
    
    plt.xlabel('Effective Dimensions')
    plt.ylabel('Principal Angle (degrees)')
    plt.title('Principal Angle vs Effective Dimensions')
    plt.legend()
    plt.show()

    return aggregated

if __name__ == "__main__":
    # Define experimental parameters
    dim_total_list = [5, 10, 15, 20, 25]
    dim_effect_list = [2, 3, 4, 5]
    surrogate_models = [
        # GaussianProcessModel, 
        # PolynomialRegressionModel, 
        RBFNetwork
    ]

    # Run comprehensive experiment
    results = run_experiment(
        dim_total_list, 
        dim_effect_list, 
        surrogate_models
    )

    # Analyze and visualize results
    aggregated_results = analyze_results(results)
    print("\nAggregated Results:")
    print(aggregated_results)