import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from surrogate_models.gpr_model import GPRSurrogate
from surrogate_models.polynomial_model import PolynomialRegressionModel
from surrogate_models.rbf_model import RBFNetwork
from utils.experimenter import Experimenter
import seaborn as sns
from datetime import datetime
import os

# Set random seeds for reproducibility
np.random.seed(21)
torch.manual_seed(21)

def run_single_experiment(
    dim_total,
    dim_effect,
    surrogate_model_class,
    surrogate_config,
    experiment_config
):
    """Run a single experiment with given configuration"""
    # Initialize surrogate model
    surrogate_model = surrogate_model_class(**surrogate_config)
    
    # Create experimenter instance
    experimenter = Experimenter(
        dim_total=dim_total,
        dim_effect=dim_effect,
        surrogate_model=surrogate_model,
        **experiment_config
    )
    
    # Run experiment
    experimenter.initialize_surrogate()
    experimenter.train()
    
    # Get results
    principal_angle, found_dim = experimenter.principal_angle()
    
    return {
        'dim_total': dim_total,
        'dim_effect': dim_effect,
        'surrogate_model': surrogate_model_class.__name__,
        'principal_angle': principal_angle,
        'found_dim': found_dim
    }

def run_experiments(configs, num_runs=3):
    """Run multiple experiments with different configurations"""
    all_results = []
    
    total_experiments = len(configs) * num_runs
    experiment_count = 0
    
    for config in configs:
        for run in range(num_runs):
            experiment_count += 1
            print(f"\nRunning experiment {experiment_count}/{total_experiments}")
            print(f"Configuration: {config}")
            print(f"Run: {run + 1}/{num_runs}")
            
            # try:
            result = run_single_experiment(**config)
            result['run'] = run
            all_results.append(result)
            
            print(f"Results - Principal Angle: {result['principal_angle']:.4f}, "
                    f"Found Dimensions: {result['found_dim']}")
            # except Exception as e:
            #     print(f"Error in experiment: {str(e)}")
            #     continue
    
    return all_results

def visualize_results(results):
    """Create visualizations of experiment results"""
    df = pd.DataFrame(results)
    
    # Create output directory for plots
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'experiment_results_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Principal angle vs effective dimensions
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='dim_effect', y='principal_angle', hue='surrogate_model')
    plt.title('Principal Angle vs Effective Dimensions')
    plt.xlabel('Effective Dimensions')
    plt.ylabel('Principal Angle (degrees)')
    plt.savefig(f'{output_dir}/principal_angle_vs_dim.png')
    plt.close()
    
    # 2. Found dimensions accuracy
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='dim_effect', y='found_dim', hue='surrogate_model')
    plt.title('Found vs Actual Effective Dimensions')
    plt.xlabel('True Effective Dimensions')
    plt.ylabel('Found Dimensions')
    plt.savefig(f'{output_dir}/found_dimensions.png')
    plt.close()
    
    # 3. Summary statistics
    summary = df.groupby(['surrogate_model', 'dim_total', 'dim_effect'])[['principal_angle', 'found_dim']].agg(['mean', 'std'])
    summary.to_csv(f'{output_dir}/summary_statistics.csv')
    
    return summary

if __name__ == "__main__":
    # Define experiment configurations
    dimension_pairs = [
        # (10, 2),
        (20, 5),
        # (50, 5),
        # (100, 10)
    ]
    
    # Define model configurations
    model_configs = {
        GPRSurrogate: {
            'learning_rate': 1e-2,
            'epochs': 100
        },
        # PolynomialRegressionModel: {
        #     'poly_degree': 2,
        #     'learning_rate': 1e-2,
        #     'epochs': 500
        # },
        # RBFNetwork: {
        #     'num_centers': 10,
        #     'gamma': 10.0,
        #     'learning_rate': 1e-2,
        #     'epochs': 100
        # }
    }
    
    # Define experimenter configurations
    experiment_base_config = {
        'num_DoE': 100,
        'num_iters': 10,
        'num_samples': 100,
        'num_epochs': 100,
        'lr': 1e-2
    }
    
    # Generate all configurations
    configs = []
    for dim_total, dim_effect in dimension_pairs:
        for model_class, model_config in model_configs.items():
            surrogate_config = model_config.copy()
            if model_class == PolynomialRegressionModel:
                surrogate_config['input_dim'] = dim_total
            if model_class == RBFNetwork:
                surrogate_config['input_dim'] = dim_total
                
            configs.append({
                'dim_total': dim_total,
                'dim_effect': dim_effect,
                'surrogate_model_class': model_class,
                'surrogate_config': surrogate_config,
                'experiment_config': experiment_base_config
            })
    
    # Run experiments
    print("Starting experiments...")
    results = run_experiments(configs, num_runs=1)
    
    # Analyze and visualize results
    print("\nAnalyzing results...")
    summary = visualize_results(results)
    
    print("\nSummary of results:")
    print(summary)