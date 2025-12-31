"""
Hyperparameter tuning for MLP on out4 configuration
Compare MLP with linear and lasso methods

Based on grr_out4 from simulation-GRR.ipynb:
- n=2000, avg_degree=10, max_degree=19, k=2
- treated_fraction=1/2
- d1=0, d2=1 (comparing untreated vs directly treated)
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
from datetime import datetime

# Import from project modules
from dgp import (
    generate_graph,
    generate_covariates,
    generate_potential_outcomes,
    assign_treatment,
    compute_exposure,
    observed_outcome
)
from propensity import monte_carlo_probabilities
from GRR import grr_estimator

# =============================================================================
# Data generation parameters (matching grr_out4 from simulation-GRR.ipynb)
# =============================================================================
n = 2000
avg_degree = 10
max_degree = 19
k = 2
d1, d2 = 0, 1
treated_fraction = 1/2
mc_reps = 5000
seed_mc = 42
seed_graph = 42
seed_cov = 42
seed_out = 42
B_tune = 20  # Number of MC reps for tuning
eps = 1e-12

# =============================================================================
# Generate data once
# =============================================================================
print("=" * 80)
print("MLP Hyperparameter Tuning for out4 Configuration")
print("=" * 80)
print("\nGenerating data...")

G, H = generate_graph(n=n, avg_degree=avg_degree, max_degree=max_degree, seed=seed_graph)
X, X_noi = generate_covariates(H, seed=seed_cov)
Y_pot = generate_potential_outcomes(H, X_noi, k=k, seed=seed_out)
true_tau = float(np.mean(Y_pot[:, d1] - Y_pot[:, d2]))
pi_id_all, _, _, _ = monte_carlo_probabilities(G, treated_fraction, k, mc_reps, seed_mc, d1, d2)
Kn = (G @ G.T > 0).astype(int)

print(f"  n={n}, avg_degree={avg_degree}, k={k}")
print(f"  d1={d1}, d2={d2}, treated_fraction={treated_fraction}")
print(f"  True tau: {true_tau:.6f}")
print(f"  B_tune: {B_tune} MC repetitions per config")

# =============================================================================
# Helper function to evaluate any method
# =============================================================================
def evaluate_method(method, n_reps=B_tune, **kwargs):
    """Run a GRR method and return bias, sd, rmse, and mean loss."""
    results = []
    losses = []
    
    np.random.seed(42)  # For reproducibility across configs
    for _ in range(n_reps):
        A = assign_treatment(n, treated_fraction=treated_fraction, seed=None)
        D = compute_exposure(G, A, k)
        Y = observed_outcome(Y_pot, D, k=k)
        
        try:
            res = grr_estimator(
                Y, D, X, d1, d2, pi_id_all, Kn,
                method=method,
                eps=eps,
                **kwargs
            )
            results.append(res['tau_hat'])
            losses.append(res['loss'])
        except Exception as e:
            results.append(np.nan)
            losses.append(np.nan)
    
    results = np.array(results)
    valid = ~np.isnan(results)
    if valid.sum() > 0:
        bias = np.mean(results[valid]) - true_tau
        sd = np.std(results[valid], ddof=1) if valid.sum() > 1 else 0
        rmse = np.sqrt(bias**2 + sd**2)
        mean_loss = np.nanmean(losses)
        return {'bias': bias, 'sd': sd, 'rmse': rmse, 'loss': mean_loss, 'n_valid': int(valid.sum())}
    return {'bias': np.nan, 'sd': np.nan, 'rmse': np.nan, 'loss': np.nan, 'n_valid': 0}

# =============================================================================
# First: Run baseline methods (linear and lasso) for comparison
# =============================================================================
print("\n" + "=" * 80)
print("BASELINE METHODS (for comparison)")
print("=" * 80)

baseline_results = {}

# Linear (OLS)
print("\nRunning linear (OLS)...", end=" ")
baseline_results['linear'] = evaluate_method('linear')
print(f"RMSE={baseline_results['linear']['rmse']:.4f}, Bias={baseline_results['linear']['bias']:.4f}")

# Lasso
print("Running lasso...", end=" ")
baseline_results['lasso'] = evaluate_method('lasso', alpha=0.01, l1_ratio=1.0)
print(f"RMSE={baseline_results['lasso']['rmse']:.4f}, Bias={baseline_results['lasso']['bias']:.4f}")

print("\nBaseline Summary:")
print(f"  {'Method':<10} {'Bias':>10} {'SD':>10} {'RMSE':>10} {'Loss':>15}")
print("  " + "-" * 60)
for method, stats in baseline_results.items():
    print(f"  {method:<10} {stats['bias']:>10.4f} {stats['sd']:>10.4f} {stats['rmse']:>10.4f} {stats['loss']:>15.2f}")

# =============================================================================
# MLP Hyperparameter Grid
# =============================================================================
param_grid = {
    'hidden_dims': [[64, 32], [128, 64], [64, 32, 16], [32, 16]],
    'lr': [0.01, 0.001, 0.0005],
    'dropout': [0.0, 0.1, 0.2],
    'weight_decay': [0.0, 0.001],
}

n_configs = (len(param_grid['hidden_dims']) * len(param_grid['lr']) * 
             len(param_grid['dropout']) * len(param_grid['weight_decay']))

# =============================================================================
# Run MLP grid search
# =============================================================================
print("\n" + "=" * 80)
print(f"MLP GRID SEARCH ({n_configs} configurations)")
print("=" * 80)

tuning_results = []
config_num = 0

for hidden_dims in param_grid['hidden_dims']:
    for lr in param_grid['lr']:
        for dropout in param_grid['dropout']:
            for weight_decay in param_grid['weight_decay']:
                config_num += 1
                config = {
                    'hidden_dims': hidden_dims,
                    'lr': lr,
                    'dropout': dropout,
                    'weight_decay': weight_decay
                }
                
                print(f"[{config_num:2d}/{n_configs}] dims={str(hidden_dims):<15} lr={lr:<6} "
                      f"drop={dropout:<3} wd={weight_decay:<5}...", end=" ")
                
                result = evaluate_method(
                    'mlp',
                    hidden_dims=hidden_dims,
                    n_epochs=500,
                    lr=lr,
                    weight_decay=weight_decay,
                    dropout=dropout,
                    early_stopping=True,
                    patience=50,
                    verbose=False
                )
                
                config.update(result)
                tuning_results.append(config)
                
                if np.isfinite(result['rmse']):
                    print(f"RMSE={result['rmse']:.4f}, Loss={result['loss']:.2f}")
                else:
                    print("FAILED")

# =============================================================================
# Results Analysis
# =============================================================================
tuning_df = pd.DataFrame(tuning_results)
tuning_df = tuning_df.sort_values('rmse')

print("\n" + "=" * 80)
print("TOP 10 MLP CONFIGURATIONS BY RMSE:")
print("=" * 80)
print(tuning_df.head(10).to_string(index=False))

# Best MLP config
best_config = tuning_df.iloc[0]
print("\n" + "=" * 80)
print("BEST MLP CONFIG:")
print("=" * 80)
print(f"  hidden_dims:  {best_config['hidden_dims']}")
print(f"  lr:           {best_config['lr']}")
print(f"  dropout:      {best_config['dropout']}")
print(f"  weight_decay: {best_config['weight_decay']}")
print(f"  RMSE:         {best_config['rmse']:.6f}")
print(f"  Bias:         {best_config['bias']:.6f}")
print(f"  SD:           {best_config['sd']:.6f}")
print(f"  Loss:         {best_config['loss']:.2f}")

# =============================================================================
# Final Comparison: Best MLP vs Baselines
# =============================================================================
print("\n" + "=" * 80)
print("FINAL COMPARISON: Best MLP vs Baseline Methods")
print("=" * 80)
print(f"\n  {'Method':<15} {'Bias':>10} {'SD':>10} {'RMSE':>10} {'Loss':>15}")
print("  " + "-" * 65)
print(f"  {'linear':<15} {baseline_results['linear']['bias']:>10.4f} "
      f"{baseline_results['linear']['sd']:>10.4f} {baseline_results['linear']['rmse']:>10.4f} "
      f"{baseline_results['linear']['loss']:>15.2f}")
print(f"  {'lasso':<15} {baseline_results['lasso']['bias']:>10.4f} "
      f"{baseline_results['lasso']['sd']:>10.4f} {baseline_results['lasso']['rmse']:>10.4f} "
      f"{baseline_results['lasso']['loss']:>15.2f}")
print(f"  {'mlp (best)':<15} {best_config['bias']:>10.4f} "
      f"{best_config['sd']:>10.4f} {best_config['rmse']:>10.4f} "
      f"{best_config['loss']:>15.2f}")

# Determine winner
all_rmse = {
    'linear': baseline_results['linear']['rmse'],
    'lasso': baseline_results['lasso']['rmse'],
    'mlp': best_config['rmse']
}
winner = min(all_rmse, key=all_rmse.get)
print(f"\n  Winner: {winner} (RMSE={all_rmse[winner]:.6f})")

# =============================================================================
# Save results to CSV
# =============================================================================
# Create results directory if it doesn't exist
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

# Generate filename with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = os.path.join(results_dir, f'mlp_tuning_results_out4_{timestamp}.csv')
tuning_df.to_csv(output_file, index=False)
print(f"\nTuning results saved to: {output_file}")

print("\n" + "=" * 80)
print("TUNING COMPLETE")
print("=" * 80)
