"""
Simulation: PNA Leaf Dimension Experiment

This script tests whether the leaf embedding dimension affects the relative
performance of pna-linear vs pna-no-harm.

Hypothesis: With larger leaf dimensions (more coefficients to estimate),
pna-linear may overfit compared to pna-no-harm (which only estimates 1 coefficient).

Usage:
    python simulation-pna-dim.py
    
    # With nohup (background):
    nohup python simulation-pna-dim.py > results/simulation_pna_dim.log 2>&1 &
"""

import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from simulation_toolkit import run_grr_simulation


# =============================================================================
# Configuration
# =============================================================================

# Only test these two methods
METHODS = ['pna-linear', 'pna-no-harm']

# Leaf dimensions to test (last element of head_hidden_dims)
LEAF_DIMS = [4, 8, 16, 32]

# Base PNA config (will vary head_hidden_dims)
BASE_PNA_CONFIG = {
    'pna_hidden_dims': [64, 32],
    'rep_dim': 32,
    'aggregators': ['mean', 'sum', 'max', 'std'],
    'scalers': ['identity', 'amplification', 'attenuation'],
    'dropout': 0.0,
    'lambda_ipm': 0.0,
    'n_epochs': 300,
    'lr': 0.001,
    'weight_decay': 1e-4,
    'early_stopping': True,
    'patience': 50,
}

# Simulation parameters (from out1)
SIM_PARAMS = {
    'n': 2000,
    'avg_degree': 3.0,
    'max_degree': 9,
    'k': 3,
    'treated_fraction': 1/3,
    'd1': 0,
    'd2': 2,
    'mc_reps': 500,
    'seed_mc': 111,
    'seed_graph': 123,
    'seed_cov': 1234,
    'seed_out': 999,
    'B': 20,  # Number of randomizations per config
    'eps': 1e-12,
    'verbose': False,
}


# =============================================================================
# Main Experiment
# =============================================================================

def run_dimension_experiment():
    """Run experiment varying PNA leaf dimensions."""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_all = []
    
    print("="*70)
    print("PNA Leaf Dimension Experiment")
    print("="*70)
    print(f"Testing leaf dimensions: {LEAF_DIMS}")
    print(f"Methods: {METHODS}")
    print(f"B = {SIM_PARAMS['B']} randomizations per config")
    print("="*70)
    
    for leaf_dim in LEAF_DIMS:
        print(f"\n>>> Testing leaf_dim = {leaf_dim}")
        
        # Create PNA config with this leaf dimension
        # head_hidden_dims ends with the leaf dimension
        pna_config = BASE_PNA_CONFIG.copy()
        pna_config['head_hidden_dims'] = [64, leaf_dim]
        
        # Run simulation
        out = run_grr_simulation(
            **SIM_PARAMS,
            methods=METHODS,
            pna_kwargs=pna_config,
            alpha=0.01,
        )
        
        # Collect results
        for method in METHODS:
            s = out['summary'][method]
            results_all.append({
                'leaf_dim': leaf_dim,
                'method': method,
                'bias': s['bias'],
                'sd': s['sd'],
                'rmse': s['rmse'],
                'n_valid': s['n_valid'],
            })
            print(f"  {method}: RMSE = {s['rmse']:.4f}, bias = {s['bias']:.4f}, sd = {s['sd']:.4f}")
    
    # Convert to DataFrame
    df = pd.DataFrame(results_all)
    
    # ==========================================================================
    # Save results
    # ==========================================================================
    os.makedirs('results', exist_ok=True)
    base_name = f"pna_dim_{timestamp}"
    
    # Save CSV
    csv_path = os.path.join('results', f"{base_name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Save config
    config = {
        'timestamp': timestamp,
        'leaf_dims': LEAF_DIMS,
        'methods': METHODS,
        'base_pna_config': BASE_PNA_CONFIG,
        'sim_params': SIM_PARAMS,
    }
    config_path = os.path.join('results', f"{base_name}_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to: {config_path}")
    
    # ==========================================================================
    # Plot results
    # ==========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Pivot for easier plotting
    df_linear = df[df['method'] == 'pna-linear'].set_index('leaf_dim')
    df_noharm = df[df['method'] == 'pna-no-harm'].set_index('leaf_dim')
    
    # Plot RMSE
    axes[0].plot(LEAF_DIMS, df_linear['rmse'], 'o-', label='pna-linear', linewidth=2, markersize=8)
    axes[0].plot(LEAF_DIMS, df_noharm['rmse'], 's-', label='pna-no-harm', linewidth=2, markersize=8)
    axes[0].set_xlabel('Leaf Embedding Dimension')
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('RMSE vs Leaf Dimension')
    axes[0].legend()
    axes[0].set_xscale('log', base=2)
    axes[0].grid(True, alpha=0.3)
    
    # Plot Bias
    axes[1].plot(LEAF_DIMS, df_linear['bias'], 'o-', label='pna-linear', linewidth=2, markersize=8)
    axes[1].plot(LEAF_DIMS, df_noharm['bias'], 's-', label='pna-no-harm', linewidth=2, markersize=8)
    axes[1].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Leaf Embedding Dimension')
    axes[1].set_ylabel('Bias')
    axes[1].set_title('Bias vs Leaf Dimension')
    axes[1].legend()
    axes[1].set_xscale('log', base=2)
    axes[1].grid(True, alpha=0.3)
    
    # Plot SD
    axes[2].plot(LEAF_DIMS, df_linear['sd'], 'o-', label='pna-linear', linewidth=2, markersize=8)
    axes[2].plot(LEAF_DIMS, df_noharm['sd'], 's-', label='pna-no-harm', linewidth=2, markersize=8)
    axes[2].set_xlabel('Leaf Embedding Dimension')
    axes[2].set_ylabel('Standard Deviation')
    axes[2].set_title('SD vs Leaf Dimension')
    axes[2].legend()
    axes[2].set_xscale('log', base=2)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join('results', f"{base_name}_plot.png")
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to: {plot_path}")
    
    # ==========================================================================
    # Print summary table
    # ==========================================================================
    print("\n" + "="*70)
    print("Summary: RMSE by Leaf Dimension")
    print("="*70)
    print(f"{'Leaf Dim':<10} {'pna-linear':>15} {'pna-no-harm':>15} {'Difference':>15}")
    print("-"*70)
    for dim in LEAF_DIMS:
        rmse_linear = df_linear.loc[dim, 'rmse']
        rmse_noharm = df_noharm.loc[dim, 'rmse']
        diff = rmse_linear - rmse_noharm
        winner = "linear" if diff < 0 else "no-harm" if diff > 0 else "tie"
        print(f"{dim:<10} {rmse_linear:>15.4f} {rmse_noharm:>15.4f} {diff:>+15.4f} ({winner})")
    print("="*70)
    
    return df


if __name__ == "__main__":
    df_results = run_dimension_experiment()

