"""
Simulation Out4: Network Causal Inference Simulation

This script runs the "out4" configuration comparing various treatment effect
estimators. Out4 uses a denser network with higher average degree and binary
exposure (k=2).

Setup:
    - n=2000 nodes
    - avg_degree=10, max_degree=19 (denser network than out1)
    - k=2 exposure levels (binary: 0=untreated, 1=treated)
    - treated_fraction=1/2
    - d1=0 vs d2=1 (comparing untreated vs treated)

Usage:
    python simulation-out4.py
    
    # With nohup (background):
    nohup python simulation-out4.py > results/simulation_out4.log 2>&1 &
"""

from simulation_toolkit import (
    run_grr_simulation,
    print_summary_table,
    save_results
)


# =============================================================================
# Configuration
# =============================================================================

# Methods to compare
ALL_METHODS = [
    # Primitive estimators
    'ht', 'hajek', 
    'haipw-truth', 'haipw-one', 
    'haipw-cov-truth', 'haipw-cov-one',
    # Standard GRR
    'linear', 'lasso',
    # GAT-enhanced GRR (softmax attention - loses some degree info)
    # 'gat-linear', 'gat-lasso',
    # PNA-enhanced GRR (preserves degree info via sum aggregator + degree scalers)
    'pna-linear', 'pna-lasso',
]

# GAT configuration (shallower for binary exposure)
GAT_CONFIG = {
    'gat_hidden_dims': [64, 64, 32],
    'rep_dim': 32,
    'head_hidden_dims': [64, 32],
    'heads': 4,
    'dropout': 0.0,
    'lambda_ipm': 0.0,
    'n_epochs': 400,
    'lr': 0.001,
    'weight_decay': 1e-4,
    'early_stopping': True,
    'patience': 50,
}

# PNA configuration (preserves degree information)
PNA_CONFIG = {
    'pna_hidden_dims': [64, 32],
    'rep_dim': 16,
    'head_hidden_dims': [32, 16],
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


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    
    # Run simulation with out4 parameters (denser network, binary exposure)
    results = run_grr_simulation(
        # Network parameters (denser than out1)
        n=2000, 
        avg_degree=10.0,  # Higher average degree
        max_degree=19, 
        k=2,              # Binary exposure
        treated_fraction=1/2,
        d1=0, d2=1,       # Untreated vs treated
        
        # Monte Carlo parameters
        mc_reps=500,     # More MC reps for better propensity estimation
        seed_mc=42,
        seed_graph=42, 
        seed_cov=42, 
        seed_out=42,
        B=100,  # Number of randomizations (increase to 200+ for final results)
        
        # Methods
        methods=ALL_METHODS,
        eps=1e-12,
        verbose=False,
        
        # GNN embedding configs
        gat_kwargs=GAT_CONFIG,
        pna_kwargs=PNA_CONFIG,
        
        # GRR method parameters
        alpha=0.01,
        l1_ratio=0.5,
    )

    # Print summary table
    print_summary_table(results)
    
    # Save results
    saved_paths = save_results(results, results_dir='results', name_prefix='out4')
