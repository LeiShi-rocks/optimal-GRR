"""
Simulation Out1 - DGP-2: Network Causal Inference Simulation with Neighbor Spillover

This script runs the "out1" configuration comparing various treatment effect
estimators including primitive (HT, Hajek, HAIPW), standard GRR (linear, lasso),
and GNN-enhanced GRR (GAT, PNA) methods.

Uses DGP-2 which includes neighbor covariate spillover effects:
Y_i(d) ~ Normal(beta0 + beta_h*H_i + beta_d*d + g(X_i) + beta_neighbor*mean(neighbor's g(X)), noise_sd^2)

Usage:
    python simulation-out1-dgp2.py
    
    # With nohup (background):
    nohup python simulation-out1-dgp2.py > results/simulation_out1_dgp2.log 2>&1 &
"""

from simulation_toolkit_dgp2 import (
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
    # 'haipw-truth', 'haipw-one', 
    # 'haipw-cov-truth', 'haipw-cov-one',
    # Standard GRR
    'linear', 'lasso',
    # GAT-enhanced GRR (softmax attention - loses some degree info)
    'gat-linear', # 'gat-lasso', 
    'gat-no-harm', 'gat-pred',
    # PNA-enhanced GRR (preserves degree info via sum aggregator + degree scalers)
    'pna-linear', #'pna-lasso', 
    'pna-no-harm', 'pna-pred'
]

# ALL_METHODS = [
#     # Standard GRR
#     'linear', 'pna'
# ]


# GAT configuration (5-layer deep, tuned for network data)
GAT_CONFIG = {
    'gat_hidden_dims': [64, 64, 64, 64, 32],
    'rep_dim': 64,
    'head_hidden_dims': [64, 32],
    'heads': 4,
    'dropout': 0.0,
    'lambda_ipm': 0.0,
    'n_epochs': 600,
    'lr': 0.001,
    'weight_decay': 1e-4,
    'early_stopping': True,
    'patience': 100,
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
    
    # Run simulation
    results = run_grr_simulation(
        # Network parameters
        n=2000, 
        avg_degree=3.0, 
        max_degree=9, 
        k=3,
        treated_fraction=1/3,
        d1=0, d2=2,
        
        # Monte Carlo parameters
        mc_reps=500, 
        seed_mc=111,
        seed_graph=123, 
        seed_cov=1234, 
        seed_out=999,
        B=30,  # Number of randomizations (increase to 200+ for final results)
        
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
    
    # Save results to dgp-2/results directory
    saved_paths = save_results(results, results_dir='dgp-2/results', name_prefix='out1_dgp2')
