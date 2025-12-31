"""
GAT-Embedder Tuning Round 2: More aggressive architectures.

Focus on:
1. More attention heads (GAT learns neighbor aggregation patterns)
2. More GAT layers (deeper = more hops of neighborhood info)
3. Higher rep_dim (more capacity to encode degree info)
4. More training epochs
"""

import numpy as np
import torch
from itertools import product
import warnings
warnings.filterwarnings('ignore')

from dgp import (
    generate_graph, generate_covariates, generate_potential_outcomes,
    assign_treatment, compute_exposure, observed_outcome
)

import importlib.util
spec = importlib.util.spec_from_file_location("gat_embedder", "GAT-embedder.py")
gat_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gat_module)
GATOutcomeModel = gat_module.GATOutcomeModel


def evaluate_model(Y_pot_pred, Y_pot_true, Y, D, k, d1, d2):
    mse_all = np.mean((Y_pot_pred - Y_pot_true) ** 2)
    var_y = np.var(Y_pot_true)
    r2 = 1 - mse_all / var_y
    true_tau = np.mean(Y_pot_true[:, d1] - Y_pot_true[:, d2])
    tau_hat = np.mean(Y_pot_pred[:, d1] - Y_pot_pred[:, d2])
    return {'mse_all': mse_all, 'r2': r2, 'tau_bias': tau_hat - true_tau}


def run_gat_model(X_run, A, D, Y, Y_pot_true, k, d1, d2, config, verbose=False):
    try:
        import sys
        from io import StringIO
        
        model = GATOutcomeModel(
            input_dim=X_run.shape[1], k=k,
            gat_hidden_dims=config['gat_hidden_dims'],
            rep_dim=config['rep_dim'],
            head_hidden_dims=config['head_hidden_dims'],
            heads=config['heads'],
            dropout=config['dropout'],
            lambda_ipm=config['lambda_ipm'],
        )
        
        if not verbose:
            old_stdout = sys.stdout
            sys.stdout = StringIO()
        
        model.fit(
            X_run, A, D, Y,
            n_epochs=config.get('n_epochs', 500),
            lr=config.get('lr', 0.001),
            weight_decay=config.get('weight_decay', 1e-4),
            early_stopping=True,
            patience=config.get('patience', 80),
            verbose=False,
        )
        
        if not verbose:
            sys.stdout = old_stdout
        
        Y_pot_pred = model.predict(X_run)
        return evaluate_model(Y_pot_pred, Y_pot_true, Y, D, k, d1, d2)
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    print("=" * 70)
    print("GAT-Embedder Tuning Round 2: More Aggressive Architectures")
    print("=" * 70)
    
    # Generate data (same setup)
    n, avg_degree, max_degree, k = 2000, 3.0, 9, 3
    treated_fraction, d1, d2 = 1/3, 0, 2
    
    G, H = generate_graph(n=n, avg_degree=avg_degree, max_degree=max_degree, seed=123)
    X, X_noi = generate_covariates(H, seed=1234)
    Y_pot_true = generate_potential_outcomes(H, X_noi, k=k, seed=999)
    X_run = X[:, [0, 3]]  # No degree info
    
    A_treat = assign_treatment(n, treated_fraction=treated_fraction, seed=42)
    D = compute_exposure(G, A_treat, k)
    Y = observed_outcome(Y_pot_true, D, k=k)
    A = G.astype(float)
    
    true_tau = np.mean(Y_pot_true[:, d1] - Y_pot_true[:, d2])
    print(f"True tau: {true_tau:.4f}")
    
    # Baselines from Round 1
    print("\nBaselines:")
    print("  Linear (no degree): R²=0.124, tau_bias=0.360")
    print("  Linear (full X):    R²=0.526, tau_bias=0.276")
    print("  Best GAT (Round 1): R²=0.219, tau_bias=0.271")
    
    # =========================================================================
    # Round 2: More aggressive configs
    # =========================================================================
    print("\n" + "=" * 70)
    print("Testing aggressive configurations...")
    print("=" * 70)
    
    configs = []
    
    # 1. More attention heads (8, 16)
    for heads in [8, 16]:
        configs.append({
            'name': f'More heads ({heads})',
            'gat_hidden_dims': [64, 32], 'rep_dim': 32,
            'head_hidden_dims': [64, 32], 'heads': heads,
            'dropout': 0.0, 'lambda_ipm': 0.0,
            'n_epochs': 500, 'patience': 80,
        })
    
    # 2. Deeper GAT (3-4 layers)
    for gat_dims in [[64, 64, 32], [128, 64, 32], [64, 64, 64, 32]]:
        configs.append({
            'name': f'Deeper GAT {gat_dims}',
            'gat_hidden_dims': gat_dims, 'rep_dim': 32,
            'head_hidden_dims': [64, 32], 'heads': 4,
            'dropout': 0.0, 'lambda_ipm': 0.0,
            'n_epochs': 500, 'patience': 80,
        })
    
    # 3. Larger rep_dim (more capacity)
    for rep_dim in [48, 64, 128]:
        configs.append({
            'name': f'Larger rep_dim={rep_dim}',
            'gat_hidden_dims': [64, 32], 'rep_dim': rep_dim,
            'head_hidden_dims': [64, 32], 'heads': 4,
            'dropout': 0.0, 'lambda_ipm': 0.0,
            'n_epochs': 500, 'patience': 80,
        })
    
    # 4. Wider GAT
    for gat_dims in [[128, 64], [256, 128], [256, 128, 64]]:
        configs.append({
            'name': f'Wider GAT {gat_dims}',
            'gat_hidden_dims': gat_dims, 'rep_dim': 64,
            'head_hidden_dims': [128, 64], 'heads': 4,
            'dropout': 0.0, 'lambda_ipm': 0.0,
            'n_epochs': 500, 'patience': 80,
        })
    
    # 5. Different learning rates
    for lr in [0.0001, 0.0003, 0.002]:
        configs.append({
            'name': f'lr={lr}',
            'gat_hidden_dims': [64, 32], 'rep_dim': 32,
            'head_hidden_dims': [64, 32], 'heads': 4,
            'dropout': 0.0, 'lambda_ipm': 0.0,
            'n_epochs': 800, 'patience': 100,
            'lr': lr,
        })
    
    # 6. Combinations of best ideas
    configs.append({
        'name': 'Deep+Wide+MoreHeads',
        'gat_hidden_dims': [128, 128, 64], 'rep_dim': 64,
        'head_hidden_dims': [128, 64, 32], 'heads': 8,
        'dropout': 0.0, 'lambda_ipm': 0.0,
        'n_epochs': 600, 'patience': 100,
    })
    
    configs.append({
        'name': 'Very Deep GAT (5 layers)',
        'gat_hidden_dims': [64, 64, 64, 64, 32], 'rep_dim': 64,
        'head_hidden_dims': [64, 32], 'heads': 4,
        'dropout': 0.0, 'lambda_ipm': 0.0,
        'n_epochs': 600, 'patience': 100,
    })
    
    # 7. Skip connection idea: shallow GAT + bigger heads
    configs.append({
        'name': 'Shallow GAT + Very Deep Head',
        'gat_hidden_dims': [32], 'rep_dim': 64,
        'head_hidden_dims': [128, 64, 32, 16], 'heads': 4,
        'dropout': 0.0, 'lambda_ipm': 0.0,
        'n_epochs': 500, 'patience': 80,
    })
    
    results = []
    for i, config in enumerate(configs):
        name = config.pop('name')
        print(f"\n[{i+1}/{len(configs)}] {name}")
        
        metrics = run_gat_model(X_run, A, D, Y, Y_pot_true, k, d1, d2, config)
        
        if metrics:
            results.append({'name': name, 'config': config, 'metrics': metrics})
            print(f"  R²={metrics['r2']:.4f}, MSE={metrics['mse_all']:.4f}, tau_bias={metrics['tau_bias']:.4f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (sorted by R²)")
    print("=" * 70)
    
    results_sorted = sorted(results, key=lambda x: -x['metrics']['r2'])
    
    print(f"\n{'Rank':<4} {'R²':>8} {'MSE':>8} {'tau_bias':>10} {'Config'}")
    print("-" * 80)
    for i, r in enumerate(results_sorted):
        m = r['metrics']
        print(f"{i+1:<4} {m['r2']:>8.4f} {m['mse_all']:>8.4f} {m['tau_bias']:>10.4f} {r['name']}")
    
    # Best result
    best = results_sorted[0]
    print("\n" + "=" * 70)
    print(f"BEST: {best['name']}")
    print(f"  R²={best['metrics']['r2']:.4f}")
    print(f"  tau_bias={best['metrics']['tau_bias']:.4f}")
    print(f"  Config: {best['config']}")
    
    # Improvement analysis
    gap = 0.526 - 0.124  # Linear full - Linear no degree
    recovered = (best['metrics']['r2'] - 0.124) / gap
    print(f"\n  Recovery: {100*recovered:.1f}% of the gap to linear with full X")


if __name__ == "__main__":
    main()

