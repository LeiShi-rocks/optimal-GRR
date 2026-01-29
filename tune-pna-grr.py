"""
Hyperparameter Tuning for End-to-End PNA-GRR

This script tunes the TorchPNAGRR model that trains PNA directly on the GRR loss.
"""

import numpy as np
from tqdm import tqdm
from datetime import datetime

from dgp import (
    generate_graph, generate_covariates, generate_potential_outcomes,
    assign_treatment, compute_exposure, observed_outcome
)
from propensity import monte_carlo_probabilities, build_kn_matrix
from GRR import grr_estimator


def run_pna_grr_tuning(
    config_name: str,
    pna_hidden_dims: list,
    rep_dim: int,
    head_hidden_dims: list,
    n_epochs: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    B: int = 20,
    verbose: bool = False
):
    """Run simulation with specific PNA-GRR config."""
    
    # Fixed DGP parameters
    n, avg_degree, max_degree, k = 2000, 3.0, 9, 3
    d1, d2 = 0, 2
    treated_fraction = 1/3
    
    # Generate fixed data
    G, H = generate_graph(n=n, avg_degree=avg_degree, max_degree=max_degree, seed=123)
    X, X_noi = generate_covariates(H, seed=1234)
    X_run = X[:, [0, 3]]
    Y_pot = generate_potential_outcomes(H, X_noi, k=k, seed=999)
    true_tau = float(np.mean(Y_pot[:, d1] - Y_pot[:, d2]))
    
    # Propensity scores
    pi_id_all, _, _, _ = monte_carlo_probabilities(
        G, treated_fraction, k, mc_reps=500, seed_mc=111, d1=d1, d2=d2
    )
    
    # Kn matrix
    Kn = (G @ G.T > 0).astype(float)
    A_adj = G.astype(float)
    
    # Storage
    tau_hats = np.zeros(B)
    losses = np.zeros(B)
    
    # Run B randomizations
    for b in range(B):
        A_treat = assign_treatment(n, treated_fraction=treated_fraction, seed=None)
        D = compute_exposure(G, A_treat, k)
        Y = observed_outcome(Y_pot, D, k=k)
        
        try:
            result = grr_estimator(
                Y, D, X_run, d1, d2, pi_id_all, Kn,
                method='pna',
                adjacency=A_adj,
                pna_hidden_dims=pna_hidden_dims,
                rep_dim=rep_dim,
                head_hidden_dims=head_hidden_dims,
                n_epochs=n_epochs,
                lr=lr,
                weight_decay=weight_decay,
                dropout=dropout,
                early_stopping=True,
                patience=50,
                verbose=False
            )
            tau_hats[b] = result['tau_hat']
            losses[b] = result['loss']
        except Exception as e:
            if verbose:
                print(f"  Rep {b} failed: {e}")
            tau_hats[b] = np.nan
            losses[b] = np.nan
    
    # Compute metrics
    valid = ~np.isnan(tau_hats)
    if valid.sum() > 0:
        bias = float(np.mean(tau_hats[valid]) - true_tau)
        sd = float(np.std(tau_hats[valid], ddof=1))
        rmse = float(np.sqrt(bias**2 + sd**2))
        mean_loss = float(np.mean(losses[valid]))
    else:
        bias = sd = rmse = mean_loss = np.nan
    
    return {
        'config': config_name,
        'bias': bias,
        'sd': sd,
        'rmse': rmse,
        'mean_loss': mean_loss,
        'n_valid': int(valid.sum()),
        'true_tau': true_tau,
        # Config details
        'pna_hidden_dims': pna_hidden_dims,
        'rep_dim': rep_dim,
        'head_hidden_dims': head_hidden_dims,
        'n_epochs': n_epochs,
        'lr': lr,
        'weight_decay': weight_decay,
        'dropout': dropout,
    }


if __name__ == "__main__":
    
    print("=" * 70)
    print("PNA-GRR Hyperparameter Tuning")
    print("=" * 70)
    
    # Baseline linear for comparison
    print("\nRunning baseline linear GRR...")
    n, avg_degree, max_degree, k = 2000, 3.0, 9, 3
    d1, d2 = 0, 2
    G, H = generate_graph(n=n, avg_degree=avg_degree, max_degree=max_degree, seed=123)
    X, X_noi = generate_covariates(H, seed=1234)
    X_run = X[:, [0, 3]]
    Y_pot = generate_potential_outcomes(H, X_noi, k=k, seed=999)
    true_tau = float(np.mean(Y_pot[:, d1] - Y_pot[:, d2]))
    pi_id_all, _, _, _ = monte_carlo_probabilities(G, 1/3, k, 500, 111, d1, d2)
    Kn = (G @ G.T > 0).astype(float)
    
    linear_taus = []
    for _ in range(20):
        A_treat = assign_treatment(n, treated_fraction=1/3, seed=None)
        D = compute_exposure(G, A_treat, k)
        Y = observed_outcome(Y_pot, D, k=k)
        result = grr_estimator(Y, D, X_run, d1, d2, pi_id_all, Kn, method='linear')
        linear_taus.append(result['tau_hat'])
    
    linear_bias = np.mean(linear_taus) - true_tau
    linear_sd = np.std(linear_taus, ddof=1)
    linear_rmse = np.sqrt(linear_bias**2 + linear_sd**2)
    print(f"Linear baseline: Bias={linear_bias:.4f}, SD={linear_sd:.4f}, RMSE={linear_rmse:.4f}")
    
    # Define configurations to test
    configs = [
        # Round 1: Architecture depth and width
        ("shallow_small", [32], 8, [16], 300, 0.001, 1e-4, 0.0),
        ("shallow_medium", [64], 16, [32], 300, 0.001, 1e-4, 0.0),
        ("medium_small", [64, 32], 16, [32], 300, 0.001, 1e-4, 0.0),
        ("medium_medium", [64, 32], 32, [64, 32], 300, 0.001, 1e-4, 0.0),
        ("deep_small", [64, 64, 32], 16, [32], 300, 0.001, 1e-4, 0.0),
        
        # Round 2: Learning rate
        ("medium_lr_high", [64, 32], 16, [32], 300, 0.01, 1e-4, 0.0),
        ("medium_lr_low", [64, 32], 16, [32], 300, 0.0001, 1e-4, 0.0),
        ("medium_lr_vlow", [64, 32], 16, [32], 500, 0.00005, 1e-4, 0.0),
        
        # Round 3: Regularization
        ("medium_wd_high", [64, 32], 16, [32], 300, 0.001, 1e-3, 0.0),
        ("medium_wd_vhigh", [64, 32], 16, [32], 300, 0.001, 1e-2, 0.0),
        ("medium_dropout", [64, 32], 16, [32], 300, 0.001, 1e-4, 0.1),
        ("medium_dropout_wd", [64, 32], 16, [32], 300, 0.001, 1e-3, 0.1),
        
        # Round 4: More epochs with regularization
        ("medium_long", [64, 32], 16, [32], 500, 0.001, 1e-3, 0.0),
        ("medium_vlong", [64, 32], 16, [32], 1000, 0.0005, 1e-3, 0.0),
        
        # Round 5: Simpler models (less overfitting)
        ("tiny", [16], 4, [8], 200, 0.001, 1e-3, 0.0),
        ("linear_head", [32], 8, [], 300, 0.001, 1e-3, 0.0),  # No hidden in head
    ]
    
    results = []
    
    print(f"\nTesting {len(configs)} configurations with B=20 each...")
    print("-" * 70)
    
    for config in tqdm(configs, desc="Configs"):
        name, pna_dims, rep_dim, head_dims, epochs, lr, wd, dropout = config
        
        result = run_pna_grr_tuning(
            config_name=name,
            pna_hidden_dims=pna_dims,
            rep_dim=rep_dim,
            head_hidden_dims=head_dims,
            n_epochs=epochs,
            lr=lr,
            weight_decay=wd,
            dropout=dropout,
            B=20,
            verbose=False
        )
        results.append(result)
        
        # Print progress
        print(f"  {name:20s}: Bias={result['bias']:+.4f}, SD={result['sd']:.4f}, "
              f"RMSE={result['rmse']:.4f}, Loss={result['mean_loss']:.2e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY (sorted by RMSE)")
    print("=" * 70)
    print(f"{'Config':<22} {'Bias':>8} {'SD':>8} {'RMSE':>8} {'vs Linear':>10}")
    print("-" * 70)
    
    # Sort by RMSE
    results_sorted = sorted(results, key=lambda x: x['rmse'] if not np.isnan(x['rmse']) else 999)
    
    for r in results_sorted:
        if not np.isnan(r['rmse']):
            vs_linear = (r['rmse'] - linear_rmse) / linear_rmse * 100
            print(f"{r['config']:<22} {r['bias']:+.4f} {r['sd']:.4f} {r['rmse']:.4f} {vs_linear:+.1f}%")
        else:
            print(f"{r['config']:<22} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
    
    print("-" * 70)
    print(f"{'Linear (baseline)':<22} {linear_bias:+.4f} {linear_sd:.4f} {linear_rmse:.4f} {'0.0%':>10}")
    print("=" * 70)
    
    # Best config
    best = results_sorted[0]
    print(f"\nBest config: {best['config']}")
    print(f"  pna_hidden_dims: {best['pna_hidden_dims']}")
    print(f"  rep_dim: {best['rep_dim']}")
    print(f"  head_hidden_dims: {best['head_hidden_dims']}")
    print(f"  n_epochs: {best['n_epochs']}")
    print(f"  lr: {best['lr']}")
    print(f"  weight_decay: {best['weight_decay']}")
    print(f"  dropout: {best['dropout']}")

