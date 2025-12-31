"""
Hyperparameter tuning for GAT-embedder on out1 setup.

The challenge: X_run = X[:, [0, 3]] excludes degree-based features (X1, X2).
GAT must learn degree information from graph structure.
A simple linear model cannot do this.
"""

import numpy as np
import torch
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Import DGP functions
from dgp import (
    generate_graph,
    generate_covariates,
    generate_potential_outcomes,
    assign_treatment,
    compute_exposure,
    observed_outcome
)

# Import GAT model
import importlib.util
spec = importlib.util.spec_from_file_location("gat_embedder", "GAT-embedder.py")
gat_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gat_module)
GATOutcomeModel = gat_module.GATOutcomeModel


def evaluate_model(Y_pot_pred, Y_pot_true, Y, D, k, d1, d2):
    """Compute evaluation metrics."""
    # MSE on observed outcomes
    mse_observed = 0
    n_obs = 0
    for d in range(k):
        mask = (D == d)
        if mask.sum() > 0:
            mse_observed += ((Y_pot_pred[mask, d] - Y[mask]) ** 2).sum()
            n_obs += mask.sum()
    mse_observed = mse_observed / n_obs
    
    # MSE on all potential outcomes (oracle metric)
    mse_all = np.mean((Y_pot_pred - Y_pot_true) ** 2)
    
    # R² relative to variance
    var_y = np.var(Y_pot_true)
    r2 = 1 - mse_all / var_y
    
    # Treatment effect estimation
    true_tau = np.mean(Y_pot_true[:, d1] - Y_pot_true[:, d2])
    tau_hat = np.mean(Y_pot_pred[:, d1] - Y_pot_pred[:, d2])
    tau_bias = tau_hat - true_tau
    
    return {
        'mse_observed': mse_observed,
        'mse_all': mse_all,
        'r2': r2,
        'true_tau': true_tau,
        'tau_hat': tau_hat,
        'tau_bias': tau_bias,
        'tau_bias_abs': abs(tau_bias),
    }


def run_linear_baseline(X_run, Y, D, Y_pot_true, k, d1, d2):
    """Run linear regression baseline (stratified by exposure)."""
    from sklearn.linear_model import LinearRegression
    
    n = X_run.shape[0]
    Y_pot_pred = np.zeros((n, k))
    
    for d in range(k):
        mask = (D == d)
        if mask.sum() > 10:
            lr = LinearRegression()
            lr.fit(X_run[mask], Y[mask])
            Y_pot_pred[:, d] = lr.predict(X_run)
        else:
            Y_pot_pred[:, d] = Y[mask].mean() if mask.sum() > 0 else 0
    
    return evaluate_model(Y_pot_pred, Y_pot_true, Y, D, k, d1, d2)


def run_gat_model(X_run, A, D, Y, Y_pot_true, k, d1, d2, config, verbose=False):
    """Run GAT model with given config."""
    try:
        model = GATOutcomeModel(
            input_dim=X_run.shape[1],
            k=k,
            gat_hidden_dims=config['gat_hidden_dims'],
            rep_dim=config['rep_dim'],
            head_hidden_dims=config['head_hidden_dims'],
            heads=config['heads'],
            dropout=config['dropout'],
            lambda_ipm=config['lambda_ipm'],
        )
        
        # Suppress GAT device info by redirecting stdout temporarily
        import sys
        from io import StringIO
        if not verbose:
            old_stdout = sys.stdout
            sys.stdout = StringIO()
        
        model.fit(
            X_run, A, D, Y,
            n_epochs=config.get('n_epochs', 400),
            lr=config.get('lr', 0.001),
            weight_decay=config.get('weight_decay', 1e-4),
            early_stopping=True,
            patience=config.get('patience', 50),
            verbose=False,
        )
        
        if not verbose:
            sys.stdout = old_stdout
        
        Y_pot_pred = model.predict(X_run)
        metrics = evaluate_model(Y_pot_pred, Y_pot_true, Y, D, k, d1, d2)
        return metrics
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    print("=" * 70)
    print("GAT-Embedder Hyperparameter Tuning")
    print("=" * 70)
    
    # out1 parameters
    n = 2000
    avg_degree = 3.0
    max_degree = 9
    k = 3
    treated_fraction = 1/3
    d1, d2 = 0, 2
    seed_graph = 123
    seed_cov = 1234
    seed_out = 999
    
    print(f"\nSetup: n={n}, avg_degree={avg_degree}, k={k}")
    print(f"Comparing exposure d={d1} vs d={d2}")
    
    # Generate data
    print("\nGenerating data...")
    G, H = generate_graph(n=n, avg_degree=avg_degree, max_degree=max_degree, seed=seed_graph)
    X, X_noi = generate_covariates(H, seed=seed_cov)
    Y_pot_true = generate_potential_outcomes(H, X_noi, k=k, seed=seed_out)
    
    # Use only first and last columns (exclude degree-based X1, X2)
    X_run = X[:, [0, 3]]  # intercept and X3
    print(f"X_run shape: {X_run.shape} (using columns 0 and 3 only)")
    
    # Treatment assignment
    A_treat = assign_treatment(n, treated_fraction=treated_fraction, seed=42)
    D = compute_exposure(G, A_treat, k)
    Y = observed_outcome(Y_pot_true, D, k=k)
    
    print(f"\nExposure distribution:")
    for d in range(k):
        count = (D == d).sum()
        print(f"  D={d}: {count} ({100*count/n:.1f}%)")
    
    true_tau = np.mean(Y_pot_true[:, d1] - Y_pot_true[:, d2])
    print(f"\nTrue tau: {true_tau:.4f}")
    
    A = G.astype(float)
    
    # =========================================================================
    # Linear Baseline
    # =========================================================================
    print("\n" + "=" * 70)
    print("Linear Baseline (stratified by exposure)")
    print("=" * 70)
    
    linear_metrics = run_linear_baseline(X_run, Y, D, Y_pot_true, k, d1, d2)
    print(f"  MSE (observed): {linear_metrics['mse_observed']:.4f}")
    print(f"  MSE (all Y_pot): {linear_metrics['mse_all']:.4f}")
    print(f"  R²: {linear_metrics['r2']:.4f}")
    print(f"  tau_hat: {linear_metrics['tau_hat']:.4f}")
    print(f"  tau_bias: {linear_metrics['tau_bias']:.4f}")
    
    # Also try with full X
    print("\n[Linear with full X (includes degree info)]")
    linear_full = run_linear_baseline(X, Y, D, Y_pot_true, k, d1, d2)
    print(f"  MSE (all Y_pot): {linear_full['mse_all']:.4f}")
    print(f"  R²: {linear_full['r2']:.4f}")
    print(f"  tau_bias: {linear_full['tau_bias']:.4f}")
    
    # =========================================================================
    # Hyperparameter Search
    # =========================================================================
    print("\n" + "=" * 70)
    print("GAT Hyperparameter Search")
    print("=" * 70)
    
    # Define search space
    configs = []
    
    # Varying architecture depth/width
    gat_architectures = [
        ([32], 16),           # shallow
        ([64], 16),           # wider shallow
        ([64, 32], 16),       # medium
        ([64, 32], 32),       # medium, larger rep
        ([128, 64], 32),      # deeper, wider
    ]
    
    head_architectures = [
        [16],                 # shallow head
        [32, 16],             # medium head  
        [64, 32],             # deeper head
        [64, 32, 16],         # very deep head
    ]
    
    # Fixed good params from previous tuning
    base_config = {
        'heads': 4,
        'dropout': 0.0,
        'lambda_ipm': 0.0,
        'n_epochs': 400,
        'lr': 0.001,
        'weight_decay': 1e-4,
        'patience': 50,
    }
    
    # Generate configs
    for (gat_dims, rep_dim), head_dims in product(gat_architectures, head_architectures):
        config = base_config.copy()
        config['gat_hidden_dims'] = gat_dims
        config['rep_dim'] = rep_dim
        config['head_hidden_dims'] = head_dims
        configs.append(config)
    
    # Also try varying IPM and dropout with best architecture
    for lambda_ipm in [0.0, 0.1, 1.0]:
        for dropout in [0.0, 0.1, 0.2]:
            config = base_config.copy()
            config['gat_hidden_dims'] = [64, 32]
            config['rep_dim'] = 16
            config['head_hidden_dims'] = [64, 32, 16]
            config['lambda_ipm'] = lambda_ipm
            config['dropout'] = dropout
            configs.append(config)
    
    # Try different learning rates
    for lr in [0.0005, 0.001, 0.002]:
        config = base_config.copy()
        config['gat_hidden_dims'] = [64, 32]
        config['rep_dim'] = 16
        config['head_hidden_dims'] = [64, 32, 16]
        config['lr'] = lr
        configs.append(config)
    
    # Remove duplicates
    unique_configs = []
    seen = set()
    for c in configs:
        key = str(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            unique_configs.append(c)
    
    print(f"Testing {len(unique_configs)} configurations...")
    
    results = []
    for i, config in enumerate(unique_configs):
        desc = (f"GAT{config['gat_hidden_dims']}→{config['rep_dim']}, "
                f"head{config['head_hidden_dims']}, "
                f"ipm={config['lambda_ipm']}, drop={config['dropout']}")
        print(f"\n[{i+1}/{len(unique_configs)}] {desc}")
        
        metrics = run_gat_model(X_run, A, D, Y, Y_pot_true, k, d1, d2, config, verbose=False)
        
        if metrics is not None:
            results.append({
                'config': config,
                'metrics': metrics,
                'desc': desc,
            })
            print(f"  MSE={metrics['mse_all']:.4f}, R²={metrics['r2']:.4f}, "
                  f"tau_bias={metrics['tau_bias']:.4f}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    # Sort by R²
    results_sorted = sorted(results, key=lambda x: -x['metrics']['r2'])
    
    print("\nTop 10 configurations by R²:")
    print("-" * 90)
    print(f"{'Rank':<5} {'R²':>8} {'MSE':>8} {'tau_bias':>10} {'Config'}")
    print("-" * 90)
    
    for i, r in enumerate(results_sorted[:10]):
        m = r['metrics']
        print(f"{i+1:<5} {m['r2']:>8.4f} {m['mse_all']:>8.4f} {m['tau_bias']:>10.4f} {r['desc']}")
    
    # Compare to baselines
    print("\n" + "-" * 70)
    print("COMPARISON TO BASELINES")
    print("-" * 70)
    
    best = results_sorted[0]
    print(f"\nLinear (X_run only, no degree info):")
    print(f"  R²={linear_metrics['r2']:.4f}, MSE={linear_metrics['mse_all']:.4f}, tau_bias={linear_metrics['tau_bias']:.4f}")
    
    print(f"\nLinear (full X, with degree info):")
    print(f"  R²={linear_full['r2']:.4f}, MSE={linear_full['mse_all']:.4f}, tau_bias={linear_full['tau_bias']:.4f}")
    
    print(f"\nBest GAT (X_run only):")
    print(f"  R²={best['metrics']['r2']:.4f}, MSE={best['metrics']['mse_all']:.4f}, tau_bias={best['metrics']['tau_bias']:.4f}")
    print(f"  Config: {best['desc']}")
    
    # Print improvement
    r2_improvement = best['metrics']['r2'] - linear_metrics['r2']
    print(f"\n  GAT R² improvement over linear (no degree): {r2_improvement:+.4f}")
    
    gap_to_full = linear_full['r2'] - linear_metrics['r2']
    recovered = r2_improvement / gap_to_full if gap_to_full > 0 else 0
    print(f"  GAT recovers {100*recovered:.1f}% of the gap to linear with full X")
    
    # Best config
    print("\n" + "=" * 70)
    print("BEST CONFIGURATION")
    print("=" * 70)
    print(f"\n{best['config']}")


if __name__ == "__main__":
    main()

