"""
GAT-Embedder Tuning Round 3: Combining best ideas from Round 2.

Best findings so far:
- Very Deep GAT (5 layers): R²=0.257
- Larger rep_dim=48: R²=0.244
- 4-layer GAT: Best tau_bias=0.271

Try: even deeper GATs, combine depth + larger rep_dim
"""

import numpy as np
import torch
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


def run_gat_model(X_run, A, D, Y, Y_pot_true, k, d1, d2, config):
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
        
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        model.fit(
            X_run, A, D, Y,
            n_epochs=config.get('n_epochs', 600),
            lr=config.get('lr', 0.001),
            weight_decay=config.get('weight_decay', 1e-4),
            early_stopping=True,
            patience=config.get('patience', 100),
            verbose=False,
        )
        
        sys.stdout = old_stdout
        
        Y_pot_pred = model.predict(X_run)
        return evaluate_model(Y_pot_pred, Y_pot_true, Y, D, k, d1, d2)
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    print("=" * 70)
    print("GAT-Embedder Tuning Round 3: Combining Best Ideas")
    print("=" * 70)
    
    # Generate data
    n, avg_degree, max_degree, k = 2000, 3.0, 9, 3
    treated_fraction, d1, d2 = 1/3, 0, 2
    
    G, H = generate_graph(n=n, avg_degree=avg_degree, max_degree=max_degree, seed=123)
    X, X_noi = generate_covariates(H, seed=1234)
    Y_pot_true = generate_potential_outcomes(H, X_noi, k=k, seed=999)
    X_run = X[:, [0, 3]]
    
    A_treat = assign_treatment(n, treated_fraction=treated_fraction, seed=42)
    D = compute_exposure(G, A_treat, k)
    Y = observed_outcome(Y_pot_true, D, k=k)
    A = G.astype(float)
    
    true_tau = np.mean(Y_pot_true[:, d1] - Y_pot_true[:, d2])
    print(f"True tau: {true_tau:.4f}")
    
    print("\nBaselines:")
    print("  Linear (no degree): R²=0.124, tau_bias=0.360")
    print("  Linear (full X):    R²=0.526, tau_bias=0.276")
    print("  Best GAT (Round 2): R²=0.257, tau_bias=0.305")
    
    print("\n" + "=" * 70)
    print("Testing Round 3 configurations...")
    print("=" * 70)
    
    configs = [
        # 1. Very deep + larger rep_dim (best combo)
        {'name': '5-layer + rep_dim=48',
         'gat_hidden_dims': [64, 64, 64, 64, 32], 'rep_dim': 48,
         'head_hidden_dims': [64, 32], 'heads': 4},
         
        {'name': '5-layer + rep_dim=64',
         'gat_hidden_dims': [64, 64, 64, 64, 32], 'rep_dim': 64,
         'head_hidden_dims': [64, 32], 'heads': 4},
        
        # 2. Even deeper (6 layers)
        {'name': '6-layer GAT',
         'gat_hidden_dims': [64, 64, 64, 64, 64, 32], 'rep_dim': 64,
         'head_hidden_dims': [64, 32], 'heads': 4},
        
        # 3. Deeper + wider heads
        {'name': '5-layer + deeper heads',
         'gat_hidden_dims': [64, 64, 64, 64, 32], 'rep_dim': 64,
         'head_hidden_dims': [128, 64, 32], 'heads': 4},
        
        # 4. All wide layers
        {'name': '5-layer wide (128)',
         'gat_hidden_dims': [128, 128, 128, 128, 64], 'rep_dim': 64,
         'head_hidden_dims': [64, 32], 'heads': 4},
        
        # 5. Tapered deep
        {'name': '5-layer tapered [128->32]',
         'gat_hidden_dims': [128, 96, 64, 48, 32], 'rep_dim': 48,
         'head_hidden_dims': [64, 32], 'heads': 4},
        
        # 6. Multiple runs of best config (check variance)
        {'name': '5-layer baseline (run 1)',
         'gat_hidden_dims': [64, 64, 64, 64, 32], 'rep_dim': 64,
         'head_hidden_dims': [64, 32], 'heads': 4},
        
        {'name': '5-layer baseline (run 2)',
         'gat_hidden_dims': [64, 64, 64, 64, 32], 'rep_dim': 64,
         'head_hidden_dims': [64, 32], 'heads': 4},
        
        {'name': '5-layer baseline (run 3)',
         'gat_hidden_dims': [64, 64, 64, 64, 32], 'rep_dim': 64,
         'head_hidden_dims': [64, 32], 'heads': 4},
        
        # 7. Longer training
        {'name': '5-layer + 1000 epochs',
         'gat_hidden_dims': [64, 64, 64, 64, 32], 'rep_dim': 64,
         'head_hidden_dims': [64, 32], 'heads': 4,
         'n_epochs': 1000, 'patience': 150},
    ]
    
    # Add defaults
    for c in configs:
        c.setdefault('heads', 4)
        c.setdefault('dropout', 0.0)
        c.setdefault('lambda_ipm', 0.0)
        c.setdefault('n_epochs', 600)
        c.setdefault('patience', 100)
    
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
    
    gap = 0.526 - 0.124
    recovered = (best['metrics']['r2'] - 0.124) / gap
    print(f"\n  Recovery: {100*recovered:.1f}% of the gap to linear with full X")


if __name__ == "__main__":
    main()

