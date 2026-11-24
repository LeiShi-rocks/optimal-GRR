#!/usr/bin/env python
"""
Test GAT network fitting on the cash transfer data.
Goal: Fit GAT to predict outcomes from covariates + graph, minimize MSE.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, '/home/colligo/personal-research/optimal-GRR')

from GRR import GAT_embedding

# =============================================================================
# Load Data
# =============================================================================

def load_data():
    """Load the preprocessed village data and adjacency matrix."""
    data_dir = os.path.join(os.path.dirname(__file__), 'processed_data')
    
    # Load village data
    village_data = pd.read_csv(os.path.join(data_dir, 'village_data.csv'))
    
    # Load adjacency matrix
    G = np.load(os.path.join(data_dir, 'adjacency_matrix_dense.npy'))
    
    return village_data, G


def main():
    print("=" * 80)
    print("GAT NETWORK FITTING TEST")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    village_data, G = load_data()
    
    n = len(village_data)
    k = 8  # Number of exposure levels
    
    # Extract arrays
    Y = village_data['profit_per_ent'].values
    D = village_data['D'].values.astype(int)
    
    # Build covariate matrix
    covariate_cols = ['n_allents_BL', 'profit_BL_per_ent', 'revenue_BL_per_ent', 'cost_BL_per_ent']
    X_cols = [col for col in covariate_cols if col in village_data.columns]
    X = village_data[X_cols].values
    X = np.nan_to_num(X, nan=0.0)
    
    print(f"\nData summary:")
    print(f"  Sample size: {n}")
    print(f"  Covariates: {X_cols} (dim={X.shape[1]})")
    print(f"  Adjacency: {G.shape}, edges: {(G > 0).sum()}")
    print(f"  Outcome (profit_per_ent): mean={np.nanmean(Y):.2f}, std={np.nanstd(Y):.2f}")
    print(f"  Exposure levels: {np.unique(D)}")
    for d in range(k):
        print(f"    D={d}: n={np.sum(D == d)}")
    
    # Normalize outcome for training
    Y_mean = np.nanmean(Y)
    Y_std = np.nanstd(Y)
    Y_normalized = (Y - Y_mean) / Y_std
    print(f"\n  Normalizing outcome: (Y - {Y_mean:.2f}) / {Y_std:.2f}")
    
    # Configurations to test
    configs = [
        # Baseline
        {'name': 'baseline', 'gat_hidden_dims': [64, 32], 'rep_dim': 16, 
         'head_hidden_dims': [32, 16], 'heads': 4, 'dropout': 0.0, 'lr': 0.001, 'n_epochs': 300},
        
        # Deeper encoder
        {'name': 'deeper', 'gat_hidden_dims': [64, 64, 32], 'rep_dim': 32, 
         'head_hidden_dims': [32, 16], 'heads': 4, 'dropout': 0.0, 'lr': 0.001, 'n_epochs': 500},
        
        # More attention heads
        {'name': '8heads', 'gat_hidden_dims': [64, 64, 32], 'rep_dim': 32, 
         'head_hidden_dims': [32, 16], 'heads': 8, 'dropout': 0.0, 'lr': 0.001, 'n_epochs': 500},
        
        # Wider
        {'name': 'wider', 'gat_hidden_dims': [128, 64], 'rep_dim': 32, 
         'head_hidden_dims': [64, 32], 'heads': 4, 'dropout': 0.0, 'lr': 0.001, 'n_epochs': 500},
        
        # More regularization
        {'name': 'regularized', 'gat_hidden_dims': [128, 64, 32], 'rep_dim': 32, 
         'head_hidden_dims': [64, 32], 'heads': 4, 'dropout': 0.2, 'lr': 0.0005, 'n_epochs': 500},
        
        # Large model
        {'name': 'large', 'gat_hidden_dims': [256, 128, 64], 'rep_dim': 64, 
         'head_hidden_dims': [128, 64], 'heads': 8, 'dropout': 0.1, 'lr': 0.0005, 'n_epochs': 500},
    ]
    
    # Fixed params for all configs
    fixed_params = {
        'lambda_ipm': 0.0,
        'weight_decay': 1e-4,
        'early_stopping': True,
        'patience': 75,
        'verbose': False,
        'return_model': True,
    }
    
    print("\n" + "=" * 80)
    print("TESTING MULTIPLE GAT CONFIGURATIONS")
    print("=" * 80)
    
    results = []
    best_r2 = -np.inf
    best_config_name = None
    best_model = None
    best_Y_pred = None
    
    for config in configs:
        name = config.pop('name')
        full_config = {**fixed_params, **config}
        
        print(f"\n[{name}] Testing: hidden={config['gat_hidden_dims']}, rep={config.get('rep_dim', 16)}, "
              f"head={config['head_hidden_dims']}, heads={config['heads']}, lr={config['lr']}, dropout={config['dropout']}")
        
        try:
            # Fit model on normalized Y
            result = GAT_embedding(X, G.astype(float), D, Y_normalized, k=k, **full_config)
            model = result['model']
            
            # Evaluate
            Y_pred_norm = np.zeros_like(Y, dtype=float)
            for d_level in range(k):
                mask = (D == d_level)
                if mask.sum() > 0:
                    preds = model.predict(X, D=d_level)
                    Y_pred_norm[mask] = preds[mask]
            
            # Denormalize
            Y_pred = Y_pred_norm * Y_std + Y_mean
            
            valid_mask = ~np.isnan(Y)
            mse = np.mean((Y[valid_mask] - Y_pred[valid_mask])**2)
            rmse = np.sqrt(mse)
            ss_res = np.sum((Y[valid_mask] - Y_pred[valid_mask])**2)
            ss_tot = np.sum((Y[valid_mask] - np.mean(Y[valid_mask]))**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            results.append({'name': name, 'mse': mse, 'rmse': rmse, 'r2': r2, **config})
            
            if r2 > best_r2:
                best_r2 = r2
                best_config_name = name
                best_model = model
                best_Y_pred = Y_pred
                print(f"  -> NEW BEST! R2={r2:.4f}, RMSE={rmse:.2f}")
            else:
                print(f"  -> R2={r2:.4f}, RMSE={rmse:.2f}")
        
        except Exception as e:
            print(f"  -> ERROR: {e}")
            results.append({'name': name, 'mse': np.inf, 'rmse': np.inf, 'r2': -np.inf, **config})
        
        # Restore name for next iteration
        config['name'] = name
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n{'Config':<15} {'R2':>10} {'RMSE':>12}")
    print("-" * 40)
    for r in sorted(results, key=lambda x: -x['r2']):
        print(f"{r['name']:<15} {r['r2']:>10.4f} {r['rmse']:>12.2f}")
    
    print(f"\nBest config: {best_config_name} (R2={best_r2:.4f})")
    
    if best_model is not None and best_Y_pred is not None:
        print("\n" + "=" * 80)
        print(f"DETAILED EVALUATION OF BEST MODEL ({best_config_name})")
        print("=" * 80)
        
        Y_pred = best_Y_pred
        valid_mask = ~np.isnan(Y)
        
        # Per-exposure-level metrics
        print(f"\nPer-exposure-level metrics:")
        for d in range(k):
            mask = (D == d) & valid_mask
            if mask.sum() > 0:
                mse_d = np.mean((Y[mask] - Y_pred[mask])**2)
                rmse_d = np.sqrt(mse_d)
                print(f"  D={d} (n={mask.sum()}): RMSE={rmse_d:.2f}, mean_Y={Y[mask].mean():.2f}, mean_Yhat={Y_pred[mask].mean():.2f}")
        
        # Show some predictions
        print(f"\nSample predictions (first 10):")
        print(f"{'idx':>5} {'D':>3} {'Y_true':>10} {'Y_pred':>10} {'Error':>10}")
        for i in range(min(10, n)):
            print(f"{i:>5} {D[i]:>3} {Y[i]:>10.2f} {Y_pred[i]:>10.2f} {Y[i]-Y_pred[i]:>10.2f}")
        
        # Print recommended config
        best_cfg = next(c for c in configs if c['name'] == best_config_name)
        print("\n" + "=" * 80)
        print("RECOMMENDED GAT_CONFIG FOR GRR-analysis.py")
        print("=" * 80)
        print(f"""
GAT_CONFIG = {{
    'gat_hidden_dims': {best_cfg['gat_hidden_dims']},
    'rep_dim': {best_cfg['rep_dim']},
    'head_hidden_dims': {best_cfg['head_hidden_dims']},
    'heads': {best_cfg['heads']},
    'dropout': {best_cfg['dropout']},
    'lambda_ipm': 0.0,
    'n_epochs': {best_cfg['n_epochs']},
    'lr': {best_cfg['lr']},
    'weight_decay': 1e-4,
    'early_stopping': True,
    'patience': 75,
}}
""")
    
    # Compare with PNA results (if available)
    print("\n" + "=" * 80)
    print("COMPARISON WITH PNA")
    print("=" * 80)
    print(f"\n  Best GAT R2: {best_r2:.4f}")
    print(f"  Best PNA R2: 0.1256 (from previous run with 'deeper' config)")
    if best_r2 > 0.1256:
        print(f"\n  -> GAT is better by {(best_r2 - 0.1256)*100:.2f} percentage points")
    else:
        print(f"\n  -> PNA is better by {(0.1256 - best_r2)*100:.2f} percentage points")


if __name__ == '__main__':
    main()
