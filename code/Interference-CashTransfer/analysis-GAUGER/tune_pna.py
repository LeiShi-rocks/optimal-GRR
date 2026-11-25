#!/usr/bin/env python
"""
Tune PNA hyperparameters for the (A, H) exposure mapping with H = more than 1/3 neighbors treated.
Goal: Find the best PNA configuration that gives:
1. Good predictions (high R2)
2. Short confidence intervals for treatment effect estimates
"""

import sys
import os
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'synthetic'))

from GRR import PNA_embedding, prepare_grr_inputs
from variance_estimators import grr_confidence_interval
from propensity import build_W_pi

# =============================================================================
# Load Data
# =============================================================================

def load_data():
    """Load the preprocessed village data, adjacency matrix, and propensity scores."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    derived_dir = os.path.join(script_dir, '..', '..', '..', 'data', 'cash-transfer-derived')
    data_dir = os.path.join(derived_dir, 'processed_data')
    propensity_dir = os.path.join(derived_dir, 'inclusion_probabilities')
    
    # Load village data
    village_data = pd.read_csv(os.path.join(data_dir, 'village_data.csv'))
    
    # Load adjacency matrix
    G = np.load(os.path.join(data_dir, 'adjacency_matrix.npy'))
    
    # Load propensity scores
    propensity_df = pd.read_csv(os.path.join(propensity_dir, 'propensity_scores.csv'))
    
    # Build propensity matrix
    ps_columns = ['(A=0,H=0)', '(A=0,H=1)', '(A=1,H=0)', '(A=1,H=1)']
    n = len(village_data)
    pi_id_all = np.zeros((n, 4))
    
    village_data['village_code'] = village_data['village_code'].astype(float)
    propensity_df['village_code'] = propensity_df['village_code'].astype(float)
    
    for i, row in village_data.iterrows():
        village = row['village_code']
        ps_row = propensity_df[propensity_df['village_code'] == village][ps_columns]
        if len(ps_row) > 0:
            pi_id_all[i] = ps_row.values[0]
    
    # Load second-order matrices
    matrix_dir = os.path.join(propensity_dir, 'second_order_matrices')
    second_order_matrices = {}
    for a in [0, 1]:
        for h in [0, 1]:
            key = (a, h)
            filename = os.path.join(matrix_dir, f"P_A{a}_H{h}_sparse.csv")
            if os.path.exists(filename):
                second_order_matrices[key] = pd.read_csv(filename)
    
    return village_data, G, pi_id_all, second_order_matrices


def build_second_order_matrix(second_order_df, n, d):
    """Build second-order matrix from sparse dataframe."""
    a = d // 2
    h = d % 2
    key = (a, h)
    
    if key not in second_order_df or second_order_df[key] is None:
        return None
    
    df = second_order_df[key]
    pi_ij = np.zeros((n, n))
    
    for _, row in df.iterrows():
        i = int(row['village_i_idx'])
        j = int(row['village_j_idx'])
        prob = row['probability']
        pi_ij[i, j] = prob
        pi_ij[j, i] = prob
    
    return pi_ij


def compute_ci_width(Y, D, X, pi_id_all, G, second_order_matrices, pna_model, 
                     Y_mean, Y_std, contrasts, eps=1e-12):
    """
    Compute average CI width for pna-pred, pna-no-harm, and pna-linear estimators.
    
    Uses global normalization.
    
    Returns dict with CI widths for each contrast and average widths.
    """
    from GRR import grr_estimator, LinearGRR
    
    n = len(Y)
    Kn = (G @ G.T > 0).astype(int)
    
    ci_widths_pred = {}
    ci_widths_noharm = {}
    ci_widths_linear = {}
    total_width_pred = 0
    total_width_noharm = 0
    total_width_linear = 0
    n_contrasts = 0
    
    for d1, d2, name in contrasts:
        # Get PNA predictions - denormalize using global stats
        Ypred_d1 = pna_model.predict(X, D=d1) * Y_std + Y_mean
        Ypred_d2 = pna_model.predict(X, D=d2) * Y_std + Y_mean
        
        X1_use = Ypred_d1.reshape(-1, 1)
        X2_use = Ypred_d2.reshape(-1, 1)
        
        # Build W matrices for variance
        try:
            pi_ij_d1d1 = build_second_order_matrix(second_order_matrices, n, d1)
            pi_ij_d2d2 = build_second_order_matrix(second_order_matrices, n, d2)
            
            if pi_ij_d1d1 is not None and pi_ij_d2d2 is not None:
                W1_pi, W2_pi = build_W_pi(pi_id_all, pi_ij_d1d1, pi_ij_d2d2, d1, d2, Kn, robust=False)
                
                # ---- pna-pred: beta = 1 ----
                beta1_pred = np.array([1.0])
                beta2_pred = np.array([1.0])
                
                pi1 = pi_id_all[:, d1]
                pi2 = pi_id_all[:, d2]
                
                f1 = X1_use @ beta1_pred
                f2 = X2_use @ beta2_pred
                
                hat1_d1 = (D == d1).astype(float) / (pi1 + eps)
                hat1_d2 = (D == d2).astype(float) / (pi2 + eps)
                
                mu1 = np.mean(hat1_d1 * Y - hat1_d1 * f1 + f1)
                mu2 = np.mean(hat1_d2 * Y - hat1_d2 * f2 + f2)
                tau_pred = mu1 - mu2
                
                inputs = prepare_grr_inputs(Y, D, X, d1, d2, pi_id_all, X1=X1_use, X2=X2_use, eps=eps)
                ci_lo, ci_hi = grr_confidence_interval(
                    tau_hat=tau_pred,
                    hatY_d1=inputs['hatY_d1'],
                    hatY_d2=inputs['hatY_d2'],
                    hat1_d1=inputs['hat1_d1'],
                    hat1_d2=inputs['hat1_d2'],
                    X1=X1_use, X2=X2_use,
                    beta1=beta1_pred, beta2=beta2_pred,
                    W1_pi=W1_pi, W2_pi=W2_pi,
                    alpha=0.05
                )
                width_pred = ci_hi - ci_lo
                ci_widths_pred[name] = {'tau': tau_pred, 'ci_lo': ci_lo, 'ci_hi': ci_hi, 'width': width_pred}
                total_width_pred += width_pred
                
                # ---- pna-no-harm: use linear GRR to find optimal beta ----
                grr_result = grr_estimator(
                    Y, D, X, d1, d2, pi_id_all, Kn,
                    method='linear', eps=eps,
                    X1=X1_use, X2=X2_use,
                    alpha=0.01
                )
                tau_noharm = grr_result['tau_hat']
                
                if hasattr(grr_result['model'], 'get_coefficients'):
                    beta1_nh, beta2_nh = grr_result['model'].get_coefficients()
                else:
                    beta1_nh, beta2_nh = beta1_pred, beta2_pred
                
                ci_lo_nh, ci_hi_nh = grr_confidence_interval(
                    tau_hat=tau_noharm,
                    hatY_d1=inputs['hatY_d1'],
                    hatY_d2=inputs['hatY_d2'],
                    hat1_d1=inputs['hat1_d1'],
                    hat1_d2=inputs['hat1_d2'],
                    X1=X1_use, X2=X2_use,
                    beta1=beta1_nh, beta2=beta2_nh,
                    W1_pi=W1_pi, W2_pi=W2_pi,
                    alpha=0.05
                )
                width_noharm = ci_hi_nh - ci_lo_nh
                ci_widths_noharm[name] = {'tau': tau_noharm, 'ci_lo': ci_lo_nh, 'ci_hi': ci_hi_nh, 'width': width_noharm}
                total_width_noharm += width_noharm
                
                # ---- pna-linear: use linear GRR on original X + PNA embeddings ----
                # Combine original covariates with PNA predictions
                X_combined = np.hstack([X, X1_use, X2_use])
                grr_linear = grr_estimator(
                    Y, D, X_combined, d1, d2, pi_id_all, Kn,
                    method='linear', eps=eps,
                    alpha=0.01
                )
                tau_linear = grr_linear['tau_hat']
                
                # For linear, we need to use X_combined
                X1_linear = X_combined.copy()
                X2_linear = X_combined.copy()
                
                if hasattr(grr_linear['model'], 'get_coefficients'):
                    beta1_lin, beta2_lin = grr_linear['model'].get_coefficients()
                else:
                    beta1_lin = np.zeros(X_combined.shape[1])
                    beta2_lin = np.zeros(X_combined.shape[1])
                
                inputs_lin = prepare_grr_inputs(Y, D, X_combined, d1, d2, pi_id_all, X1=X1_linear, X2=X2_linear, eps=eps)
                ci_lo_lin, ci_hi_lin = grr_confidence_interval(
                    tau_hat=tau_linear,
                    hatY_d1=inputs_lin['hatY_d1'],
                    hatY_d2=inputs_lin['hatY_d2'],
                    hat1_d1=inputs_lin['hat1_d1'],
                    hat1_d2=inputs_lin['hat1_d2'],
                    X1=X1_linear, X2=X2_linear,
                    beta1=beta1_lin, beta2=beta2_lin,
                    W1_pi=W1_pi, W2_pi=W2_pi,
                    alpha=0.05
                )
                width_linear = ci_hi_lin - ci_lo_lin
                ci_widths_linear[name] = {'tau': tau_linear, 'ci_lo': ci_lo_lin, 'ci_hi': ci_hi_lin, 'width': width_linear}
                total_width_linear += width_linear
                
                n_contrasts += 1
        except Exception as e:
            ci_widths_pred[name] = {'tau': 0, 'error': str(e)}
            ci_widths_noharm[name] = {'tau': 0, 'error': str(e)}
            ci_widths_linear[name] = {'tau': 0, 'error': str(e)}
    
    avg_width_pred = total_width_pred / n_contrasts if n_contrasts > 0 else float('inf')
    avg_width_noharm = total_width_noharm / n_contrasts if n_contrasts > 0 else float('inf')
    avg_width_linear = total_width_linear / n_contrasts if n_contrasts > 0 else float('inf')
    
    return ci_widths_pred, avg_width_pred, ci_widths_noharm, avg_width_noharm, ci_widths_linear, avg_width_linear


def main():
    print("=" * 80)
    print("PNA HYPERPARAMETER TUNING - H = more than 1/3 neighbors treated")
    print("Optimizing for: R2 AND CI width")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    village_data, G, pi_id_all, second_order_matrices = load_data()
    
    n = len(village_data)
    k = 4  # Number of exposure levels (A, H)
    
    # Extract arrays
    Y = village_data['revenue_per_ent'].values  # Using revenue as in the analysis
    D = village_data['D'].values.astype(int)
    
    # Define contrasts for CI evaluation
    contrasts = [
        (2, 0, "Direct(H=0)"),   # A=1 vs A=0 when H=0
        (3, 1, "Direct(H=1)"),   # A=1 vs A=0 when H=1
        (1, 0, "Spillover(A=0)"), # H=1 vs H=0 when A=0
        (3, 2, "Spillover(A=1)"), # H=1 vs H=0 when A=1
    ]
    
    # Build covariate matrix
    covariate_cols = ['n_allents_BL', 'profit_BL_per_ent', 'revenue_BL_per_ent', 'cost_BL_per_ent']
    X_cols = [col for col in covariate_cols if col in village_data.columns]
    X = village_data[X_cols].values
    X = np.nan_to_num(X, nan=0.0)
    
    print(f"\nData summary:")
    print(f"  Sample size: {n}")
    print(f"  Covariates: {X_cols} (dim={X.shape[1]})")
    print(f"  Adjacency: {G.shape}, edges: {(G > 0).sum()}")
    print(f"  Outcome (revenue_per_ent): mean={np.nanmean(Y):.2f}, std={np.nanstd(Y):.2f}")
    print(f"  Exposure levels: {np.unique(D)} (4 levels: (A,H))")
    for d in range(k):
        a = d // 2
        h = d % 2
        print(f"    D={d} (A={a}, H={h}): n={np.sum(D == d)}")
    
    # Normalize outcome for training - GLOBAL normalization
    Y_mean = np.nanmean(Y)
    Y_std = np.nanstd(Y)
    Y_normalized = (Y - Y_mean) / Y_std
    print(f"\n  Global normalization: (Y - {Y_mean:.2f}) / {Y_std:.2f}")
    
    # Configurations to test - expanded search focused on CI width reduction
    configs = [
        # ===== STRONG REGULARIZATION (key for narrow CIs) =====
        {'name': 'strong_wd05', 'pna_hidden_dims': [128], 'rep_dim': 32, 
         'head_hidden_dims': [32], 'dropout': 0.0, 'lr': 0.001, 'n_epochs': 800,
         'weight_decay': 0.05,
         'aggregators': ['mean', 'sum', 'max', 'std'], 'scalers': ['identity', 'amplification', 'attenuation']},
        
        {'name': 'strong_wd1', 'pna_hidden_dims': [128], 'rep_dim': 32, 
         'head_hidden_dims': [32], 'dropout': 0.0, 'lr': 0.001, 'n_epochs': 800,
         'weight_decay': 0.1,
         'aggregators': ['mean', 'sum', 'max', 'std'], 'scalers': ['identity', 'amplification', 'attenuation']},
        
        {'name': 'strong_wd2', 'pna_hidden_dims': [128], 'rep_dim': 32, 
         'head_hidden_dims': [32], 'dropout': 0.0, 'lr': 0.001, 'n_epochs': 1000,
         'weight_decay': 0.2,
         'aggregators': ['mean', 'sum', 'max', 'std'], 'scalers': ['identity', 'amplification', 'attenuation']},
        
        # ===== DROPOUT REGULARIZATION =====
        {'name': 'dropout30', 'pna_hidden_dims': [128], 'rep_dim': 32, 
         'head_hidden_dims': [32], 'dropout': 0.3, 'lr': 0.001, 'n_epochs': 800,
         'aggregators': ['mean', 'sum', 'max', 'std'], 'scalers': ['identity', 'amplification', 'attenuation']},
        
        {'name': 'dropout50', 'pna_hidden_dims': [128], 'rep_dim': 32, 
         'head_hidden_dims': [32], 'dropout': 0.5, 'lr': 0.001, 'n_epochs': 1000,
         'aggregators': ['mean', 'sum', 'max', 'std'], 'scalers': ['identity', 'amplification', 'attenuation']},
        
        {'name': 'dropout_wd', 'pna_hidden_dims': [128], 'rep_dim': 32, 
         'head_hidden_dims': [32], 'dropout': 0.3, 'lr': 0.001, 'n_epochs': 800,
         'weight_decay': 0.05,
         'aggregators': ['mean', 'sum', 'max', 'std'], 'scalers': ['identity', 'amplification', 'attenuation']},
        
        # ===== SIMPLER/SMALLER MODELS (less variance) =====
        {'name': 'tiny', 'pna_hidden_dims': [32], 'rep_dim': 16, 
         'head_hidden_dims': [16], 'dropout': 0.0, 'lr': 0.001, 'n_epochs': 500,
         'aggregators': ['mean', 'sum'], 'scalers': ['identity']},
        
        {'name': 'tiny_wd', 'pna_hidden_dims': [32], 'rep_dim': 16, 
         'head_hidden_dims': [16], 'dropout': 0.0, 'lr': 0.001, 'n_epochs': 500,
         'weight_decay': 0.05,
         'aggregators': ['mean', 'sum'], 'scalers': ['identity']},
        
        {'name': 'small', 'pna_hidden_dims': [64], 'rep_dim': 16, 
         'head_hidden_dims': [16], 'dropout': 0.0, 'lr': 0.001, 'n_epochs': 500,
         'aggregators': ['mean', 'sum', 'max'], 'scalers': ['identity', 'amplification']},
        
        {'name': 'small_wd', 'pna_hidden_dims': [64], 'rep_dim': 16, 
         'head_hidden_dims': [16], 'dropout': 0.0, 'lr': 0.001, 'n_epochs': 500,
         'weight_decay': 0.05,
         'aggregators': ['mean', 'sum', 'max'], 'scalers': ['identity', 'amplification']},
        
        {'name': 'mean_only', 'pna_hidden_dims': [64], 'rep_dim': 16, 
         'head_hidden_dims': [16], 'dropout': 0.0, 'lr': 0.001, 'n_epochs': 500,
         'weight_decay': 0.01,
         'aggregators': ['mean'], 'scalers': ['identity']},
        
        # ===== BASELINE from previous tuning =====
        {'name': 'shallow_wd01', 'pna_hidden_dims': [128], 'rep_dim': 32, 
         'head_hidden_dims': [32], 'dropout': 0.0, 'lr': 0.001, 'n_epochs': 500,
         'weight_decay': 0.01,
         'aggregators': ['mean', 'sum', 'max', 'std'], 'scalers': ['identity', 'amplification', 'attenuation']},
        
        {'name': 'shallow_wide', 'pna_hidden_dims': [128], 'rep_dim': 32, 
         'head_hidden_dims': [32], 'dropout': 0.0, 'lr': 0.001, 'n_epochs': 500,
         'aggregators': ['mean', 'sum', 'max', 'std'], 'scalers': ['identity', 'amplification', 'attenuation']},
        
        # ===== EARLY STOPPING VARIATIONS =====
        {'name': 'patience50', 'pna_hidden_dims': [128], 'rep_dim': 32, 
         'head_hidden_dims': [32], 'dropout': 0.0, 'lr': 0.001, 'n_epochs': 500,
         'weight_decay': 0.01, 'patience': 50,
         'aggregators': ['mean', 'sum', 'max', 'std'], 'scalers': ['identity', 'amplification', 'attenuation']},
        
        {'name': 'patience30', 'pna_hidden_dims': [128], 'rep_dim': 32, 
         'head_hidden_dims': [32], 'dropout': 0.0, 'lr': 0.001, 'n_epochs': 500,
         'weight_decay': 0.01, 'patience': 30,
         'aggregators': ['mean', 'sum', 'max', 'std'], 'scalers': ['identity', 'amplification', 'attenuation']},
        
        # ===== SLOW LEARNING WITH STRONG REGULARIZATION =====
        {'name': 'slow_wd05', 'pna_hidden_dims': [128], 'rep_dim': 32, 
         'head_hidden_dims': [32], 'dropout': 0.0, 'lr': 0.0005, 'n_epochs': 1000,
         'weight_decay': 0.05,
         'aggregators': ['mean', 'sum', 'max', 'std'], 'scalers': ['identity', 'amplification', 'attenuation']},
        
        {'name': 'vslow_wd1', 'pna_hidden_dims': [128], 'rep_dim': 32, 
         'head_hidden_dims': [32], 'dropout': 0.0, 'lr': 0.0002, 'n_epochs': 2000,
         'weight_decay': 0.1,
         'aggregators': ['mean', 'sum', 'max', 'std'], 'scalers': ['identity', 'amplification', 'attenuation']},
        
        # ===== LINEAR HEAD (simpler predictions) =====
        {'name': 'linear_head', 'pna_hidden_dims': [128], 'rep_dim': 32, 
         'head_hidden_dims': [], 'dropout': 0.0, 'lr': 0.001, 'n_epochs': 500,
         'weight_decay': 0.01,
         'aggregators': ['mean', 'sum', 'max', 'std'], 'scalers': ['identity', 'amplification', 'attenuation']},
        
        {'name': 'linear_head_wd05', 'pna_hidden_dims': [128], 'rep_dim': 32, 
         'head_hidden_dims': [], 'dropout': 0.0, 'lr': 0.001, 'n_epochs': 500,
         'weight_decay': 0.05,
         'aggregators': ['mean', 'sum', 'max', 'std'], 'scalers': ['identity', 'amplification', 'attenuation']},
        
        # ===== WIDER WITH STRONG REGULARIZATION =====
        {'name': 'wide_wd05', 'pna_hidden_dims': [256], 'rep_dim': 64, 
         'head_hidden_dims': [64], 'dropout': 0.0, 'lr': 0.001, 'n_epochs': 500,
         'weight_decay': 0.05,
         'aggregators': ['mean', 'sum', 'max', 'std'], 'scalers': ['identity', 'amplification', 'attenuation']},
        
        {'name': 'wide_dropout', 'pna_hidden_dims': [256], 'rep_dim': 64, 
         'head_hidden_dims': [64], 'dropout': 0.3, 'lr': 0.001, 'n_epochs': 800,
         'aggregators': ['mean', 'sum', 'max', 'std'], 'scalers': ['identity', 'amplification', 'attenuation']},
        
        # ===== TWO LAYER WITH REGULARIZATION =====
        {'name': 'two_layer_wd', 'pna_hidden_dims': [128, 64], 'rep_dim': 32, 
         'head_hidden_dims': [32], 'dropout': 0.0, 'lr': 0.001, 'n_epochs': 500,
         'weight_decay': 0.05,
         'aggregators': ['mean', 'sum', 'max', 'std'], 'scalers': ['identity', 'amplification', 'attenuation']},
        
        {'name': 'two_layer_dropout', 'pna_hidden_dims': [128, 64], 'rep_dim': 32, 
         'head_hidden_dims': [32], 'dropout': 0.3, 'lr': 0.001, 'n_epochs': 800,
         'aggregators': ['mean', 'sum', 'max', 'std'], 'scalers': ['identity', 'amplification', 'attenuation']},
    ]
    
    # Fixed params for all configs (can be overridden by config)
    fixed_params = {
        'lambda_ipm': 0.0,
        'weight_decay': 1e-4,
        'early_stopping': True,
        'patience': 100,
        'verbose': False,
        'return_model': True,
    }
    
    print("\n" + "=" * 80)
    print("TESTING MULTIPLE PNA CONFIGURATIONS")
    print("=" * 80)
    
    results = []
    best_avg_ci_width = float('inf')
    best_config_name = None
    best_model = None
    best_Y_pred = None
    best_ci_widths_pred = None
    best_ci_widths_noharm = None
    best_ci_widths_linear = None
    best_r2 = -np.inf
    
    for config in configs:
        name = config.pop('name')
        # Merge: fixed_params first, then config overrides
        full_config = {**fixed_params, **config}
        
        agg = config.get('aggregators', ['mean', 'sum', 'max', 'std'])
        wd = config.get('weight_decay', fixed_params.get('weight_decay', 1e-4))
        print(f"\n[{name}] hidden={config['pna_hidden_dims']}, rep={config.get('rep_dim', 16)}, "
              f"head={config['head_hidden_dims']}, lr={config['lr']}, wd={wd}")
        
        try:
            # Fit model on normalized Y
            result = PNA_embedding(X, G.astype(float), D, Y_normalized, k=k, **full_config)
            model = result['model']
            
            # Evaluate R2 - denormalize using global mean/std
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
            
            # Evaluate CI widths for pna-pred, pna-no-harm, and pna-linear
            ci_widths_pred, avg_ci_pred, ci_widths_noharm, avg_ci_noharm, ci_widths_linear, avg_ci_linear = compute_ci_width(
                Y, D, X, pi_id_all, G, second_order_matrices, model,
                Y_mean, Y_std, contrasts
            )
            
            # Use average of all three for ranking
            avg_ci_width = (avg_ci_pred + avg_ci_noharm + avg_ci_linear) / 3
            
            results.append({
                'name': name, 'mse': mse, 'rmse': rmse, 'r2': r2, 
                'avg_ci_pred': avg_ci_pred, 'avg_ci_noharm': avg_ci_noharm,
                'avg_ci_linear': avg_ci_linear, 'avg_ci_width': avg_ci_width,
                'ci_widths_pred': ci_widths_pred, 'ci_widths_noharm': ci_widths_noharm,
                'ci_widths_linear': ci_widths_linear,
                **config
            })
            
            # Track best by average CI width
            if avg_ci_width < best_avg_ci_width:
                best_avg_ci_width = avg_ci_width
                best_config_name = name
                best_model = model
                best_Y_pred = Y_pred
                best_ci_widths_pred = ci_widths_pred
                best_ci_widths_noharm = ci_widths_noharm
                best_ci_widths_linear = ci_widths_linear
                best_r2 = r2
                print(f"  -> NEW BEST! R2={r2:.4f}, pred={avg_ci_pred:.0f}, noharm={avg_ci_noharm:.0f}, linear={avg_ci_linear:.0f}")
            else:
                print(f"  -> R2={r2:.4f}, pred={avg_ci_pred:.0f}, noharm={avg_ci_noharm:.0f}, linear={avg_ci_linear:.0f}")
        
        except Exception as e:
            print(f"  -> ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({'name': name, 'mse': np.inf, 'rmse': np.inf, 'r2': -np.inf, 
                           'avg_ci_width': float('inf'), 'avg_ci_pred': float('inf'), 
                           'avg_ci_noharm': float('inf'), 'avg_ci_linear': float('inf'), **config})
        
        # Restore name for next iteration
        config['name'] = name
    
    print("\n" + "=" * 90)
    print("RESULTS SUMMARY (sorted by avg CI width)")
    print("=" * 90)
    print(f"\n{'Config':<20} {'R2':>8} {'pred':>8} {'noharm':>8} {'linear':>8} {'Avg CI':>8}")
    print("-" * 75)
    for r in sorted(results, key=lambda x: x.get('avg_ci_width', float('inf'))):
        avg_ci = r.get('avg_ci_width', float('inf'))
        avg_pred = r.get('avg_ci_pred', float('inf'))
        avg_nh = r.get('avg_ci_noharm', float('inf'))
        avg_lin = r.get('avg_ci_linear', float('inf'))
        if avg_ci < float('inf'):
            print(f"{r['name']:<20} {r['r2']:>8.4f} {avg_pred:>8.0f} {avg_nh:>8.0f} {avg_lin:>8.0f} {avg_ci:>8.0f}")
        else:
            print(f"{r['name']:<20} {r['r2']:>8.4f} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
    
    print(f"\nBest config by CI width: {best_config_name}")
    best_res = next((r for r in results if r['name'] == best_config_name), None)
    if best_res:
        print(f"  R2={best_r2:.4f}, pred={best_res['avg_ci_pred']:.0f}, noharm={best_res['avg_ci_noharm']:.0f}, linear={best_res['avg_ci_linear']:.0f}")
    
    if best_model is not None and best_Y_pred is not None:
        print("\n" + "=" * 80)
        print(f"DETAILED EVALUATION OF BEST MODEL ({best_config_name})")
        print("=" * 80)
        
        Y_pred = best_Y_pred
        valid_mask = ~np.isnan(Y)
        
        # Per-exposure-level metrics
        print(f"\nPer-exposure-level metrics:")
        for d in range(k):
            a = d // 2
            h = d % 2
            mask = (D == d) & valid_mask
            if mask.sum() > 0:
                mse_d = np.mean((Y[mask] - Y_pred[mask])**2)
                rmse_d = np.sqrt(mse_d)
                print(f"  D={d} (A={a}, H={h}, n={mask.sum()}): RMSE={rmse_d:.2f}, mean_Y={Y[mask].mean():.2f}, mean_Yhat={Y_pred[mask].mean():.2f}")
        
        # CI widths for each contrast - pna-pred
        print(f"\nPNA-PRED confidence intervals:")
        if best_ci_widths_pred:
            for contrast_name, ci_info in best_ci_widths_pred.items():
                if 'width' in ci_info:
                    print(f"  {contrast_name}: tau={ci_info['tau']:.0f}, 95% CI=[{ci_info['ci_lo']:.0f}, {ci_info['ci_hi']:.0f}], width={ci_info['width']:.0f}")
                else:
                    print(f"  {contrast_name}: error: {ci_info.get('error', 'unknown')}")
        
        # CI widths for each contrast - pna-no-harm
        print(f"\nPNA-NO-HARM confidence intervals:")
        if best_ci_widths_noharm:
            for contrast_name, ci_info in best_ci_widths_noharm.items():
                if 'width' in ci_info:
                    print(f"  {contrast_name}: tau={ci_info['tau']:.0f}, 95% CI=[{ci_info['ci_lo']:.0f}, {ci_info['ci_hi']:.0f}], width={ci_info['width']:.0f}")
                else:
                    print(f"  {contrast_name}: error: {ci_info.get('error', 'unknown')}")
        
        # CI widths for each contrast - pna-linear
        print(f"\nPNA-LINEAR confidence intervals:")
        if best_ci_widths_linear:
            for contrast_name, ci_info in best_ci_widths_linear.items():
                if 'width' in ci_info:
                    print(f"  {contrast_name}: tau={ci_info['tau']:.0f}, 95% CI=[{ci_info['ci_lo']:.0f}, {ci_info['ci_hi']:.0f}], width={ci_info['width']:.0f}")
                else:
                    print(f"  {contrast_name}: error: {ci_info.get('error', 'unknown')}")
        
        # Print recommended config
        best_cfg = next(c for c in configs if c['name'] == best_config_name)
        print("\n" + "=" * 80)
        print("RECOMMENDED PNA_CONFIG FOR GRR-analysis.py")
        print("=" * 80)
        agg = best_cfg.get('aggregators', ['mean', 'sum', 'max', 'std'])
        scl = best_cfg.get('scalers', ['identity', 'amplification', 'attenuation'])
        wd = best_cfg.get('weight_decay', 1e-4)
        print(f"""
PNA_CONFIG = {{
    'pna_hidden_dims': {best_cfg['pna_hidden_dims']},
    'rep_dim': {best_cfg['rep_dim']},
    'head_hidden_dims': {best_cfg['head_hidden_dims']},
    'aggregators': {agg},
    'scalers': {scl},
    'dropout': {best_cfg['dropout']},
    'lambda_ipm': 0.0,
    'n_epochs': {best_cfg['n_epochs']},
    'lr': {best_cfg['lr']},
    'weight_decay': {wd},
    'early_stopping': True,
    'patience': 100,
}}
""")
        
        # Also compute baseline stratified OLS R2 and CI widths for comparison
        print("\n" + "=" * 80)
        print("COMPARISON WITH STRATIFIED OLS (stratified-pred)")
        print("=" * 80)
        
        from sklearn.linear_model import LinearRegression
        ss_tot = np.sum((Y[valid_mask] - np.mean(Y[valid_mask]))**2)
        
        # Compute stratified-pred CI widths
        print("\nStratified-pred CI widths:")
        strat_total_width = 0
        strat_n_contrasts = 0
        Kn = (G @ G.T > 0).astype(int)
        eps = 1e-12
        
        for d1, d2, contrast_name in contrasts:
            # Fit stratified OLS
            mask_d1 = (D == d1)
            mask_d2 = (D == d2)
            
            if mask_d1.sum() > X.shape[1]:
                model_d1 = LinearRegression()
                model_d1.fit(X[mask_d1], Y[mask_d1])
                Ypred_d1 = model_d1.predict(X)
            else:
                Ypred_d1 = np.full(n, np.mean(Y[mask_d1]) if mask_d1.sum() > 0 else 0)
            
            if mask_d2.sum() > X.shape[1]:
                model_d2 = LinearRegression()
                model_d2.fit(X[mask_d2], Y[mask_d2])
                Ypred_d2 = model_d2.predict(X)
            else:
                Ypred_d2 = np.full(n, np.mean(Y[mask_d2]) if mask_d2.sum() > 0 else 0)
            
            X1_use = Ypred_d1.reshape(-1, 1)
            X2_use = Ypred_d2.reshape(-1, 1)
            beta1 = np.array([1.0])
            beta2 = np.array([1.0])
            
            pi1 = pi_id_all[:, d1]
            pi2 = pi_id_all[:, d2]
            
            f1 = X1_use @ beta1
            f2 = X2_use @ beta2
            
            hat1_d1 = (D == d1).astype(float) / (pi1 + eps)
            hat1_d2 = (D == d2).astype(float) / (pi2 + eps)
            
            mu1 = np.mean(hat1_d1 * Y - hat1_d1 * f1 + f1)
            mu2 = np.mean(hat1_d2 * Y - hat1_d2 * f2 + f2)
            tau_hat = mu1 - mu2
            
            try:
                pi_ij_d1d1 = build_second_order_matrix(second_order_matrices, n, d1)
                pi_ij_d2d2 = build_second_order_matrix(second_order_matrices, n, d2)
                
                if pi_ij_d1d1 is not None and pi_ij_d2d2 is not None:
                    W1_pi, W2_pi = build_W_pi(pi_id_all, pi_ij_d1d1, pi_ij_d2d2, d1, d2, Kn, robust=False)
                    
                    inputs = prepare_grr_inputs(Y, D, X, d1, d2, pi_id_all, X1=X1_use, X2=X2_use, eps=eps)
                    ci_lo, ci_hi = grr_confidence_interval(
                        tau_hat=tau_hat,
                        hatY_d1=inputs['hatY_d1'],
                        hatY_d2=inputs['hatY_d2'],
                        hat1_d1=inputs['hat1_d1'],
                        hat1_d2=inputs['hat1_d2'],
                        X1=X1_use, X2=X2_use,
                        beta1=beta1, beta2=beta2,
                        W1_pi=W1_pi, W2_pi=W2_pi,
                        alpha=0.05
                    )
                    width = ci_hi - ci_lo
                    strat_total_width += width
                    strat_n_contrasts += 1
                    print(f"  {contrast_name}: tau={tau_hat:.0f}, 95% CI=[{ci_lo:.0f}, {ci_hi:.0f}], width={width:.0f}")
            except Exception as e:
                print(f"  {contrast_name}: error - {e}")
        
        strat_avg_width = strat_total_width / strat_n_contrasts if strat_n_contrasts > 0 else float('inf')
        
        print(f"\n  Best PNA avg CI width:     {best_avg_ci_width:.0f}")
        print(f"  Stratified-pred avg CI:    {strat_avg_width:.0f}")
        if best_avg_ci_width < strat_avg_width:
            print(f"  -> PNA has {(strat_avg_width - best_avg_ci_width):.0f} narrower CIs on average!")
        else:
            print(f"  -> Stratified-pred has {(best_avg_ci_width - strat_avg_width):.0f} narrower CIs on average")


if __name__ == '__main__':
    main()
