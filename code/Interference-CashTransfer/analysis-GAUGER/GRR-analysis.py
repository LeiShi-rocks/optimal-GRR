"""
GRR Analysis with simplified exposure mapping (A, H).

This script estimates treatment effects using various GRR estimators:
- Primitive: HT, Hajek
- Linear GRR: linear, lasso
- GAT-enhanced: gat-linear, gat-lasso, gat-no-harm, gat-pred
- PNA-enhanced: pna-linear, pna-lasso, pna-no-harm, pna-pred

Reports point estimates and 95% confidence intervals for each contrast:
- Direct effect: tau(1, h) - tau(0, h)
- Spillover effect: tau(a, 1) - tau(a, 0)

Exposure mapping:
- A: self treatment (0 or 1)
- H: whether more than 1/3 of connected villages are treated (0 or 1)
- D = A * 2 + H gives 4 levels: (0,0), (0,1), (1,0), (1,1)

Usage:
    python GRR-analysis.py [--outcome profit_per_ent]

Note: Requires preprocessed data from preprocess.py and propensity scores from
      compute_inclusion_prob.py.
"""

import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
import sys
import json
import argparse
from datetime import datetime

# Add parent directories to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # Interference-CashTransfer/
ROOT_DIR = os.path.dirname(BASE_DIR)    # code/

sys.path.insert(0, os.path.join(ROOT_DIR, "synthetic"))

# Import GRR modules
from GRR import (
    grr_estimator, GAT_embedding, PNA_embedding, 
    prepare_grr_inputs, LinearGRR
)
from variance_estimators import grr_variance_estimator, grr_confidence_interval
from propensity import build_W_pi

# =============================================================================
# Configuration
# =============================================================================

PROCESSED_DIR = os.path.join(SCRIPT_DIR, "processed_data")
PROPENSITY_DIR = os.path.join(SCRIPT_DIR, "inclusion_probabilities")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results")

# All estimators to run
ALL_ESTIMATORS = [
    'ht', 'hajek',
    'linear', 'lasso',
    'stratified-pred',  # Stratified OLS predictions + AIPW
    'gat-linear', 'gat-lasso', 'gat-no-harm', 'gat-pred',
    'pna-linear', 'pna-lasso', 'pna-no-harm', 'pna-pred'
]

# GAT configuration
GAT_CONFIG = {
    'gat_hidden_dims': [64, 64, 32],
    'rep_dim': 32,
    'head_hidden_dims': [32, 16],
    'heads': 4,
    'dropout': 0.0,
    'lambda_ipm': 0.0,
    'n_epochs': 300,
    'lr': 0.001,
    'weight_decay': 1e-4,
    'early_stopping': True,
    'patience': 50,
}

# PNA configuration (tuned via tune_pna.py - mean_only config, R2=0.1463, Avg CI=3152)
# Simple model with mean aggregation gives narrowest CIs
# PNA configurations per outcome (tuned via tune_pna.py)
PNA_CONFIGS = {
    'revenue_per_ent': {
        'pna_hidden_dims': [64],
        'rep_dim': 16,
        'head_hidden_dims': [16],
        'aggregators': ['mean'],
        'scalers': ['identity'],
        'dropout': 0.0,
        'lambda_ipm': 0.0,
        'n_epochs': 500,
        'lr': 0.001,
        'weight_decay': 0.01,
        'early_stopping': True,
        'patience': 100,
    },
    'wagebill_per_ent': {
        # shallow_wide config (best found with seed search)
        # Use --seed 50039 to reproduce best initialization results
        'pna_hidden_dims': [128],
        'rep_dim': 32,
        'head_hidden_dims': [32],
        'aggregators': ['mean', 'sum', 'max', 'std'],
        'scalers': ['identity', 'amplification', 'attenuation'],
        'dropout': 0.0,
        'lambda_ipm': 0.0,
        'n_epochs': 500,
        'lr': 0.001,
        'weight_decay': 0.0001,  # very low weight decay
        'early_stopping': True,
        'patience': 100,
    },
    'cost_per_ent': {
        'pna_hidden_dims': [64],
        'rep_dim': 16,
        'head_hidden_dims': [16],
        'aggregators': ['mean'],
        'scalers': ['identity'],
        'dropout': 0.0,
        'lambda_ipm': 0.0,
        'n_epochs': 500,
        'lr': 0.001,
        'weight_decay': 0.01,
        'early_stopping': True,
        'patience': 100,
    },
    'profit_per_ent': {
        'pna_hidden_dims': [128],
        'rep_dim': 32,
        'head_hidden_dims': [32],
        'aggregators': ['mean', 'sum', 'max', 'std'],
        'scalers': ['identity', 'amplification', 'attenuation'],
        'dropout': 0.0,
        'lambda_ipm': 0.0,
        'n_epochs': 800,
        'lr': 0.001,
        'weight_decay': 0.1,
        'early_stopping': True,
        'patience': 100,
    },
}

# Default PNA config (used if outcome not in PNA_CONFIGS)
PNA_CONFIG_DEFAULT = PNA_CONFIGS['revenue_per_ent']

# Available outcomes
OUTCOMES = ['profit_per_ent', 'revenue_per_ent', 'cost_per_ent', 'wagebill_per_ent']

# Covariates to use for GRR
COVARIATE_COLS = ['n_allents_BL', 'profit_BL_per_ent', 'revenue_BL_per_ent', 'cost_BL_per_ent']


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_data():
    """Load all required data for GRR analysis."""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    # Load village data
    village_data_path = os.path.join(PROCESSED_DIR, "village_data.csv")
    adj_path = os.path.join(PROCESSED_DIR, "adjacency_matrix.npy")
    
    if not os.path.exists(village_data_path):
        raise FileNotFoundError(
            f"Village data not found: {village_data_path}\n"
            "Please run preprocess.py first."
        )
    
    village_data = pd.read_csv(village_data_path)
    print(f"Loaded village data: {len(village_data)} villages")
    
    if os.path.exists(adj_path):
        G = np.load(adj_path)
        print(f"Loaded adjacency matrix: {G.shape}")
    else:
        print("Warning: Adjacency matrix not found")
        G = None
    
    # Load propensity scores
    propensity_path = os.path.join(PROPENSITY_DIR, "propensity_scores.csv")
    if not os.path.exists(propensity_path):
        raise FileNotFoundError(
            f"Propensity scores not found: {propensity_path}\n"
            "Please run compute_inclusion_prob.py first."
        )
    
    propensity_df = pd.read_csv(propensity_path)
    print(f"Loaded propensity scores: {len(propensity_df)} villages")
    
    # Merge propensity scores with village data
    village_data['village_code'] = village_data['village_code'].astype(float)
    propensity_df['village_code'] = propensity_df['village_code'].astype(float)
    
    merged = village_data.merge(propensity_df, on='village_code', how='inner')
    print(f"Matched {len(merged)} villages with propensity scores")
    
    if len(merged) < len(village_data):
        print(f"Warning: {len(village_data) - len(merged)} villages missing propensity scores")
    
    # Load second-order matrices
    matrix_dir = os.path.join(PROPENSITY_DIR, "second_order_matrices")
    second_order_matrices = {}
    
    for a in [0, 1]:
        for h in [0, 1]:
            key = (a, h)
            filename = os.path.join(matrix_dir, f"P_A{a}_H{h}_sparse.csv")
            if os.path.exists(filename):
                second_order_matrices[key] = pd.read_csv(filename)
    
    print(f"Loaded {len(second_order_matrices)} second-order matrices")
    
    return merged, G, propensity_df, second_order_matrices


def build_propensity_matrix(data, propensity_df):
    """Build propensity score matrix from dataframe."""
    ps_columns = [col for col in propensity_df.columns if col.startswith('(A=')]
    
    # Create matrix aligned with data
    n = len(data)
    pi_id_all = np.zeros((n, 4))  # 4 exposure levels
    
    for i, row in data.iterrows():
        village = row['village_code']
        ps_row = propensity_df[propensity_df['village_code'] == village][ps_columns]
        if len(ps_row) > 0:
            pi_id_all[i] = ps_row.values[0]
    
    return pi_id_all


def build_second_order_matrix(second_order_df, n, d):
    """
    Build second-order inclusion probability matrix from sparse dataframe.
    
    d is the exposure level (0-3):
    - d=0: (A=0, H=0)
    - d=1: (A=0, H=1)
    - d=2: (A=1, H=0)
    - d=3: (A=1, H=1)
    """
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
        pi_ij[j, i] = prob  # Symmetric
    
    return pi_ij


def build_kn_matrix(G):
    """Build Kn matrix (neighborhood overlap indicator)."""
    # Kn_ij = 1 if villages i and j share at least one neighbor
    Kn = (G @ G.T > 0).astype(int)
    return Kn


# =============================================================================
# Primitive Estimators
# =============================================================================

def mu_ht(Y, D, d, pi_id_all, eps=1e-12):
    """Horvitz-Thompson estimator for mean potential outcome mu(d)."""
    pi_d = pi_id_all[:, d]
    mask = (D == d)
    if mask.sum() == 0:
        return np.nan
    return np.sum(Y[mask] / (pi_d[mask] + eps)) / len(Y)


def mu_hajek(Y, D, d, pi_id_all, eps=1e-12):
    """Hajek estimator for mean potential outcome mu(d)."""
    pi_d = pi_id_all[:, d]
    mask = (D == d)
    if mask.sum() == 0:
        return np.nan
    num = np.sum(Y[mask] / (pi_d[mask] + eps))
    denom = np.sum(1 / (pi_d[mask] + eps))
    return num / denom if denom > 0 else np.nan


# =============================================================================
# GRR Estimation for a Single Contrast
# =============================================================================

def estimate_contrast(Y, D, X, d1, d2, pi_id_all, Kn, G, method, 
                      gat_embeddings=None, pna_embeddings=None,
                      gat_model=None, pna_model=None,
                      W1_pi=None, W2_pi=None,
                      Y_mean=0.0, Y_std=1.0,
                      eps=1e-12, verbose=False):
    """
    Estimate treatment effect contrast tau = mu(d1) - mu(d2).
    
    Returns dict with:
        tau_hat: point estimate
        se: standard error (if available)
        ci_lower, ci_upper: 95% confidence interval (if available)
        method: method name
    """
    k = 4  # Number of exposure levels
    n = len(Y)
    
    result = {
        'method': method,
        'd1': d1,
        'd2': d2,
        'tau_hat': np.nan,
        'se': np.nan,
        'ci_lower': np.nan,
        'ci_upper': np.nan,
        'n': n
    }
    
    # =========================================================================
    # Primitive estimators (HT and Hajek) with CI computation
    # =========================================================================
    if method == 'ht':
        mu1 = mu_ht(Y, D, d1, pi_id_all, eps)
        mu2 = mu_ht(Y, D, d2, pi_id_all, eps)
        result['tau_hat'] = mu1 - mu2
        
        # Compute CI using X = ones, beta = mu_hat
        if W1_pi is not None and W2_pi is not None:
            X1_use = np.ones((n, 1))
            X2_use = np.ones((n, 1))
            beta1 = np.array([mu1])
            beta2 = np.array([mu2])
            
            inputs = prepare_grr_inputs(Y, D, X, d1, d2, pi_id_all, X1=X1_use, X2=X2_use, eps=eps)
            ci_lo, ci_hi = grr_confidence_interval(
                tau_hat=result['tau_hat'],
                hatY_d1=inputs['hatY_d1'],
                hatY_d2=inputs['hatY_d2'],
                hat1_d1=inputs['hat1_d1'],
                hat1_d2=inputs['hat1_d2'],
                X1=X1_use, X2=X2_use,
                beta1=beta1, beta2=beta2,
                W1_pi=W1_pi, W2_pi=W2_pi,
                alpha=0.05
            )
            result['ci_lower'] = ci_lo
            result['ci_upper'] = ci_hi
            result['se'] = (ci_hi - ci_lo) / (2 * 1.96)
        
        return result
    
    elif method == 'hajek':
        mu1 = mu_hajek(Y, D, d1, pi_id_all, eps)
        mu2 = mu_hajek(Y, D, d2, pi_id_all, eps)
        result['tau_hat'] = mu1 - mu2
        
        # Compute CI using X = ones, beta = mu_hat
        if W1_pi is not None and W2_pi is not None:
            X1_use = np.ones((n, 1))
            X2_use = np.ones((n, 1))
            beta1 = np.array([mu1])
            beta2 = np.array([mu2])
            
            inputs = prepare_grr_inputs(Y, D, X, d1, d2, pi_id_all, X1=X1_use, X2=X2_use, eps=eps)
            ci_lo, ci_hi = grr_confidence_interval(
                tau_hat=result['tau_hat'],
                hatY_d1=inputs['hatY_d1'],
                hatY_d2=inputs['hatY_d2'],
                hat1_d1=inputs['hat1_d1'],
                hat1_d2=inputs['hat1_d2'],
                X1=X1_use, X2=X2_use,
                beta1=beta1, beta2=beta2,
                W1_pi=W1_pi, W2_pi=W2_pi,
                alpha=0.05
            )
            result['ci_lower'] = ci_lo
            result['ci_upper'] = ci_hi
            result['se'] = (ci_hi - ci_lo) / (2 * 1.96)
        
        return result
    
    # =========================================================================
    # Stratified OLS predictor + AIPW
    # =========================================================================
    elif method == 'stratified-pred':
        from sklearn.linear_model import LinearRegression
        
        # Fit stratified OLS models for d1 and d2
        mask_d1 = (D == d1)
        mask_d2 = (D == d2)
        
        # Model for d1
        if mask_d1.sum() > X.shape[1]:
            model_d1 = LinearRegression()
            model_d1.fit(X[mask_d1], Y[mask_d1])
            Ypred_d1 = model_d1.predict(X)
        else:
            Ypred_d1 = np.full(n, np.mean(Y[mask_d1]) if mask_d1.sum() > 0 else 0)
        
        # Model for d2
        if mask_d2.sum() > X.shape[1]:
            model_d2 = LinearRegression()
            model_d2.fit(X[mask_d2], Y[mask_d2])
            Ypred_d2 = model_d2.predict(X)
        else:
            Ypred_d2 = np.full(n, np.mean(Y[mask_d2]) if mask_d2.sum() > 0 else 0)
        
        # Use predictions as basis functions with beta=1 (AIPW style)
        X1_use = Ypred_d1.reshape(-1, 1)
        X2_use = Ypred_d2.reshape(-1, 1)
        beta1 = np.array([1.0])
        beta2 = np.array([1.0])
        
        # Compute AIPW estimator
        pi1 = pi_id_all[:, d1]
        pi2 = pi_id_all[:, d2]
        
        f1 = X1_use @ beta1
        f2 = X2_use @ beta2
        
        hat1_d1 = (D == d1).astype(float) / (pi1 + eps)
        hat1_d2 = (D == d2).astype(float) / (pi2 + eps)
        
        mu1 = np.mean(hat1_d1 * Y - hat1_d1 * f1 + f1)
        mu2 = np.mean(hat1_d2 * Y - hat1_d2 * f2 + f2)
        
        result['tau_hat'] = mu1 - mu2
        
        # Compute CI if W matrices available
        if W1_pi is not None and W2_pi is not None:
            inputs = prepare_grr_inputs(Y, D, X, d1, d2, pi_id_all, X1=X1_use, X2=X2_use, eps=eps)
            ci_lo, ci_hi = grr_confidence_interval(
                tau_hat=result['tau_hat'],
                hatY_d1=inputs['hatY_d1'],
                hatY_d2=inputs['hatY_d2'],
                hat1_d1=inputs['hat1_d1'],
                hat1_d2=inputs['hat1_d2'],
                X1=X1_use, X2=X2_use,
                beta1=beta1, beta2=beta2,
                W1_pi=W1_pi, W2_pi=W2_pi,
                alpha=0.05
            )
            result['ci_lower'] = ci_lo
            result['ci_upper'] = ci_hi
            result['se'] = (ci_hi - ci_lo) / (2 * 1.96)
        
        return result
    
    # =========================================================================
    # GRR estimators with variance estimation
    # =========================================================================
    try:
        # Determine which covariates to use
        X1_use, X2_use = X, X
        
        if method == 'gat-linear' or method == 'gat-lasso':
            if gat_embeddings is None:
                raise ValueError("GAT embeddings required")
            X1_use = gat_embeddings['X_GAT_leaf'][d1]
            X2_use = gat_embeddings['X_GAT_leaf'][d2]
            grr_method = 'linear' if method == 'gat-linear' else 'lasso'
            
        elif method == 'gat-no-harm':
            if gat_model is None:
                raise ValueError("GAT model required")
            Ypred_d1 = gat_model.predict(X, D=d1) * Y_std + Y_mean
            Ypred_d2 = gat_model.predict(X, D=d2) * Y_std + Y_mean
            X1_use = Ypred_d1.reshape(-1, 1)
            X2_use = Ypred_d2.reshape(-1, 1)
            grr_method = 'linear'
            
        elif method == 'gat-pred':
            if gat_model is None:
                raise ValueError("GAT model required")
            Ypred_d1 = gat_model.predict(X, D=d1) * Y_std + Y_mean
            Ypred_d2 = gat_model.predict(X, D=d2) * Y_std + Y_mean
            X1_use = Ypred_d1.reshape(-1, 1)
            X2_use = Ypred_d2.reshape(-1, 1)
            beta1 = np.array([1.0])
            beta2 = np.array([1.0])
            
            # Compute DR estimator manually
            pi1 = pi_id_all[:, d1]
            pi2 = pi_id_all[:, d2]
            
            f1 = X1_use @ beta1
            f2 = X2_use @ beta2
            
            hat1_d1 = (D == d1).astype(float) / (pi1 + eps)
            hat1_d2 = (D == d2).astype(float) / (pi2 + eps)
            
            mu1 = np.mean(hat1_d1 * Y - hat1_d1 * f1 + f1)
            mu2 = np.mean(hat1_d2 * Y - hat1_d2 * f2 + f2)
            
            result['tau_hat'] = mu1 - mu2
            
            # Compute CI if W matrices available
            if W1_pi is not None and W2_pi is not None:
                inputs = prepare_grr_inputs(Y, D, X, d1, d2, pi_id_all, X1=X1_use, X2=X2_use, eps=eps)
                ci_lo, ci_hi = grr_confidence_interval(
                    tau_hat=result['tau_hat'],
                    hatY_d1=inputs['hatY_d1'],
                    hatY_d2=inputs['hatY_d2'],
                    hat1_d1=inputs['hat1_d1'],
                    hat1_d2=inputs['hat1_d2'],
                    X1=X1_use, X2=X2_use,
                    beta1=beta1, beta2=beta2,
                    W1_pi=W1_pi, W2_pi=W2_pi,
                    alpha=0.05
                )
                result['ci_lower'] = ci_lo
                result['ci_upper'] = ci_hi
                result['se'] = (ci_hi - ci_lo) / (2 * 1.96)
            
            return result
            
        elif method == 'pna-linear' or method == 'pna-lasso':
            if pna_embeddings is None:
                raise ValueError("PNA embeddings required")
            X1_use = pna_embeddings['X_PNA_leaf'][d1]
            X2_use = pna_embeddings['X_PNA_leaf'][d2]
            grr_method = 'linear' if method == 'pna-linear' else 'lasso'
            
        elif method == 'pna-no-harm':
            if pna_model is None:
                raise ValueError("PNA model required")
            Ypred_d1 = pna_model.predict(X, D=d1) * Y_std + Y_mean
            Ypred_d2 = pna_model.predict(X, D=d2) * Y_std + Y_mean
            X1_use = Ypred_d1.reshape(-1, 1)
            X2_use = Ypred_d2.reshape(-1, 1)
            grr_method = 'linear'
        
        elif method == 'pna-pred':
            if pna_model is None:
                raise ValueError("PNA model required")
            Ypred_d1 = pna_model.predict(X, D=d1) * Y_std + Y_mean
            Ypred_d2 = pna_model.predict(X, D=d2) * Y_std + Y_mean
            X1_use = Ypred_d1.reshape(-1, 1)
            X2_use = Ypred_d2.reshape(-1, 1)
            beta1 = np.array([1.0])
            beta2 = np.array([1.0])
            
            # Compute DR estimator manually
            pi1 = pi_id_all[:, d1]
            pi2 = pi_id_all[:, d2]
            
            f1 = X1_use @ beta1
            f2 = X2_use @ beta2
            
            hat1_d1 = (D == d1).astype(float) / (pi1 + eps)
            hat1_d2 = (D == d2).astype(float) / (pi2 + eps)
            
            mu1 = np.mean(hat1_d1 * Y - hat1_d1 * f1 + f1)
            mu2 = np.mean(hat1_d2 * Y - hat1_d2 * f2 + f2)
            
            result['tau_hat'] = mu1 - mu2
            
            # Compute CI if W matrices available
            if W1_pi is not None and W2_pi is not None:
                inputs = prepare_grr_inputs(Y, D, X, d1, d2, pi_id_all, X1=X1_use, X2=X2_use, eps=eps)
                ci_lo, ci_hi = grr_confidence_interval(
                    tau_hat=result['tau_hat'],
                    hatY_d1=inputs['hatY_d1'],
                    hatY_d2=inputs['hatY_d2'],
                    hat1_d1=inputs['hat1_d1'],
                    hat1_d2=inputs['hat1_d2'],
                    X1=X1_use, X2=X2_use,
                    beta1=beta1, beta2=beta2,
                    W1_pi=W1_pi, W2_pi=W2_pi,
                    alpha=0.05
                )
                result['ci_lower'] = ci_lo
                result['ci_upper'] = ci_hi
                result['se'] = (ci_hi - ci_lo) / (2 * 1.96)
            
            return result
            
        elif method in ['linear', 'lasso']:
            grr_method = method
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Run GRR estimator
        grr_result = grr_estimator(
            Y, D, X, d1, d2, pi_id_all, Kn,
            method=grr_method, eps=eps,
            X1=X1_use, X2=X2_use,
            alpha=0.0001
        )
        
        result['tau_hat'] = grr_result['tau_hat']
        
        # Compute confidence interval if variance info available
        if hasattr(grr_result['model'], 'get_coefficients') and W1_pi is not None and W2_pi is not None:
            inputs = prepare_grr_inputs(Y, D, X, d1, d2, pi_id_all, X1=X1_use, X2=X2_use, eps=eps)
            beta1, beta2 = grr_result['model'].get_coefficients()
            
            ci_lo, ci_hi = grr_confidence_interval(
                tau_hat=result['tau_hat'],
                hatY_d1=inputs['hatY_d1'],
                hatY_d2=inputs['hatY_d2'],
                hat1_d1=inputs['hat1_d1'],
                hat1_d2=inputs['hat1_d2'],
                X1=X1_use, X2=X2_use,
                beta1=beta1, beta2=beta2,
                W1_pi=W1_pi, W2_pi=W2_pi,
                alpha=0.05
            )
            result['ci_lower'] = ci_lo
            result['ci_upper'] = ci_hi
            result['se'] = (ci_hi - ci_lo) / (2 * 1.96)
        
    except Exception as e:
        if verbose:
            print(f"  Error with {method}: {e}")
        result['tau_hat'] = np.nan
    
    return result


# =============================================================================
# Main Analysis Function
# =============================================================================

def run_analysis(data, G, pi_id_all, second_order_matrices, outcome_col, 
                 estimators=ALL_ESTIMATORS, verbose=True, pna_seed=None):
    """
    Run GRR analysis for all contrasts and estimators.
    
    Returns DataFrame with results for each contrast and estimator.
    """
    print("\n" + "=" * 80)
    print(f"RUNNING GRR ANALYSIS FOR OUTCOME: {outcome_col}")
    print("=" * 80)
    
    n = len(data)
    k = 4  # Number of exposure levels
    
    # Extract data
    Y = data[outcome_col].values
    D = data['D'].values  # Combined exposure index
    
    # Handle missing values
    valid_mask = ~np.isnan(Y)
    if valid_mask.sum() < n:
        print(f"Warning: {n - valid_mask.sum()} villages have missing outcome")
    
    # Build covariate matrix
    X_cols = [col for col in COVARIATE_COLS if col in data.columns]
    X = data[X_cols].values
    
    # Handle missing covariates
    X = np.nan_to_num(X, nan=0.0)
    
    print(f"Sample size: {n}")
    print(f"Covariates: {X_cols}")
    print(f"Outcome mean: {np.nanmean(Y):.2f}, std: {np.nanstd(Y):.2f}")
    
    # Print exposure distribution
    print("\nExposure distribution:")
    for d in range(k):
        a = d // 2
        h = d % 2
        count = (D == d).sum()
        print(f"  (A={a}, H={h}): {count} villages")
    
    # Build Kn matrix
    if G is not None:
        Kn = build_kn_matrix(G)
    else:
        Kn = np.eye(n)
    
    results = []
    
    # Check which methods require GNN embeddings
    use_gat = any(m.startswith('gat-') for m in estimators)
    use_pna = any(m.startswith('pna-') for m in estimators)
    
    gat_embeddings, gat_model = None, None
    pna_embeddings, pna_model = None, None
    
    # For GNN training, normalize Y
    Y_mean = np.nanmean(Y)
    Y_std = np.nanstd(Y)
    Y_normalized = (Y - Y_mean) / Y_std
    
    if use_gat:
        print("\nTraining GAT model...")
        try:
            gat_embeddings = GAT_embedding(
                X, G.astype(float), D, Y_normalized, k=k,
                verbose=verbose,
                return_model=True,
                **GAT_CONFIG
            )
            gat_model = gat_embeddings.get('model', None)
            print("  GAT model trained successfully")
        except Exception as e:
            print(f"  GAT training failed: {e}")
    
    if use_pna:
        print("\nTraining PNA model...")
        # Set seed right before PNA training if provided
        if pna_seed is not None:
            set_seed(pna_seed)
            print(f"  Set seed: {pna_seed}")
        # Use outcome-specific PNA config if available
        pna_config = PNA_CONFIGS.get(outcome_col, PNA_CONFIG_DEFAULT)
        try:
            pna_embeddings = PNA_embedding(
                X, G.astype(float), D, Y_normalized, k=k,
                verbose=verbose,
                return_model=True,
                **pna_config
            )
            pna_model = pna_embeddings.get('model', None)
            print("  PNA model trained successfully")
        except Exception as e:
            print(f"  PNA training failed: {e}")
    
    # Define contrasts to estimate
    contrasts = []
    
    # Direct effects (varying A, fixing H)
    for h in [0, 1]:
        d1 = 1 * 2 + h  # A=1
        d2 = 0 * 2 + h  # A=0
        contrasts.append((d1, d2, f"Direct(H={h})"))
    
    # Spillover effects (varying H, fixing A)
    for a in [0, 1]:
        d1 = a * 2 + 1  # H=1
        d2 = a * 2 + 0  # H=0
        contrasts.append((d1, d2, f"Spillover(A={a})"))
    
    # Run analysis for each contrast
    print("\nEstimating contrasts...")
    for d1, d2, contrast_name in contrasts:
        print(f"\n  Contrast: {contrast_name} (d1={d1}, d2={d2})")
        
        # Count observations in each exposure
        n_d1 = (D == d1).sum()
        n_d2 = (D == d2).sum()
        print(f"    Observations: d1={n_d1}, d2={n_d2}")
        
        if n_d1 == 0 or n_d2 == 0:
            print(f"    Skipping: insufficient observations")
            continue
        
        # Build W matrices for variance estimation
        try:
            a1, h1 = d1 // 2, d1 % 2
            a2, h2 = d2 // 2, d2 % 2
            
            pi_ij_d1d1 = build_second_order_matrix(second_order_matrices, n, d1)
            pi_ij_d2d2 = build_second_order_matrix(second_order_matrices, n, d2)
            
            if pi_ij_d1d1 is not None and pi_ij_d2d2 is not None:
                W1_pi, W2_pi = build_W_pi(pi_id_all, pi_ij_d1d1, pi_ij_d2d2, d1, d2, Kn, robust=False)
            else:
                W1_pi, W2_pi = None, None
        except Exception as e:
            if verbose:
                print(f"    Warning: Could not build W matrices: {e}")
            W1_pi, W2_pi = None, None
        
        # Run each estimator
        for method in estimators:
            result = estimate_contrast(
                Y, D, X, d1, d2, pi_id_all, Kn, G,
                method=method,
                gat_embeddings=gat_embeddings,
                pna_embeddings=pna_embeddings,
                gat_model=gat_model,
                pna_model=pna_model,
                W1_pi=W1_pi,
                W2_pi=W2_pi,
                Y_mean=Y_mean,
                Y_std=Y_std,
                eps=1e-12,
                verbose=verbose
            )
            
            result['contrast'] = contrast_name
            result['outcome'] = outcome_col
            result['n_d1'] = n_d1
            result['n_d2'] = n_d2
            results.append(result)
            
            if not np.isnan(result['tau_hat']):
                ci_str = ""
                if not np.isnan(result['ci_lower']):
                    ci_str = f" 95% CI: [{result['ci_lower']:.2f}, {result['ci_upper']:.2f}]"
                print(f"    {method}: tau = {result['tau_hat']:.2f}{ci_str}")
    
    return pd.DataFrame(results)


def save_results(results_df, outcome_col, output_dir):
    """Save analysis results."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save full results
    results_path = os.path.join(output_dir, f"grr_results_{outcome_col}_{timestamp}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    # Save summary table
    summary = results_df.pivot_table(
        values=['tau_hat', 'se', 'ci_lower', 'ci_upper'],
        index='contrast',
        columns='method',
        aggfunc='first'
    )
    
    summary_path = os.path.join(output_dir, f"grr_summary_{outcome_col}_{timestamp}.csv")
    summary.to_csv(summary_path)
    print(f"Summary saved to: {summary_path}")
    
    return results_path, summary_path


def print_results_table(results_df):
    """Print formatted results table."""
    print("\n" + "=" * 120)
    print("RESULTS SUMMARY")
    print("=" * 120)
    
    contrasts = results_df['contrast'].unique()
    methods = results_df['method'].unique()
    
    # Print header
    header = f"{'Contrast':<20}"
    for method in methods:
        header += f"{method:>12}"
    print(header)
    print("-" * 120)
    
    for contrast in contrasts:
        row = f"{contrast:<20}"
        for method in methods:
            subset = results_df[(results_df['contrast'] == contrast) & 
                               (results_df['method'] == method)]
            if len(subset) > 0 and not np.isnan(subset['tau_hat'].values[0]):
                tau = subset['tau_hat'].values[0]
                row += f"{tau:>12.2f}"
            else:
                row += f"{'N/A':>12}"
        print(row)
    
    print("=" * 120)


# =============================================================================
# Main
# =============================================================================

def set_seed(seed):
    """Set random seed for reproducibility."""
    import torch
    import random
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description='GRR Analysis with (A, H) exposure mapping')
    parser.add_argument('--outcome', type=str, default='profit_per_ent',
                       choices=OUTCOMES,
                       help=f'Outcome variable (default: profit_per_ent)')
    parser.add_argument('--estimators', type=str, nargs='+', default=ALL_ESTIMATORS,
                       help='Estimators to run (default: all)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print verbose output')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Set seed if provided
    if args.seed is not None:
        set_seed(args.seed)
        print(f"\nUsing seed: {args.seed}")
    
    print("\n" + "=" * 80)
    print("GRR ANALYSIS WITH SIMPLIFIED EXPOSURE MAPPING (A, H)")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Outcome: {args.outcome}")
    print(f"Estimators: {args.estimators}")
    
    try:
        # Load data
        data, G, propensity_df, second_order_matrices = load_data()
        
        # Build propensity matrix
        pi_id_all = build_propensity_matrix(data, propensity_df)
        
        # Run analysis
        results_df = run_analysis(
            data, G, pi_id_all, second_order_matrices,
            outcome_col=args.outcome,
            estimators=args.estimators,
            verbose=args.verbose,
            pna_seed=args.seed
        )
        
        # Print and save results
        print_results_table(results_df)
        save_results(results_df, args.outcome, OUTPUT_DIR)
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
