"""
Simulation Out1 - DGP-2: PARALLELIZED Version

This script runs the simulation with parallel Monte Carlo replications.
Uses joblib for CPU parallelization and limits GPU concurrency.

Usage:
    python simulation-out1-dgp2-parallel.py
    
    # With nohup (background):
    nohup python simulation-out1-dgp2-parallel.py > dgp-2/results/simulation_out1_dgp2_parallel.log 2>&1 &
"""

import os
import sys
import numpy as np
from datetime import datetime
from joblib import Parallel, delayed
import multiprocessing as mp

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import DGP-2 functions
import importlib.util
_dgp2_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dgp-2.py')
_spec = importlib.util.spec_from_file_location("dgp2", _dgp2_path)
_dgp2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dgp2)

generate_graph = _dgp2.generate_graph
generate_covariates = _dgp2.generate_covariates
generate_potential_outcomes = _dgp2.generate_potential_outcomes
assign_treatment = _dgp2.assign_treatment
compute_exposure = _dgp2.compute_exposure
observed_outcome = _dgp2.observed_outcome

# Import other modules
from propensity import monte_carlo_probabilities, build_lambdas, build_W_pi
from GRR import grr_estimator, GAT_embedding, PNA_embedding, prepare_grr_inputs
from variance_estimators import grr_confidence_interval
from estimators import (
    mu_ht, mu_haj, 
    _weights, _Omega_const, _vec_const, _solve, _mu_from_c, _stack_for_reg,
    _Omega_cov, _vec_cov, _mu_from_beta
)
from sklearn.linear_model import LinearRegression

# =============================================================================
# Configuration
# =============================================================================

# Methods to compare
ALL_METHODS = [
    'linear-no-cal',  # Only run this new method
]

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

# Simulation parameters
N = 2000
AVG_DEGREE = 3.0
MAX_DEGREE = 9
K = 3
TREATED_FRACTION = 1/3
D1, D2 = 0, 2
MC_REPS = 500
SEED_MC = 111
SEED_GRAPH = 123
SEED_COV = 1234
SEED_OUT = 999
B = 100  # Number of MC replications (production run)
N_JOBS = 3  # Number of parallel workers (limit GPU concurrency)
EPS = 1e-12


# =============================================================================
# Helper: Single MC Replication
# =============================================================================

def run_single_replication(
    b, G, H, X, X_run, X_noi, Y_pot, A_adj,
    pi_id_all, Kn, L1, L2, L12, W1_pi, W2_pi,
    d1, d2, k, n, treated_fraction, true_tau,
    methods, gat_kwargs, pna_kwargs, grr_kwargs, eps,
    gpu_id=0
):
    """Run a single MC replication."""
    # Set GPU device for this worker
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Random treatment assignment (use b as part of seed for reproducibility)
    np.random.seed(b * 12345 + 42)
    A_treat = assign_treatment(n, treated_fraction=treated_fraction, seed=None)
    D = compute_exposure(G, A_treat, k)
    Y = observed_outcome(Y_pot, D, k=k)
    
    pi1 = pi_id_all[:, d1]
    pi2 = pi_id_all[:, d2]
    
    # Results storage
    results_b = {}
    coverage_b = {}
    power_b = {}
    
    # Check which GNN methods are needed
    methods_set = set(methods)
    GAT_METHODS = {'gat-linear', 'gat-lasso', 'gat-no-harm', 'gat-pred'}
    PNA_METHODS = {'pna-linear', 'pna-lasso', 'pna-no-harm', 'pna-pred'}
    use_gat = False  # Not needed for linear-no-cal
    use_pna = False  # Not needed for linear-no-cal
    
    # Train GNN models if needed
    gat_embeddings, gat_model = None, None
    pna_embeddings, pna_model = None, None
    
    if use_gat:
        try:
            gat_embeddings = GAT_embedding(
                X_run, A_adj, D, Y, k=k, d1=d1, d2=d2,
                verbose=False, return_model=True, **gat_kwargs
            )
            gat_model = gat_embeddings.get('model', None)
        except Exception as e:
            print(f"Rep {b}: GAT failed - {e}")
    
    if use_pna:
        try:
            pna_embeddings = PNA_embedding(
                X_run, A_adj, D, Y, k=k, d1=d1, d2=d2,
                verbose=False, return_model=True, **pna_kwargs
            )
            pna_model = pna_embeddings.get('model', None)
        except Exception as e:
            print(f"Rep {b}: PNA failed - {e}")
    
    # Run each method
    for method in methods:
        try:
            tau_hat, has_var, var_info = _run_method(
                method, Y, D, X_run, d1, d2, n, pi1, pi2, pi_id_all, Kn, eps,
                L1, L2, L12, gat_embeddings, pna_embeddings, gat_model, pna_model,
                grr_kwargs
            )
            results_b[method] = tau_hat
            
            # Compute CI if variance info available
            if has_var:
                try:
                    ci_lo, ci_hi = grr_confidence_interval(
                        tau_hat=tau_hat,
                        hatY_d1=var_info['hatY_d1'],
                        hatY_d2=var_info['hatY_d2'],
                        hat1_d1=var_info['hat1_d1'],
                        hat1_d2=var_info['hat1_d2'],
                        X1=var_info['X1'],
                        X2=var_info['X2'],
                        beta1=var_info['beta1'],
                        beta2=var_info['beta2'],
                        W1_pi=W1_pi,
                        W2_pi=W2_pi,
                        alpha=0.05
                    )
                    coverage_b[method] = 1.0 if (ci_lo <= true_tau <= ci_hi) else 0.0
                    power_b[method] = 1.0 if (ci_lo > 0 or ci_hi < 0) else 0.0
                except:
                    coverage_b[method] = np.nan
                    power_b[method] = np.nan
            else:
                coverage_b[method] = np.nan
                power_b[method] = np.nan
                
        except Exception as e:
            print(f"Rep {b}, method {method}: {e}")
            results_b[method] = np.nan
            coverage_b[method] = np.nan
            power_b[method] = np.nan
    
    return b, results_b, coverage_b, power_b


def _run_method(method, Y, D, X_run, d1, d2, n, pi1, pi2, pi_id_all, Kn, eps,
                L1, L2, L12, gat_embeddings, pna_embeddings, gat_model, pna_model,
                grr_kwargs):
    """Run a single estimation method. Returns (tau_hat, has_variance_info, var_info_dict)."""
    
    if method == 'ht':
        mu1 = mu_ht(Y, D, d1, pi_id_all, eps)
        mu2 = mu_ht(Y, D, d2, pi_id_all, eps)
        tau_hat = float(mu1 - mu2)
        X_ones = np.ones((n, 1))
        inputs = prepare_grr_inputs(Y, D, X_run, d1, d2, pi_id_all, X1=X_ones, X2=X_ones, eps=eps)
        var_info = {
            'hatY_d1': inputs['hatY_d1'], 'hatY_d2': inputs['hatY_d2'],
            'hat1_d1': inputs['hat1_d1'], 'hat1_d2': inputs['hat1_d2'],
            'X1': X_ones, 'X2': X_ones,
            'beta1': np.array([mu1]), 'beta2': np.array([mu2])
        }
        return tau_hat, True, var_info
    
    elif method == 'hajek':
        mu1 = mu_haj(Y, D, d1, pi_id_all, eps)
        mu2 = mu_haj(Y, D, d2, pi_id_all, eps)
        tau_hat = float(mu1 - mu2)
        X_ones = np.ones((n, 1))
        inputs = prepare_grr_inputs(Y, D, X_run, d1, d2, pi_id_all, X1=X_ones, X2=X_ones, eps=eps)
        var_info = {
            'hatY_d1': inputs['hatY_d1'], 'hatY_d2': inputs['hatY_d2'],
            'hat1_d1': inputs['hat1_d1'], 'hat1_d2': inputs['hat1_d2'],
            'X1': X_ones, 'X2': X_ones,
            'beta1': np.array([mu1]), 'beta2': np.array([mu2])
        }
        return tau_hat, True, var_info
    
    elif method == 'linear-no-cal':
        # Fit separate OLS models for each exposure level on observed data
        # Then use predictions directly for AIPW (no calibration)
        
        # Prepare X with intercept
        X_with_intercept = np.column_stack([np.ones(n), X_run])
        
        # Fit OLS for exposure d1
        mask_d1 = (D == d1)
        if np.sum(mask_d1) > X_with_intercept.shape[1]:
            ols_d1 = LinearRegression(fit_intercept=False)
            ols_d1.fit(X_with_intercept[mask_d1], Y[mask_d1])
            beta1 = ols_d1.coef_
            Ypred_d1 = X_with_intercept @ beta1
        else:
            # Not enough data, use mean
            beta1 = np.zeros(X_with_intercept.shape[1])
            beta1[0] = np.mean(Y[mask_d1]) if np.sum(mask_d1) > 0 else 0
            Ypred_d1 = X_with_intercept @ beta1
        
        # Fit OLS for exposure d2
        mask_d2 = (D == d2)
        if np.sum(mask_d2) > X_with_intercept.shape[1]:
            ols_d2 = LinearRegression(fit_intercept=False)
            ols_d2.fit(X_with_intercept[mask_d2], Y[mask_d2])
            beta2 = ols_d2.coef_
            Ypred_d2 = X_with_intercept @ beta2
        else:
            # Not enough data, use mean
            beta2 = np.zeros(X_with_intercept.shape[1])
            beta2[0] = np.mean(Y[mask_d2]) if np.sum(mask_d2) > 0 else 0
            Ypred_d2 = X_with_intercept @ beta2
        
        # Use predictions directly (no calibration, like pred methods)
        # Set regression coefficients to 1 (trust the predictions)
        X1_use = Ypred_d1.reshape(-1, 1)
        X2_use = Ypred_d2.reshape(-1, 1)
        beta_aipw_1 = np.array([1.0])
        beta_aipw_2 = np.array([1.0])
        
        # Compute AIPW estimate
        mu1 = _mu_from_beta(Y, D, d1, pi1, X1_use, beta_aipw_1, eps)
        mu2 = _mu_from_beta(Y, D, d2, pi2, X2_use, beta_aipw_2, eps)
        tau_hat = float(mu1 - mu2)
        
        # For variance: use the original covariates X_with_intercept and fitted OLS coefficients
        inputs = prepare_grr_inputs(Y, D, X_run, d1, d2, pi_id_all, 
                                    X1=X_with_intercept, X2=X_with_intercept, eps=eps)
        var_info = {
            'hatY_d1': inputs['hatY_d1'], 'hatY_d2': inputs['hatY_d2'],
            'hat1_d1': inputs['hat1_d1'], 'hat1_d2': inputs['hat1_d2'],
            'X1': X_with_intercept, 'X2': X_with_intercept,
            'beta1': beta1, 'beta2': beta2
        }
        return tau_hat, True, var_info
    
    elif method in ['linear', 'lasso']:
        res = grr_estimator(Y, D, X_run, d1, d2, pi_id_all, Kn, method=method, eps=eps, **grr_kwargs)
        inputs = prepare_grr_inputs(Y, D, X_run, d1, d2, pi_id_all, eps=eps)
        beta1, beta2 = res['model'].get_coefficients()
        var_info = {
            'hatY_d1': inputs['hatY_d1'], 'hatY_d2': inputs['hatY_d2'],
            'hat1_d1': inputs['hat1_d1'], 'hat1_d2': inputs['hat1_d2'],
            'X1': inputs['X1'], 'X2': inputs['X2'],
            'beta1': beta1, 'beta2': beta2
        }
        return res['tau_hat'], True, var_info
    
    elif method == 'gat-linear':
        if gat_embeddings is None:
            raise ValueError("GAT not available")
        X1_use = gat_embeddings['X_GAT_leaf'][0]
        X2_use = gat_embeddings['X_GAT_leaf'][1]
        res = grr_estimator(Y, D, X_run, d1, d2, pi_id_all, Kn, method='linear', eps=eps,
                           X1=X1_use, X2=X2_use, **{k: v for k, v in grr_kwargs.items() if k in ['alpha']})
        inputs = prepare_grr_inputs(Y, D, X_run, d1, d2, pi_id_all, X1=X1_use, X2=X2_use, eps=eps)
        beta1, beta2 = res['model'].get_coefficients()
        var_info = {
            'hatY_d1': inputs['hatY_d1'], 'hatY_d2': inputs['hatY_d2'],
            'hat1_d1': inputs['hat1_d1'], 'hat1_d2': inputs['hat1_d2'],
            'X1': X1_use, 'X2': X2_use, 'beta1': beta1, 'beta2': beta2
        }
        return res['tau_hat'], True, var_info
    
    elif method == 'gat-no-harm':
        if gat_model is None:
            raise ValueError("GAT model not available")
        Ypred_d1 = gat_model.predict(X_run, D=d1)
        Ypred_d2 = gat_model.predict(X_run, D=d2)
        X1_use = Ypred_d1.reshape(-1, 1)
        X2_use = Ypred_d2.reshape(-1, 1)
        res = grr_estimator(Y, D, X_run, d1, d2, pi_id_all, Kn, method='linear', eps=eps,
                           X1=X1_use, X2=X2_use, **{k: v for k, v in grr_kwargs.items() if k in ['alpha']})
        inputs = prepare_grr_inputs(Y, D, X_run, d1, d2, pi_id_all, X1=X1_use, X2=X2_use, eps=eps)
        beta1, beta2 = res['model'].get_coefficients()
        var_info = {
            'hatY_d1': inputs['hatY_d1'], 'hatY_d2': inputs['hatY_d2'],
            'hat1_d1': inputs['hat1_d1'], 'hat1_d2': inputs['hat1_d2'],
            'X1': X1_use, 'X2': X2_use, 'beta1': beta1, 'beta2': beta2
        }
        return res['tau_hat'], True, var_info
    
    elif method == 'gat-pred':
        if gat_model is None:
            raise ValueError("GAT model not available")
        Ypred_d1 = gat_model.predict(X_run, D=d1)
        Ypred_d2 = gat_model.predict(X_run, D=d2)
        X1_use = Ypred_d1.reshape(-1, 1)
        X2_use = Ypred_d2.reshape(-1, 1)
        beta1 = np.array([1.0])
        beta2 = np.array([1.0])
        mu1 = _mu_from_beta(Y, D, d1, pi1, X1_use, beta1, eps)
        mu2 = _mu_from_beta(Y, D, d2, pi2, X2_use, beta2, eps)
        inputs = prepare_grr_inputs(Y, D, X_run, d1, d2, pi_id_all, X1=X1_use, X2=X2_use, eps=eps)
        var_info = {
            'hatY_d1': inputs['hatY_d1'], 'hatY_d2': inputs['hatY_d2'],
            'hat1_d1': inputs['hat1_d1'], 'hat1_d2': inputs['hat1_d2'],
            'X1': X1_use, 'X2': X2_use, 'beta1': beta1, 'beta2': beta2
        }
        return float(mu1 - mu2), True, var_info
    
    elif method == 'pna-linear':
        if pna_embeddings is None:
            raise ValueError("PNA not available")
        X1_use = pna_embeddings['X_PNA_leaf'][0]
        X2_use = pna_embeddings['X_PNA_leaf'][1]
        res = grr_estimator(Y, D, X_run, d1, d2, pi_id_all, Kn, method='linear', eps=eps,
                           X1=X1_use, X2=X2_use, **{k: v for k, v in grr_kwargs.items() if k in ['alpha']})
        inputs = prepare_grr_inputs(Y, D, X_run, d1, d2, pi_id_all, X1=X1_use, X2=X2_use, eps=eps)
        beta1, beta2 = res['model'].get_coefficients()
        var_info = {
            'hatY_d1': inputs['hatY_d1'], 'hatY_d2': inputs['hatY_d2'],
            'hat1_d1': inputs['hat1_d1'], 'hat1_d2': inputs['hat1_d2'],
            'X1': X1_use, 'X2': X2_use, 'beta1': beta1, 'beta2': beta2
        }
        return res['tau_hat'], True, var_info
    
    elif method == 'pna-no-harm':
        if pna_model is None:
            raise ValueError("PNA model not available")
        Ypred_d1 = pna_model.predict(X_run, D=d1)
        Ypred_d2 = pna_model.predict(X_run, D=d2)
        X1_use = Ypred_d1.reshape(-1, 1)
        X2_use = Ypred_d2.reshape(-1, 1)
        res = grr_estimator(Y, D, X_run, d1, d2, pi_id_all, Kn, method='linear', eps=eps,
                           X1=X1_use, X2=X2_use, **{k: v for k, v in grr_kwargs.items() if k in ['alpha']})
        inputs = prepare_grr_inputs(Y, D, X_run, d1, d2, pi_id_all, X1=X1_use, X2=X2_use, eps=eps)
        beta1, beta2 = res['model'].get_coefficients()
        var_info = {
            'hatY_d1': inputs['hatY_d1'], 'hatY_d2': inputs['hatY_d2'],
            'hat1_d1': inputs['hat1_d1'], 'hat1_d2': inputs['hat1_d2'],
            'X1': X1_use, 'X2': X2_use, 'beta1': beta1, 'beta2': beta2
        }
        return res['tau_hat'], True, var_info
    
    elif method == 'pna-pred':
        if pna_model is None:
            raise ValueError("PNA model not available")
        Ypred_d1 = pna_model.predict(X_run, D=d1)
        Ypred_d2 = pna_model.predict(X_run, D=d2)
        X1_use = Ypred_d1.reshape(-1, 1)
        X2_use = Ypred_d2.reshape(-1, 1)
        beta1 = np.array([1.0])
        beta2 = np.array([1.0])
        mu1 = _mu_from_beta(Y, D, d1, pi1, X1_use, beta1, eps)
        mu2 = _mu_from_beta(Y, D, d2, pi2, X2_use, beta2, eps)
        inputs = prepare_grr_inputs(Y, D, X_run, d1, d2, pi_id_all, X1=X1_use, X2=X2_use, eps=eps)
        var_info = {
            'hatY_d1': inputs['hatY_d1'], 'hatY_d2': inputs['hatY_d2'],
            'hat1_d1': inputs['hat1_d1'], 'hat1_d2': inputs['hat1_d2'],
            'X1': X1_use, 'X2': X2_use, 'beta1': beta1, 'beta2': beta2
        }
        return float(mu1 - mu2), True, var_info
    
    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    from tqdm import tqdm
    import pandas as pd
    import json
    
    print(f"Starting simulation for LINEAR-NO-CAL method only with {N_JOBS} workers, B={B} replications")
    print(f"Methods: {ALL_METHODS}")
    start_time = datetime.now()
    
    # =========================================================================
    # Step 1: Generate fixed data (graph, covariates, potential outcomes)
    # =========================================================================
    print("Generating graph and covariates...")
    G, H = generate_graph(n=N, avg_degree=AVG_DEGREE, max_degree=MAX_DEGREE, seed=SEED_GRAPH)
    X, X_noi = generate_covariates(H, seed=SEED_COV)
    X_run = X[:, [0, 3]]
    
    print("Generating potential outcomes with neighbor spillover...")
    Y_pot = generate_potential_outcomes(G, H, X_noi, k=K, beta_neighbor=1.0, seed=SEED_OUT)
    true_tau = float(np.mean(Y_pot[:, D1] - Y_pot[:, D2]))
    print(f"True tau = {true_tau:.4f}")
    
    A_adj = G.astype(float)
    
    # =========================================================================
    # Step 2: Compute propensity scores (one-time)
    # =========================================================================
    print("Computing propensity scores via Monte Carlo...")
    pi_id_all, pi_ij_d1d2, pi_ij_d1d1, pi_ij_d2d2 = monte_carlo_probabilities(
        G, TREATED_FRACTION, K, MC_REPS, SEED_MC, D1, D2
    )
    
    # Build matrices
    Kn = (G @ G.T > 0).astype(int)
    L1, L2, L12 = build_lambdas(pi_id_all, pi_ij_d1d2, pi_ij_d1d1, pi_ij_d2d2, D1, D2, EPS)
    W1_pi, W2_pi = build_W_pi(pi_id_all, pi_ij_d1d1, pi_ij_d2d2, D1, D2, Kn)
    
    # =========================================================================
    # Step 3: Run parallel MC replications
    # =========================================================================
    print(f"Running {B} MC replications in parallel...")
    
    grr_kwargs = {'alpha': 0.01, 'l1_ratio': 0.5}
    
    # Use joblib for parallel execution
    # Each worker gets assigned to GPU 0 (single GPU), but they run sequentially on GPU
    results_list = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(run_single_replication)(
            b, G, H, X, X_run, X_noi, Y_pot, A_adj,
            pi_id_all, Kn, L1, L2, L12, W1_pi, W2_pi,
            D1, D2, K, N, TREATED_FRACTION, true_tau,
            ALL_METHODS, GAT_CONFIG, PNA_CONFIG, grr_kwargs, EPS,
            gpu_id=0  # All workers share GPU 0
        )
        for b in range(B)
    )
    
    # =========================================================================
    # Step 4: Aggregate results
    # =========================================================================
    print("Aggregating results...")
    
    results = {m: np.zeros(B) for m in ALL_METHODS}
    coverage = {m: np.zeros(B) for m in ALL_METHODS}
    power = {m: np.zeros(B) for m in ALL_METHODS}
    
    for b, results_b, coverage_b, power_b in results_list:
        for m in ALL_METHODS:
            results[m][b] = results_b.get(m, np.nan)
            coverage[m][b] = coverage_b.get(m, np.nan)
            power[m][b] = power_b.get(m, np.nan)
    
    # Compute summaries
    summary = {}
    for m in ALL_METHODS:
        v = results[m]
        v_valid = v[~np.isnan(v)]
        cov_valid = coverage[m][~np.isnan(coverage[m])]
        pow_valid = power[m][~np.isnan(power[m])]
        
        if len(v_valid) > 0:
            bias = float(np.mean(v_valid) - true_tau)
            sd = float(np.std(v_valid, ddof=1))
            rmse = float(np.sqrt(bias**2 + sd**2))
            summary[m] = {
                'bias': bias, 'sd': sd, 'rmse': rmse,
                'n_valid': len(v_valid),
                'coverage': float(np.mean(cov_valid)) if len(cov_valid) > 0 else np.nan,
                'power': float(np.mean(pow_valid)) if len(pow_valid) > 0 else np.nan,
            }
        else:
            summary[m] = {'bias': np.nan, 'sd': np.nan, 'rmse': np.nan, 'n_valid': 0,
                         'coverage': np.nan, 'power': np.nan}
    
    # =========================================================================
    # Step 5: Print and save results
    # =========================================================================
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\nCompleted in {elapsed/60:.1f} minutes ({elapsed/B:.1f}s per rep)")
    
    print("\n" + "="*100)
    print(f"Simulation Results (true_tau = {true_tau:.4f}, B = {B})")
    print("="*100)
    print(f"{'Method':<18} {'Bias':>10} {'SD':>10} {'RMSE':>10} {'Coverage':>10} {'Power':>10} {'N Valid':>8}")
    print("-"*100)
    
    for m in ALL_METHODS:
        s = summary[m]
        if s['n_valid'] > 0:
            cov_str = f"{s['coverage']:.3f}" if not np.isnan(s['coverage']) else "-"
            pow_str = f"{s['power']:.3f}" if not np.isnan(s['power']) else "-"
            print(f"{m:<18} {s['bias']:>10.4f} {s['sd']:>10.4f} {s['rmse']:>10.4f} {cov_str:>10} {pow_str:>10} {s['n_valid']:>8}")
        else:
            print(f"{m:<18} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'-':>10} {'-':>10} {0:>8}")
    print("="*100)
    
    # Save results
    results_dir = 'dgp-2/results'
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save summary CSV
    summary_data = [{'method': m, **summary[m]} for m in ALL_METHODS]
    pd.DataFrame(summary_data).to_csv(f"{results_dir}/out1_dgp2_linear_no_cal_{timestamp}_summary.csv", index=False)
    
    # Save config JSON
    config = {
        'timestamp': timestamp,
        'true_tau': true_tau,
        'n': N,
        'B': B,
        'n_jobs': N_JOBS,
        'elapsed_seconds': elapsed,
        'methods': ALL_METHODS,
        'summary': summary
    }
    with open(f"{results_dir}/out1_dgp2_linear_no_cal_{timestamp}_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save full results NPZ
    np.savez(f"{results_dir}/out1_dgp2_linear_no_cal_{timestamp}_full.npz",
             **{f"results_{m}": results[m] for m in ALL_METHODS},
             **{f"coverage_{m}": coverage[m] for m in ALL_METHODS},
             **{f"power_{m}": power[m] for m in ALL_METHODS})
    
    print(f"\nResults saved to {results_dir}/out1_dgp2_linear_no_cal_{timestamp}_*")
