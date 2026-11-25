"""
Simulation Toolkit for Network Causal Inference

This module provides the core functions for running Monte Carlo simulations
comparing different treatment effect estimators in network settings.

Main Functions:
    - run_grr_simulation: Run full simulation pipeline with multiple estimators
    - plot_grr_results: Visualize simulation results
    - save_results: Save results to files
    - print_summary_table: Print formatted summary

Supported Estimators:
    - Primitive: HT, Hajek, HAIPW variants
    - Standard GRR: Linear, Ridge, Lasso, ElasticNet, RF, GBM, MLP
    - End-to-end PNA-GRR: 'pna' (trains PNA directly on GRR loss)
    - GAT-enhanced GRR: gat-linear, gat-lasso, gat-mlp, gat-no-harm, gat-pred
    - PNA-enhanced GRR: pna-linear, pna-lasso, pna-mlp, pna-no-harm, pna-pred
    
    End-to-end PNA-GRR ('pna'):
        Trains a PNA network directly to minimize the GRR loss res' @ Kn @ res.
        Uses graph structure to learn f1 and f2 adjustment functions.
    
    No-harm calibration methods (gat-no-harm, pna-no-harm):
        Uses predicted potential outcomes Ypred(d1) and Ypred(d2) from the GNN
        as covariates for GRR linear estimation.
    
    Doubly robust prediction methods (gat-pred, pna-pred):
        Uses GNN predictions directly in a doubly robust estimator via _mu_from_beta
        with the prediction as covariate and beta=1.
"""

# =============================================================================
# Imports
# =============================================================================

import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

# Import from local modules
from dgp import (
    generate_graph, 
    generate_covariates, 
    generate_potential_outcomes,
    assign_treatment, 
    compute_exposure, 
    observed_outcome
)
from propensity import monte_carlo_probabilities, build_lambdas, build_W_pi, build_kn_matrix
from GRR import grr_estimator, GAT_embedding, PNA_embedding, prepare_grr_inputs
from variance_estimators import grr_variance_estimator, grr_confidence_interval
from estimators import (
    mu_ht, mu_haj, 
    _weights, _Omega_const, _vec_const, _solve, _mu_from_c, _stack_for_reg,
    _Omega_cov, _vec_cov, _mu_from_beta
)


# =============================================================================
# Constants
# =============================================================================

# Methods that don't minimize a loss (primitive estimators)
PRIMITIVE_METHODS = {
    'ht', 'hajek', 
    'haipw-truth', 'haipw-one', 
    'haipw-cov-truth', 'haipw-cov-one',
    'gat-pred', 'pna-pred'  # DR estimators using GNN predictions
}

# Graph neural network enhanced methods
GAT_METHODS = {'gat-linear', 'gat-lasso', 'gat-mlp', 'gat-no-harm', 'gat-pred'}
PNA_METHODS = {'pna-linear', 'pna-lasso', 'pna-mlp', 'pna-no-harm', 'pna-pred'}
GNN_METHODS = GAT_METHODS | PNA_METHODS

# End-to-end PNA-GRR (trains PNA directly on GRR loss)
PNA_GRR_METHOD = {'pna'}


# =============================================================================
# Main Simulation Function
# =============================================================================

def run_grr_simulation(
    # --- data gen / design ---
    n=200, avg_degree=3.0, max_degree=9, k=3,
    treated_fraction=1/3,
    # exposures to compare
    d1=0, d2=2,
    # MC for pi's
    mc_reps=1000, seed_mc=111,
    # seeds
    seed_graph=123, seed_cov=1234, seed_out=999,
    # experiment reps
    B=200,
    # GRR methods to compare
    methods=['linear', 'ridge', 'lasso'],
    # options
    eps=1e-12,
    verbose=False,
    # GNN embedding options
    gat_kwargs=None,
    pna_kwargs=None,
    **grr_kwargs
):
    """
    Full pipeline: generate (G,X,Y_pot), compute probabilities, then run
    GRR estimators for B randomizations.

    Supported methods:
      - Primitive estimators: 'ht', 'hajek', 'haipw-truth', 'haipw-one',
                              'haipw-cov-truth', 'haipw-cov-one'
      - Standard GRR: 'linear', 'ridge', 'lasso', 'elasticnet', 'rf', 'gbm', 'mlp'
      - End-to-end PNA-GRR: 'pna'
        * Trains PNA network directly to minimize GRR loss res' @ Kn @ res
        * Uses graph structure to learn adjustment functions f1, f2
      - GAT-enhanced GRR: 'gat-linear', 'gat-lasso', 'gat-mlp', 'gat-no-harm', 'gat-pred'
        * Uses Graph Attention Network embeddings (note: softmax normalizes degree)
        * gat-no-harm: uses predicted potential outcomes as covariates for GRR
        * gat-pred: doubly robust estimator using predictions directly
      - PNA-enhanced GRR: 'pna-linear', 'pna-lasso', 'pna-mlp', 'pna-no-harm', 'pna-pred'
        * Uses Principal Neighbourhood Aggregation (preserves degree information)
        * pna-no-harm: uses predicted potential outcomes as covariates for GRR
        * pna-pred: doubly robust estimator using predictions directly

    Args:
        n: number of nodes
        avg_degree: target average degree
        max_degree: maximum degree per node
        k: number of exposure levels
        treated_fraction: fraction of units to treat
        d1, d2: exposure levels to compare
        mc_reps: Monte Carlo reps for propensity estimation
        seed_mc: seed for MC propensity estimation
        seed_graph, seed_cov, seed_out: seeds for data generation
        B: number of experiment randomizations
        methods: list of methods to compare
        eps: numerical stability constant
        verbose: print progress
        gat_kwargs: kwargs for GAT_embedding (if using gat-* methods)
        pna_kwargs: kwargs for PNA_embedding (if using pna-* methods)
        **grr_kwargs: additional kwargs for grr_estimator

    Returns:
        dict with keys:
            'G', 'H', 'X', 'Y_pot', 'pi_id_all', 'Kn', 'true_tau',
            'results': {method -> np.ndarray of tau_hat values},
            'losses': {method -> np.ndarray of loss values},
            'summary': {method -> {'bias', 'sd', 'rmse', 'n_valid'}},
            'r2_linear', 'r2_gat', 'r2_pna', 'r2_summary'
    """
    # Default kwargs
    if gat_kwargs is None:
        gat_kwargs = {}
    if pna_kwargs is None:
        pna_kwargs = {}
    
    # Check which method types are requested
    methods_set = set(methods)
    use_gat = bool(GAT_METHODS & methods_set)
    use_pna = bool(PNA_METHODS & methods_set)
    use_primitive = bool(PRIMITIVE_METHODS & methods_set)
    
    # =========================================================================
    # Step 1-3: Generate graph, covariates, and potential outcomes
    # =========================================================================
    G, H = generate_graph(n=n, avg_degree=avg_degree, max_degree=max_degree, seed=seed_graph)
    X, X_noi = generate_covariates(H, seed=seed_cov)
    # X_run = X[:, [0, 3]]  # [intercept, X3] - note: limited degree info
    # X_run = X[:, [0, 1, 3]]
    X_run = X[:, [0, 3]]
    Y_pot = generate_potential_outcomes(H, X_noi, k=k, seed=seed_out)
    true_tau = float(np.mean(Y_pot[:, d1] - Y_pot[:, d2]))

    # =========================================================================
    # Step 4: MC estimation of propensity scores and joint probabilities
    # =========================================================================
    pi_id_all, pi_ij_d1d2, pi_ij_d1d1, pi_ij_d2d2 = monte_carlo_probabilities(
        G, treated_fraction, k, mc_reps, seed_mc, d1, d2
    )

    # =========================================================================
    # Step 5: Build Kn matrix (neighborhood overlap indicator)
    # =========================================================================
    Kn = (G @ G.T > 0).astype(int)
    
    # =========================================================================
    # Step 6: Build Lambda matrices (for HAIPW estimators)
    # =========================================================================
    L1, L2, L12 = None, None, None
    if use_primitive:
        L1, L2, L12 = build_lambdas(pi_id_all, pi_ij_d1d2, pi_ij_d1d1, pi_ij_d2d2, d1, d2, eps)
    
    # Adjacency matrix for GNN
    A_adj = G.astype(float)

    # =========================================================================
    # Step 7: Build W matrices for variance estimation
    # =========================================================================
    W1_pi, W2_pi = build_W_pi(pi_id_all, pi_ij_d1d1, pi_ij_d2d2, d1, d2, Kn)
    
    # =========================================================================
    # Storage for results
    # =========================================================================
    results = {m: np.zeros(B, dtype=float) for m in methods}
    losses = {m: np.zeros(B, dtype=float) for m in methods}
    
    # Coverage and power tracking (for methods with variance estimation)
    # coverage[m][b] = 1 if CI covers true_tau, 0 otherwise
    # power[m][b] = 1 if CI does NOT cover 0 (reject null), 0 otherwise
    coverage = {m: np.zeros(B, dtype=float) for m in methods}
    power = {m: np.zeros(B, dtype=float) for m in methods}
    ci_lower = {m: np.zeros(B, dtype=float) for m in methods}
    ci_upper = {m: np.zeros(B, dtype=float) for m in methods}
    
    # R² tracking
    r2_linear = np.zeros(B, dtype=float)
    r2_gat = np.zeros(B, dtype=float) if use_gat else None
    r2_pna = np.zeros(B, dtype=float) if use_pna else None

    # =========================================================================
    # Monte Carlo loop
    # =========================================================================
    for b in tqdm(range(B), desc="Monte Carlo simulations", disable=not True):
        A_treat = assign_treatment(n, treated_fraction=treated_fraction, seed=None)
        D = compute_exposure(G, A_treat, k)
        Y = observed_outcome(Y_pot, D, k=k)
        
        # ---------------------------------------------------------------------
        # Get GNN embeddings (once per randomization)
        # ---------------------------------------------------------------------
        gat_embeddings, gat_model = None, None
        pna_embeddings, pna_model = None, None
        
        # GAT embeddings
        if use_gat:
            try:
                gat_embeddings = GAT_embedding(
                    X_run, A_adj, D, Y, k=k, d1=d1, d2=d2,
                    verbose=(b == 0 and verbose),
                    return_model=True,
                    **gat_kwargs
                )
                gat_model = gat_embeddings.get('model', None)
                
                # Compute R² for GAT
                if gat_model is not None:
                    Y_pred = np.zeros(n)
                    for dd in range(k):
                        mask_d = (D == dd)
                        if mask_d.sum() > 0:
                            Y_pred[mask_d] = gat_model.predict(X_run, D=dd)[mask_d]
                    ss_res = np.sum((Y - Y_pred) ** 2)
                    ss_tot = np.sum((Y - Y.mean()) ** 2)
                    r2_gat[b] = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                else:
                    r2_gat[b] = np.nan
            except Exception as e:
                if verbose:
                    print(f"Rep {b}, GAT embedding failed: {e}")
                for m in GAT_METHODS & methods_set:
                    results[m][b] = np.nan
                    losses[m][b] = np.nan
                if r2_gat is not None:
                    r2_gat[b] = np.nan
        
        # PNA embeddings
        if use_pna:
            try:
                pna_embeddings = PNA_embedding(
                    X_run, A_adj, D, Y, k=k, d1=d1, d2=d2,
                    verbose=(b == 0 and verbose),
                    return_model=True,
                    **pna_kwargs
                )
                pna_model = pna_embeddings.get('model', None)
                
                # Compute R² for PNA
                if pna_model is not None:
                    Y_pred = np.zeros(n)
                    for dd in range(k):
                        mask_d = (D == dd)
                        if mask_d.sum() > 0:
                            Y_pred[mask_d] = pna_model.predict(X_run, D=dd)[mask_d]
                    ss_res = np.sum((Y - Y_pred) ** 2)
                    ss_tot = np.sum((Y - Y.mean()) ** 2)
                    r2_pna[b] = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                else:
                    r2_pna[b] = np.nan
            except Exception as e:
                if verbose:
                    print(f"Rep {b}, PNA embedding failed: {e}")
                for m in PNA_METHODS & methods_set:
                    results[m][b] = np.nan
                    losses[m][b] = np.nan
                if r2_pna is not None:
                    r2_pna[b] = np.nan
        
        # ---------------------------------------------------------------------
        # Compute R² for linear model (baseline)
        # ---------------------------------------------------------------------
        Y_pred_linear = np.zeros(n)
        for dd in range(k):
            mask_d = (D == dd)
            if mask_d.sum() > 5:
                lr = LinearRegression()
                lr.fit(X_run[mask_d], Y[mask_d])
                Y_pred_linear[mask_d] = lr.predict(X_run[mask_d])
            else:
                Y_pred_linear[mask_d] = Y[mask_d].mean() if mask_d.sum() > 0 else 0
        ss_res = np.sum((Y - Y_pred_linear) ** 2)
        ss_tot = np.sum((Y - Y.mean()) ** 2)
        r2_linear[b] = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # ---------------------------------------------------------------------
        # Run each estimator
        # ---------------------------------------------------------------------
        for method in methods:
            try:
                result = _run_single_method(
                    method, Y, D, X_run, d1, d2, pi_id_all, Kn, eps,
                    L1, L2, L12,
                    gat_embeddings, pna_embeddings,
                    gat_model, pna_model,
                    grr_kwargs,
                    A_adj=A_adj
                )
                tau_hat = result['tau_hat']
                loss = result['loss']
                results[method][b] = tau_hat
                losses[method][b] = loss
                
                # Compute confidence interval if variance info is available
                if result.get('has_variance_info', False):
                    try:
                        ci_lo, ci_hi = grr_confidence_interval(
                            tau_hat=tau_hat,
                            hatY_d1=result['hatY_d1'],
                            hatY_d2=result['hatY_d2'],
                            hat1_d1=result['hat1_d1'],
                            hat1_d2=result['hat1_d2'],
                            X1=result['X1'],
                            X2=result['X2'],
                            beta1=result['beta1'],
                            beta2=result['beta2'],
                            W1_pi=W1_pi,
                            W2_pi=W2_pi,
                            alpha=0.05
                        )
                        ci_lower[method][b] = ci_lo
                        ci_upper[method][b] = ci_hi
                        
                        # Coverage: does CI contain true_tau?
                        coverage[method][b] = 1.0 if (ci_lo <= true_tau <= ci_hi) else 0.0
                        
                        # Power: does CI NOT contain 0? (reject null hypothesis)
                        power[method][b] = 1.0 if (ci_lo > 0 or ci_hi < 0) else 0.0
                    except Exception as e_ci:
                        if verbose:
                            print(f"Rep {b}, method {method}, CI computation failed: {e_ci}")
                        ci_lower[method][b] = np.nan
                        ci_upper[method][b] = np.nan
                        coverage[method][b] = np.nan
                        power[method][b] = np.nan
                else:
                    # No variance info for primitive methods
                    ci_lower[method][b] = np.nan
                    ci_upper[method][b] = np.nan
                    coverage[method][b] = np.nan
                    power[method][b] = np.nan
                    
            except Exception as e:
                if verbose:
                    print(f"Rep {b}, method {method}: {e}")
                results[method][b] = np.nan
                losses[method][b] = np.nan
                ci_lower[method][b] = np.nan
                ci_upper[method][b] = np.nan
                coverage[method][b] = np.nan
                power[method][b] = np.nan

    # =========================================================================
    # Compute summaries
    # =========================================================================
    summary = {}
    for m in methods:
        v = results[m]
        v_valid = v[~np.isnan(v)]
        if len(v_valid) > 0:
            bias = float(np.mean(v_valid) - true_tau)
            sd = float(np.std(v_valid, ddof=1))
            rmse = float(np.sqrt(bias**2 + sd**2))
            
            # Coverage and power (for methods with variance estimation)
            cov_valid = coverage[m][~np.isnan(coverage[m])]
            pow_valid = power[m][~np.isnan(power[m])]
            
            coverage_rate = float(np.mean(cov_valid)) if len(cov_valid) > 0 else np.nan
            power_rate = float(np.mean(pow_valid)) if len(pow_valid) > 0 else np.nan
            
            summary[m] = {
                "bias": bias, 
                "sd": sd, 
                "rmse": rmse, 
                "n_valid": len(v_valid),
                "coverage": coverage_rate,
                "power": power_rate,
                "n_ci_valid": len(cov_valid)
            }
        else:
            summary[m] = {
                "bias": np.nan, 
                "sd": np.nan, 
                "rmse": np.nan, 
                "n_valid": 0,
                "coverage": np.nan,
                "power": np.nan,
                "n_ci_valid": 0
            }

    # R² summaries
    r2_summary = {}
    r2_linear_valid = r2_linear[~np.isnan(r2_linear)]
    r2_summary['linear'] = {
        'mean': float(np.mean(r2_linear_valid)) if len(r2_linear_valid) > 0 else np.nan,
        'std': float(np.std(r2_linear_valid)) if len(r2_linear_valid) > 0 else np.nan,
    }
    if r2_gat is not None:
        r2_gat_valid = r2_gat[~np.isnan(r2_gat)]
        r2_summary['gat'] = {
            'mean': float(np.mean(r2_gat_valid)) if len(r2_gat_valid) > 0 else np.nan,
            'std': float(np.std(r2_gat_valid)) if len(r2_gat_valid) > 0 else np.nan,
        }
    if r2_pna is not None:
        r2_pna_valid = r2_pna[~np.isnan(r2_pna)]
        r2_summary['pna'] = {
            'mean': float(np.mean(r2_pna_valid)) if len(r2_pna_valid) > 0 else np.nan,
            'std': float(np.std(r2_pna_valid)) if len(r2_pna_valid) > 0 else np.nan,
        }

    return dict(
        G=G, H=H, X=X, Y_pot=Y_pot,
        pi_id_all=pi_id_all, Kn=Kn,
        W1_pi=W1_pi, W2_pi=W2_pi,
        true_tau=true_tau,
        results=results,
        losses=losses,
        coverage=coverage,
        power=power,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        summary=summary,
        methods=methods,
        r2_linear=r2_linear,
        r2_gat=r2_gat,
        r2_pna=r2_pna,
        r2_summary=r2_summary,
    )


# =============================================================================
# Helper: Run Single Method
# =============================================================================

def _run_single_method(method, Y, D, X_run, d1, d2, pi_id_all, Kn, eps,
                       L1, L2, L12, gat_embeddings, pna_embeddings,
                       gat_model, pna_model, grr_kwargs, A_adj=None):
    """
    Run a single estimation method and return result dict.
    
    Returns dict with keys:
        tau_hat: estimated treatment effect
        loss: GRR loss value (np.nan for primitive methods)
        has_variance_info: whether variance estimation info is available
        hatY_d1, hatY_d2, hat1_d1, hat1_d2, X1, X2, beta1, beta2: 
            (only if has_variance_info=True)
    
    Internal helper for run_grr_simulation.
    """
    pi1 = pi_id_all[:, d1]
    pi2 = pi_id_all[:, d2]
    n = len(Y)
    
    # Helper to build result dict without variance info
    def _result_no_var(tau_hat, loss=np.nan):
        return {'tau_hat': tau_hat, 'loss': loss, 'has_variance_info': False}
    
    # Helper to build result dict with variance info
    def _result_with_var(tau_hat, loss, hatY_d1, hatY_d2, hat1_d1, hat1_d2, 
                         X1, X2, beta1, beta2):
        return {
            'tau_hat': tau_hat, 
            'loss': loss, 
            'has_variance_info': True,
            'hatY_d1': hatY_d1,
            'hatY_d2': hatY_d2,
            'hat1_d1': hat1_d1,
            'hat1_d2': hat1_d2,
            'X1': X1,
            'X2': X2,
            'beta1': beta1,
            'beta2': beta2
        }
    
    # =========================================================================
    # HT and Hajek estimators (with variance estimation)
    # =========================================================================
    if method == 'ht':
        mu1 = mu_ht(Y, D, d1, pi_id_all, eps)
        mu2 = mu_ht(Y, D, d2, pi_id_all, eps)
        tau_hat = float(mu1 - mu2)
        # Variance estimation: use X = ones, beta = mu_ht estimates
        X_ones = np.ones((n, 1))
        beta1 = np.array([mu1])
        beta2 = np.array([mu2])
        inputs = prepare_grr_inputs(Y, D, X_run, d1, d2, pi_id_all, X1=X_ones, X2=X_ones, eps=eps)
        return _result_with_var(
            tau_hat, np.nan,
            inputs['hatY_d1'], inputs['hatY_d2'],
            inputs['hat1_d1'], inputs['hat1_d2'],
            X_ones, X_ones, beta1, beta2
        )
        
    elif method == 'hajek':
        mu1 = mu_haj(Y, D, d1, pi_id_all, eps)
        mu2 = mu_haj(Y, D, d2, pi_id_all, eps)
        tau_hat = float(mu1 - mu2)
        # Variance estimation: use X = ones, beta = mu_haj estimates
        X_ones = np.ones((n, 1))
        beta1 = np.array([mu1])
        beta2 = np.array([mu2])
        inputs = prepare_grr_inputs(Y, D, X_run, d1, d2, pi_id_all, X1=X_ones, X2=X_ones, eps=eps)
        return _result_with_var(
            tau_hat, np.nan,
            inputs['hatY_d1'], inputs['hatY_d2'],
            inputs['hat1_d1'], inputs['hat1_d2'],
            X_ones, X_ones, beta1, beta2
        )
        
    elif method == 'haipw-truth':
        w1 = _weights(D, d1, pi1, eps)
        w2 = _weights(D, d2, pi2, eps)
        w1Y, w2Y = w1 * Y, w2 * Y
        hT1, _, dHY = _stack_for_reg(Y, D, X_run, d1, d2, pi1, pi2, eps)
        Om = _Omega_const('truth', L1, L2, L12, Kn, w1=w1, w2=w2, hT1=hT1)
        vec = _vec_const('one_sided', L1, L2, L12, Kn, w1Y=w1Y, w2Y=w2Y, hT1=hT1, dHY=dHY)
        c_hat = _solve(Om, vec)
        c1, c2 = float(c_hat[0]), float(c_hat[1])
        mu1 = _mu_from_c(Y, D, d1, pi1, c1, eps)
        mu2 = _mu_from_c(Y, D, d2, pi2, c2, eps)
        return _result_no_var(float(mu1 - mu2))
        
    elif method == 'haipw-one':
        w1 = _weights(D, d1, pi1, eps)
        w2 = _weights(D, d2, pi2, eps)
        w1Y, w2Y = w1 * Y, w2 * Y
        hT1, _, dHY = _stack_for_reg(Y, D, X_run, d1, d2, pi1, pi2, eps)
        Om = _Omega_const('one_sided', L1, L2, L12, Kn, w1=w1, w2=w2, hT1=hT1)
        vec = _vec_const('one_sided', L1, L2, L12, Kn, w1Y=w1Y, w2Y=w2Y, hT1=hT1, dHY=dHY)
        c_hat = _solve(Om, vec)
        c1, c2 = float(c_hat[0]), float(c_hat[1])
        mu1 = _mu_from_c(Y, D, d1, pi1, c1, eps)
        mu2 = _mu_from_c(Y, D, d2, pi2, c2, eps)
        return _result_no_var(float(mu1 - mu2))
        
    elif method == 'haipw-cov-truth':
        w1 = _weights(D, d1, pi1, eps)
        w2 = _weights(D, d2, pi2, eps)
        w1Y, w2Y = w1 * Y, w2 * Y
        hT1, hTX, dHY = _stack_for_reg(Y, D, X_run, d1, d2, pi1, pi2, eps)
        Xw1 = X_run * w1[:, None]
        Xw2 = X_run * w2[:, None]
        OmX = _Omega_cov('truth', L1, L2, L12, Kn, X=X_run, Xw1=Xw1, Xw2=Xw2, hTX=hTX)
        vecX = _vec_cov('one_sided', L1, L2, L12, Kn, X=X_run, w1Y=w1Y, w2Y=w2Y, hTX=hTX, dHY=dHY)
        beta_hat = _solve(OmX, vecX)
        p = X_run.shape[1]
        beta1, beta2 = beta_hat[:p], beta_hat[p:]
        mu1 = _mu_from_beta(Y, D, d1, pi1, X_run, beta1, eps)
        mu2 = _mu_from_beta(Y, D, d2, pi2, X_run, beta2, eps)
        return _result_no_var(float(mu1 - mu2))
        
    elif method == 'haipw-cov-one':
        w1 = _weights(D, d1, pi1, eps)
        w2 = _weights(D, d2, pi2, eps)
        w1Y, w2Y = w1 * Y, w2 * Y
        hT1, hTX, dHY = _stack_for_reg(Y, D, X_run, d1, d2, pi1, pi2, eps)
        Xw1 = X_run * w1[:, None]
        Xw2 = X_run * w2[:, None]
        OmX = _Omega_cov('one_sided', L1, L2, L12, Kn, X=X_run, Xw1=Xw1, Xw2=Xw2, hTX=hTX)
        vecX = _vec_cov('one_sided', L1, L2, L12, Kn, X=X_run, w1Y=w1Y, w2Y=w2Y, hTX=hTX, dHY=dHY)
        beta_hat = _solve(OmX, vecX)
        p = X_run.shape[1]
        beta1, beta2 = beta_hat[:p], beta_hat[p:]
        mu1 = _mu_from_beta(Y, D, d1, pi1, X_run, beta1, eps)
        mu2 = _mu_from_beta(Y, D, d2, pi2, X_run, beta2, eps)
        return _result_no_var(float(mu1 - mu2))
    
    # =========================================================================
    # GAT-enhanced GRR (with variance estimation)
    # =========================================================================
    elif method == 'gat-linear':
        if gat_embeddings is None:
            raise ValueError("GAT embeddings not available")
        X1_use = gat_embeddings['X_GAT_leaf'][0]
        X2_use = gat_embeddings['X_GAT_leaf'][1]
        res = grr_estimator(
            Y, D, X_run, d1, d2, pi_id_all, Kn,
            method='linear', eps=eps,
            X1=X1_use, X2=X2_use,
            **{k: v for k, v in grr_kwargs.items() if k in ['alpha']}
        )
        inputs = prepare_grr_inputs(Y, D, X_run, d1, d2, pi_id_all, X1=X1_use, X2=X2_use, eps=eps)
        beta1, beta2 = res['model'].get_coefficients()
        return _result_with_var(
            res['tau_hat'], res['loss'],
            inputs['hatY_d1'], inputs['hatY_d2'],
            inputs['hat1_d1'], inputs['hat1_d2'],
            X1_use, X2_use, beta1, beta2
        )
        
    elif method == 'gat-lasso':
        if gat_embeddings is None:
            raise ValueError("GAT embeddings not available")
        X1_use = gat_embeddings['X_GAT_leaf'][0]
        X2_use = gat_embeddings['X_GAT_leaf'][1]
        res = grr_estimator(
            Y, D, X_run, d1, d2, pi_id_all, Kn,
            method='lasso', eps=eps,
            X1=X1_use, X2=X2_use,
            **{k: v for k, v in grr_kwargs.items() if k in ['alpha']}
        )
        inputs = prepare_grr_inputs(Y, D, X_run, d1, d2, pi_id_all, X1=X1_use, X2=X2_use, eps=eps)
        beta1, beta2 = res['model'].get_coefficients()
        return _result_with_var(
            res['tau_hat'], res['loss'],
            inputs['hatY_d1'], inputs['hatY_d2'],
            inputs['hat1_d1'], inputs['hat1_d2'],
            X1_use, X2_use, beta1, beta2
        )
        
    elif method == 'gat-mlp':
        if gat_embeddings is None:
            raise ValueError("GAT embeddings not available")
        # MLP methods don't have simple beta coefficients, skip variance estimation
        res = grr_estimator(
            Y, D, X_run, d1, d2, pi_id_all, Kn,
            method='mlp', eps=eps,
            X1=gat_embeddings['X_GAT_root'],
            X2=gat_embeddings['X_GAT_root'],
            **{k: v for k, v in grr_kwargs.items() 
               if k in ['hidden_dims', 'n_epochs', 'lr', 'weight_decay', 
                       'dropout', 'early_stopping', 'patience']}
        )
        return _result_no_var(res['tau_hat'], res['loss'])
    
    elif method == 'gat-no-harm':
        # No-harm calibration: use predicted potential outcomes as covariates
        if gat_model is None:
            raise ValueError("GAT model not available")
        Ypred_d1 = gat_model.predict(X_run, D=d1)
        Ypred_d2 = gat_model.predict(X_run, D=d2)
        X1_noharm = Ypred_d1.reshape(-1, 1)
        X2_noharm = Ypred_d2.reshape(-1, 1)
        res = grr_estimator(
            Y, D, X_run, d1, d2, pi_id_all, Kn,
            method='linear', eps=eps,
            X1=X1_noharm, X2=X2_noharm,
            **{k: v for k, v in grr_kwargs.items() if k in ['alpha']}
        )
        inputs = prepare_grr_inputs(Y, D, X_run, d1, d2, pi_id_all, X1=X1_noharm, X2=X2_noharm, eps=eps)
        beta1, beta2 = res['model'].get_coefficients()
        return _result_with_var(
            res['tau_hat'], res['loss'],
            inputs['hatY_d1'], inputs['hatY_d2'],
            inputs['hat1_d1'], inputs['hat1_d2'],
            X1_noharm, X2_noharm, beta1, beta2
        )
    
    elif method == 'gat-pred':
        # Doubly robust estimator using GAT predictions directly
        if gat_model is None:
            raise ValueError("GAT model not available")
        Ypred_d1 = gat_model.predict(X_run, D=d1)
        Ypred_d2 = gat_model.predict(X_run, D=d2)
        X1_pred = Ypred_d1.reshape(-1, 1)
        X2_pred = Ypred_d2.reshape(-1, 1)
        beta1 = np.array([1.0])
        beta2 = np.array([1.0])
        mu1 = _mu_from_beta(Y, D, d1, pi1, X1_pred, beta1, eps)
        mu2 = _mu_from_beta(Y, D, d2, pi2, X2_pred, beta2, eps)
        # Can compute variance for this method
        inputs = prepare_grr_inputs(Y, D, X_run, d1, d2, pi_id_all, X1=X1_pred, X2=X2_pred, eps=eps)
        return _result_with_var(
            float(mu1 - mu2), np.nan,
            inputs['hatY_d1'], inputs['hatY_d2'],
            inputs['hat1_d1'], inputs['hat1_d2'],
            X1_pred, X2_pred, beta1, beta2
        )
    
    # =========================================================================
    # PNA-enhanced GRR (with variance estimation)
    # =========================================================================
    elif method == 'pna-linear':
        if pna_embeddings is None:
            raise ValueError("PNA embeddings not available")
        X1_use = pna_embeddings['X_PNA_leaf'][0]
        X2_use = pna_embeddings['X_PNA_leaf'][1]
        res = grr_estimator(
            Y, D, X_run, d1, d2, pi_id_all, Kn,
            method='linear', eps=eps,
            X1=X1_use, X2=X2_use,
            **{k: v for k, v in grr_kwargs.items() if k in ['alpha']}
        )
        inputs = prepare_grr_inputs(Y, D, X_run, d1, d2, pi_id_all, X1=X1_use, X2=X2_use, eps=eps)
        beta1, beta2 = res['model'].get_coefficients()
        return _result_with_var(
            res['tau_hat'], res['loss'],
            inputs['hatY_d1'], inputs['hatY_d2'],
            inputs['hat1_d1'], inputs['hat1_d2'],
            X1_use, X2_use, beta1, beta2
        )
        
    elif method == 'pna-lasso':
        if pna_embeddings is None:
            raise ValueError("PNA embeddings not available")
        X1_use = pna_embeddings['X_PNA_leaf'][0]
        X2_use = pna_embeddings['X_PNA_leaf'][1]
        res = grr_estimator(
            Y, D, X_run, d1, d2, pi_id_all, Kn,
            method='lasso', eps=eps,
            X1=X1_use, X2=X2_use,
            **{k: v for k, v in grr_kwargs.items() if k in ['alpha']}
        )
        inputs = prepare_grr_inputs(Y, D, X_run, d1, d2, pi_id_all, X1=X1_use, X2=X2_use, eps=eps)
        beta1, beta2 = res['model'].get_coefficients()
        return _result_with_var(
            res['tau_hat'], res['loss'],
            inputs['hatY_d1'], inputs['hatY_d2'],
            inputs['hat1_d1'], inputs['hat1_d2'],
            X1_use, X2_use, beta1, beta2
        )
        
    elif method == 'pna-mlp':
        if pna_embeddings is None:
            raise ValueError("PNA embeddings not available")
        # MLP methods don't have simple beta coefficients, skip variance estimation
        res = grr_estimator(
            Y, D, X_run, d1, d2, pi_id_all, Kn,
            method='mlp', eps=eps,
            X1=pna_embeddings['X_PNA_root'],
            X2=pna_embeddings['X_PNA_root'],
            **{k: v for k, v in grr_kwargs.items() 
               if k in ['hidden_dims', 'n_epochs', 'lr', 'weight_decay', 
                       'dropout', 'early_stopping', 'patience']}
        )
        return _result_no_var(res['tau_hat'], res['loss'])
    
    elif method == 'pna-no-harm':
        # No-harm calibration: use predicted potential outcomes as covariates
        if pna_model is None:
            raise ValueError("PNA model not available")
        Ypred_d1 = pna_model.predict(X_run, D=d1)
        Ypred_d2 = pna_model.predict(X_run, D=d2)
        X1_noharm = Ypred_d1.reshape(-1, 1)
        X2_noharm = Ypred_d2.reshape(-1, 1)
        res = grr_estimator(
            Y, D, X_run, d1, d2, pi_id_all, Kn,
            method='linear', eps=eps,
            X1=X1_noharm, X2=X2_noharm,
            **{k: v for k, v in grr_kwargs.items() if k in ['alpha']}
        )
        inputs = prepare_grr_inputs(Y, D, X_run, d1, d2, pi_id_all, X1=X1_noharm, X2=X2_noharm, eps=eps)
        beta1, beta2 = res['model'].get_coefficients()
        return _result_with_var(
            res['tau_hat'], res['loss'],
            inputs['hatY_d1'], inputs['hatY_d2'],
            inputs['hat1_d1'], inputs['hat1_d2'],
            X1_noharm, X2_noharm, beta1, beta2
        )
    
    elif method == 'pna-pred':
        # Doubly robust estimator using PNA predictions directly
        if pna_model is None:
            raise ValueError("PNA model not available")
        Ypred_d1 = pna_model.predict(X_run, D=d1)
        Ypred_d2 = pna_model.predict(X_run, D=d2)
        X1_pred = Ypred_d1.reshape(-1, 1)
        X2_pred = Ypred_d2.reshape(-1, 1)
        beta1 = np.array([1.0])
        beta2 = np.array([1.0])
        mu1 = _mu_from_beta(Y, D, d1, pi1, X1_pred, beta1, eps)
        mu2 = _mu_from_beta(Y, D, d2, pi2, X2_pred, beta2, eps)
        # Can compute variance for this method
        inputs = prepare_grr_inputs(Y, D, X_run, d1, d2, pi_id_all, X1=X1_pred, X2=X2_pred, eps=eps)
        return _result_with_var(
            float(mu1 - mu2), np.nan,
            inputs['hatY_d1'], inputs['hatY_d2'],
            inputs['hat1_d1'], inputs['hat1_d2'],
            X1_pred, X2_pred, beta1, beta2
        )
    
    # =========================================================================
    # End-to-end PNA-GRR (trains PNA directly on GRR loss)
    # =========================================================================
    elif method == 'pna':
        if A_adj is None:
            raise ValueError("PNA method requires adjacency matrix A_adj")
        res = grr_estimator(
            Y, D, X_run, d1, d2, pi_id_all, Kn,
            method='pna', eps=eps,
            adjacency=A_adj,
            **{k: v for k, v in grr_kwargs.items() 
               if k in ['pna_hidden_dims', 'rep_dim', 'head_hidden_dims',
                       'aggregators', 'scalers', 'dropout',
                       'n_epochs', 'lr', 'weight_decay', 
                       'early_stopping', 'patience', 'verbose']}
        )
        # End-to-end PNA doesn't have simple beta coefficients
        return _result_no_var(res['tau_hat'], res['loss'])
    
    # =========================================================================
    # Standard GRR methods (with variance estimation for linear methods)
    # =========================================================================
    else:
        res = grr_estimator(
            Y, D, X_run, d1, d2, pi_id_all, Kn,
            method=method, eps=eps, **grr_kwargs
        )
        # Check if method has get_coefficients (linear methods do, MLP doesn't)
        if hasattr(res['model'], 'get_coefficients'):
            try:
                inputs = prepare_grr_inputs(Y, D, X_run, d1, d2, pi_id_all, eps=eps)
                beta1, beta2 = res['model'].get_coefficients()
                return _result_with_var(
                    res['tau_hat'], res['loss'],
                    inputs['hatY_d1'], inputs['hatY_d2'],
                    inputs['hat1_d1'], inputs['hat1_d2'],
                    inputs['X1'], inputs['X2'], beta1, beta2
                )
            except Exception:
                pass
        return _result_no_var(res['tau_hat'], res['loss'])


# =============================================================================
# Visualization
# =============================================================================

def plot_grr_results(out, figsize=(16, 5)):
    """
    Plot simulation results: boxplot of errors and bar chart of bias/sd/rmse.
    
    Args:
        out: output dict from run_grr_simulation
        figsize: figure size tuple
    
    Returns:
        matplotlib Figure object
    """
    true_tau = out['true_tau']
    summary = out['summary']
    results = out['results']
    
    # Filter to valid methods only
    valid_methods = [m for m in out['methods'] if summary[m]['n_valid'] > 0]
    
    if len(valid_methods) == 0:
        print("No valid methods to plot!")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Boxplot of errors
    errors = [results[m][~np.isnan(results[m])] - true_tau for m in valid_methods]
    bp = axes[0].boxplot(errors, labels=valid_methods, patch_artist=True, showfliers=False)
    colors = plt.cm.Set3(np.linspace(0, 1, len(valid_methods)))
    for box, color in zip(bp['boxes'], colors):
        box.set_facecolor(color)
        box.set_edgecolor('#444444')
    for median in bp['medians']:
        median.set_color('#222222')
        median.set_linewidth(1.5)
    axes[0].axhline(0.0, color='k', linestyle='--', linewidth=1, alpha=0.6)
    axes[0].set_ylabel(r'Error $\hat{\tau} - \tau^*$')
    axes[0].set_title('Estimator Errors')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Bar chart of bias/sd/rmse
    x = np.arange(len(valid_methods))
    width = 0.25
    
    biases = [summary[m]['bias'] for m in valid_methods]
    sds = [summary[m]['sd'] for m in valid_methods]
    rmses = [summary[m]['rmse'] for m in valid_methods]
    
    axes[1].bar(x - width, biases, width=width, label='Bias', alpha=0.8)
    axes[1].bar(x, sds, width=width, label='SD', alpha=0.8)
    axes[1].bar(x + width, rmses, width=width, label='RMSE', alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(valid_methods)
    axes[1].set_ylabel('Value')
    axes[1].set_title('Bias / SD / RMSE by Method')
    axes[1].legend()
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig


# =============================================================================
# Results I/O
# =============================================================================

def save_results(out, results_dir='results', name_prefix='simulation'):
    """
    Save simulation results to files with timestamp.
    
    Saves:
    - Summary statistics as CSV
    - Config/metadata as JSON
    - Full results as NPZ
    - Plot as PNG
    
    Args:
        out: output dict from run_grr_simulation
        results_dir: directory to save results
        name_prefix: prefix for output files
    
    Returns:
        dict with paths to saved files
    """
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = f"{name_prefix}_{timestamp}"
    
    # -------------------------------------------------------------------------
    # Save summary as CSV
    # -------------------------------------------------------------------------
    summary_data = []
    for method in out['methods']:
        s = out['summary'][method]
        row = {
            'method': method,
            'bias': s['bias'],
            'sd': s['sd'],
            'rmse': s['rmse'],
            'n_valid': s['n_valid']
        }
        # Add mean loss only for methods that have one
        if method not in PRIMITIVE_METHODS:
            losses = out['losses'][method]
            valid_losses = losses[~np.isnan(losses)]
            row['mean_loss'] = float(np.mean(valid_losses)) if len(valid_losses) > 0 else np.nan
        else:
            row['mean_loss'] = np.nan
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(results_dir, f"{base_name}_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")
    
    # -------------------------------------------------------------------------
    # Save config/metadata as JSON
    # -------------------------------------------------------------------------
    config = {
        'timestamp': timestamp,
        'true_tau': out['true_tau'],
        'n': out['G'].shape[0],
        'methods': out['methods'],
        'summary': out['summary'],
        'r2_summary': out.get('r2_summary', {})
    }
    config_path = os.path.join(results_dir, f"{base_name}_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to: {config_path}")
    
    # -------------------------------------------------------------------------
    # Save full results as NPZ
    # -------------------------------------------------------------------------
    results_arrays = {f"results_{m}": out['results'][m] for m in out['methods']}
    losses_arrays = {f"losses_{m}": out['losses'][m] for m in out['methods']}
    npz_path = os.path.join(results_dir, f"{base_name}_full.npz")
    np.savez(npz_path, **results_arrays, **losses_arrays)
    print(f"Full results saved to: {npz_path}")
    
    # -------------------------------------------------------------------------
    # Save plot as PNG
    # -------------------------------------------------------------------------
    fig = plot_grr_results(out, figsize=(16, 5))
    if fig is not None:
        plot_path = os.path.join(results_dir, f"{base_name}_plot.png")
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Plot saved to: {plot_path}")
    else:
        plot_path = None
    
    return {
        'summary_csv': summary_path,
        'config_json': config_path,
        'full_npz': npz_path,
        'plot_png': plot_path
    }


def print_summary_table(out):
    """
    Print a nicely formatted summary table to console.
    
    Args:
        out: output dict from run_grr_simulation
    """
    print("\n" + "="*110)
    print(f"Simulation Results (true_tau = {out['true_tau']:.4f})")
    print("="*110)
    
    # Header
    print(f"{'Method':<18} {'Bias':>10} {'SD':>10} {'RMSE':>10} {'Coverage':>10} {'Power':>10} {'Mean Loss':>12} {'N Valid':>8}")
    print("-"*110)
    
    for method in out['methods']:
        s = out['summary'][method]
        if s['n_valid'] > 0:
            bias_str = f"{s['bias']:.4f}"
            sd_str = f"{s['sd']:.4f}"
            rmse_str = f"{s['rmse']:.4f}"
        else:
            bias_str = sd_str = rmse_str = "N/A"
        
        # Coverage and power
        if s.get('n_ci_valid', 0) > 0 and not np.isnan(s.get('coverage', np.nan)):
            cov_str = f"{s['coverage']:.3f}"
            pow_str = f"{s['power']:.3f}"
        else:
            cov_str = pow_str = "-"
        
        # Mean loss
        if method not in PRIMITIVE_METHODS and s['n_valid'] > 0:
            losses = out['losses'][method]
            valid_losses = losses[~np.isnan(losses)]
            loss_str = f"{np.mean(valid_losses):.4f}" if len(valid_losses) > 0 else "N/A"
        else:
            loss_str = "-"
        
        print(f"{method:<18} {bias_str:>10} {sd_str:>10} {rmse_str:>10} {cov_str:>10} {pow_str:>10} {loss_str:>12} {s['n_valid']:>8}")
    
    print("="*110)
    
    # Print R² summary
    if 'r2_summary' in out:
        print("\nOutcome Prediction R² (how well models predict Y given X):")
        print("-"*55)
        r2_sum = out['r2_summary']
        if 'linear' in r2_sum:
            print(f"  Linear model:  R² = {r2_sum['linear']['mean']:.4f} ± {r2_sum['linear']['std']:.4f}")
        if 'gat' in r2_sum:
            print(f"  GAT model:     R² = {r2_sum['gat']['mean']:.4f} ± {r2_sum['gat']['std']:.4f}")
        if 'pna' in r2_sum:
            print(f"  PNA model:     R² = {r2_sum['pna']['mean']:.4f} ± {r2_sum['pna']['std']:.4f}")
        print("-"*55)
    
    # Print coverage/power note
    print("\nNote: Coverage = proportion of 95% CIs containing true_tau")
    print("      Power = proportion of 95% CIs not containing 0 (reject H0: tau=0)")