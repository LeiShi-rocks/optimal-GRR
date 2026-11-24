"""
Variance estimation for GRR estimators.

This module implements variance estimation for the Generalized Regression Regression (GRR) 
treatment effect estimators, following a reweighted residual approach.

The variance estimator works as follows:
1. Compute reweighted residuals: eps_hat_d = hatY_d - hat1_d * pred_d
   where pred_d = X_d @ beta_d (covariates/embeddings times final calibrated coefficients)
2. Construct variance components: vhat_d = eps_hat_d' @ W_d_pi @ eps_hat_d / sum(hat1_d)^2
   (normalized by sum of IPW weights squared for stability)
3. Final variance estimator: vhat = (sqrt(vhat_d1) + sqrt(vhat_d2))^2
"""

import numpy as np
from typing import Tuple


def compute_reweighted_residuals(
    hatY_d: np.ndarray,
    hat1_d: np.ndarray,
    X: np.ndarray,
    beta: np.ndarray
) -> np.ndarray:
    """
    Compute reweighted residuals for variance estimation.
    
    eps_hat = hatY_d - hat1_d * (X @ beta)
    
    Args:
        hatY_d: weighted outcomes hat1_d * Y (n,)
        hat1_d: IPW weights I(D=d) / pi_d (n,)
        X: covariates or embeddings (n x p)
        beta: regression coefficients (p,)
    
    Returns:
        eps_hat: reweighted residuals (n,)
    """
    pred = X @ beta
    eps_hat = hatY_d - hat1_d * pred
    return eps_hat


def compute_variance_component(
    eps_hat: np.ndarray,
    W_pi: np.ndarray,
    hat1_d: np.ndarray
) -> float:
    """
    Compute variance component for a single exposure level.
    
    vhat = eps_hat' @ W_pi @ eps_hat / sum(hat1_d)^2
    
    Using sum(hat1_d) instead of n stabilizes the estimator when 
    propensity scores vary significantly.
    
    Args:
        eps_hat: reweighted residuals (n,)
        W_pi: variance weighting matrix (n x n)
        hat1_d: IPW weights I(D=d) / pi_d (n,)
    
    Returns:
        vhat: variance component (scalar)
    """
    # sum_weights = np.sum(hat1_d) if robust else len(hat1_d)
    sum_weights = len(hat1_d)
    vhat = eps_hat @ W_pi @ eps_hat / (sum_weights ** 2)
    return float(vhat)


def grr_variance_estimator(
    hatY_d1: np.ndarray,
    hatY_d2: np.ndarray,
    hat1_d1: np.ndarray,
    hat1_d2: np.ndarray,
    X1: np.ndarray,
    X2: np.ndarray,
    beta1: np.ndarray,
    beta2: np.ndarray,
    W1_pi: np.ndarray,
    W2_pi: np.ndarray
) -> Tuple[float, float, float]:
    """
    Compute variance estimate for GRR treatment effect estimator.
    
    The estimator is: vhat = (sqrt(vhat_d1) + sqrt(vhat_d2))^2
    
    This works for any GRR method where predictions are of the form X @ beta:
    - Linear GRR: X is covariates, beta is OLS/Ridge/Lasso coefficients
    - Embedding GRR (PNA/GAT): X is embeddings, beta is calibrated coefficients
    - No-harm calibration: X is original predictions, beta is calibration coefficient
    
    Args:
        hatY_d1: weighted outcomes for d1, i.e., hat1_d1 * Y (n,)
        hatY_d2: weighted outcomes for d2, i.e., hat1_d2 * Y (n,)
        hat1_d1: IPW weights for d1, i.e., I(D=d1) / pi_d1 (n,)
        hat1_d2: IPW weights for d2, i.e., I(D=d2) / pi_d2 (n,)
        X1: covariates/embeddings for d1 (n x p1)
        X2: covariates/embeddings for d2 (n x p2)
        beta1: coefficients for d1 (p1,)
        beta2: coefficients for d2 (p2,)
        W1_pi: variance weighting matrix for d1 (n x n), from build_W_pi
        W2_pi: variance weighting matrix for d2 (n x n), from build_W_pi
    
    Returns:
        vhat: final variance estimate (sqrt(vhat_d1) + sqrt(vhat_d2))^2
        vhat_d1: variance component for d1
        vhat_d2: variance component for d2
    """
    # Compute reweighted residuals
    eps_hat_d1 = compute_reweighted_residuals(hatY_d1, hat1_d1, X1, beta1)
    eps_hat_d2 = compute_reweighted_residuals(hatY_d2, hat1_d2, X2, beta2)
    
    # Compute variance components (normalized by sum of weights for stability)
    vhat_d1 = compute_variance_component(eps_hat_d1, W1_pi, hat1_d1)
    vhat_d2 = compute_variance_component(eps_hat_d2, W2_pi, hat1_d2)
    
    # Final variance estimator: (sqrt(vhat_d1) + sqrt(vhat_d2))^2
    vhat = (np.sqrt(max(vhat_d1, 0)) + np.sqrt(max(vhat_d2, 0))) ** 2
    
    return vhat, vhat_d1, vhat_d2


def grr_variance_from_grr_result(
    grr_result: dict,
    inputs: dict,
    W1_pi: np.ndarray,
    W2_pi: np.ndarray
) -> Tuple[float, float, float]:
    """
    Compute variance estimate from GRR result dictionary.
    
    This is a convenience function that extracts the necessary components
    from the output of grr_estimator() and prepare_grr_inputs().
    
    Args:
        grr_result: output from grr_estimator() containing 'model' with coefficients
        inputs: output from prepare_grr_inputs() containing hatY, hat1, X
        W1_pi: variance weighting matrix for d1 (n x n)
        W2_pi: variance weighting matrix for d2 (n x n)
    
    Returns:
        vhat: final variance estimate
        vhat_d1: variance component for d1
        vhat_d2: variance component for d2
    """
    model = grr_result['model']
    
    # Get coefficients from model
    beta1, beta2 = model.get_coefficients()
    
    return grr_variance_estimator(
        hatY_d1=inputs['hatY_d1'],
        hatY_d2=inputs['hatY_d2'],
        hat1_d1=inputs['hat1_d1'],
        hat1_d2=inputs['hat1_d2'],
        X1=inputs['X1'],
        X2=inputs['X2'],
        beta1=beta1,
        beta2=beta2,
        W1_pi=W1_pi,
        W2_pi=W2_pi
    )


def grr_standard_error(
    hatY_d1: np.ndarray,
    hatY_d2: np.ndarray,
    hat1_d1: np.ndarray,
    hat1_d2: np.ndarray,
    X1: np.ndarray,
    X2: np.ndarray,
    beta1: np.ndarray,
    beta2: np.ndarray,
    W1_pi: np.ndarray,
    W2_pi: np.ndarray
) -> float:
    """
    Compute standard error for the GRR treatment effect estimator.
    
    SE = sqrt(vhat)
    
    (Note: vhat is already normalized by sum(hat1_d)^2 in compute_variance_component)
    
    Args:
        (same as grr_variance_estimator)
    
    Returns:
        se: standard error of tau_hat
    """
    vhat, _, _ = grr_variance_estimator(
        hatY_d1, hatY_d2, hat1_d1, hat1_d2,
        X1, X2, beta1, beta2, W1_pi, W2_pi
    )
    
    se = np.sqrt(vhat)
    return se


def grr_confidence_interval(
    tau_hat: float,
    hatY_d1: np.ndarray,
    hatY_d2: np.ndarray,
    hat1_d1: np.ndarray,
    hat1_d2: np.ndarray,
    X1: np.ndarray,
    X2: np.ndarray,
    beta1: np.ndarray,
    beta2: np.ndarray,
    W1_pi: np.ndarray,
    W2_pi: np.ndarray,
    alpha: float = 0.05
) -> Tuple[float, float]:
    """
    Compute confidence interval for the GRR treatment effect estimator.
    
    CI = tau_hat +/- z_{1-alpha/2} * SE
    
    Args:
        tau_hat: point estimate of treatment effect
        (other args same as grr_variance_estimator)
        alpha: significance level (default 0.05 for 95% CI)
    
    Returns:
        ci_lower: lower bound of confidence interval
        ci_upper: upper bound of confidence interval
    """
    from scipy import stats
    
    se = grr_standard_error(
        hatY_d1, hatY_d2, hat1_d1, hat1_d2,
        X1, X2, beta1, beta2, W1_pi, W2_pi
    )
    
    z = stats.norm.ppf(1 - alpha / 2)
    ci_lower = tau_hat - z * se
    ci_upper = tau_hat + z * se
    
    return ci_lower, ci_upper
