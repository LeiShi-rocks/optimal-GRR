"""
Estimators module for HAIPW simulation.

This module contains functions for constructing various causal effect estimators:
- Horvitz-Thompson (HT) estimator
- Hájek estimator
- HAIPW estimators with different matrix/vector combinations
- HAIPW-cov estimators (with covariates)
"""

import numpy as np


# ---------- Base weights / HT / Hájek ----------
def _weights(D, d, pi, eps=1e-12):
    """
    Compute IPW weights for exposure level d.
    
    Args:
        D: observed exposure vector (n,)
        d: target exposure level
        pi: propensity scores for exposure d (n,)
        eps: small constant for numerical stability
    
    Returns:
        weights: IPW weights (n,)
    """
    return (D == d).astype(float) / (pi + eps)


def mu_ht(Y, D, d, pi_id_all, eps=1e-12):
    """
    Horvitz-Thompson estimator for E[Y(d)].
    
    Args:
        Y: observed outcomes (n,)
        D: observed exposure (n,)
        d: target exposure level
        pi_id_all: propensity score matrix (n x k)
        eps: small constant for numerical stability
    
    Returns:
        mu_hat: HT estimate of E[Y(d)]
    """
    w = _weights(D, d, pi_id_all[:, d], eps)
    return float(np.mean(w * Y))


def one_ht(D, d, pi_id_all, eps=1e-12):
    """
    HT estimator for the normalizing constant (should equal 1 in expectation).
    
    Args:
        D: observed exposure (n,)
        d: target exposure level
        pi_id_all: propensity score matrix (n x k)
        eps: small constant for numerical stability
    
    Returns:
        one_hat: HT estimate of 1
    """
    w = _weights(D, d, pi_id_all[:, d], eps)
    return float(np.mean(w))


def mu_haj(Y, D, d, pi_id_all, eps=1e-12):
    """
    Hájek estimator for E[Y(d)] (normalized HT).
    
    Args:
        Y: observed outcomes (n,)
        D: observed exposure (n,)
        d: target exposure level
        pi_id_all: propensity score matrix (n x k)
        eps: small constant for numerical stability
    
    Returns:
        mu_hat: Hájek estimate of E[Y(d)]
    """
    return mu_ht(Y, D, d, pi_id_all, eps) / (one_ht(D, d, pi_id_all, eps) + eps)


def _solve(A, b):
    """
    Solve linear system Ax = b using pseudoinverse.
    
    Args:
        A: matrix (m x n)
        b: vector (m,)
    
    Returns:
        x: solution vector (n,)
    """
    return np.linalg.pinv(A, rcond=1e-2) @ b


# ---------- HAIPW estimator forms ----------
def _mu_from_c(Y, D, d, pi, c, eps=1e-12):
    """
    Compute HAIPW estimator using constant adjustment c.
    
    mu(d) = (1/n) sum_i { w_i(d) * (Y_i - c) } + c
    
    Args:
        Y: observed outcomes (n,)
        D: observed exposure (n,)
        d: target exposure level
        pi: propensity scores for exposure d (n,)
        c: constant adjustment
        eps: small constant for numerical stability
    
    Returns:
        mu_hat: HAIPW estimate of E[Y(d)]
    """
    w = _weights(D, d, pi, eps)
    n = Y.size
    return float((w * (Y - c)).sum() / n + c)


def _mu_from_beta(Y, D, d, pi, X, beta, eps=1e-12):
    """
    Compute HAIPW-cov estimator using covariate adjustment.
    
    mu(d) = (1/n) sum_i { w_i(d) * (Y_i - X_i'beta) } + (1/n) sum_i X_i'beta
    
    Args:
        Y: observed outcomes (n,)
        D: observed exposure (n,)
        d: target exposure level
        pi: propensity scores for exposure d (n,)
        X: covariate matrix (n x p)
        beta: coefficient vector (p,)
        eps: small constant for numerical stability
    
    Returns:
        mu_hat: HAIPW-cov estimate of E[Y(d)]
    """
    w = _weights(D, d, pi, eps)
    n = Y.size
    m = X @ beta
    return float((w * (Y - m)).sum() / n + m.mean())


# ---------- Building blocks for the regression strategy ----------
def _stack_for_reg(Y, D, X, d1, d2, pi1, pi2, eps=1e-12):
    """
    Build stacked quantities for regression-based estimation.
    
    Args:
        Y: observed outcomes (n,)
        D: observed exposure (n,)
        X: covariate matrix (n x p)
        d1, d2: exposure levels to compare
        pi1, pi2: propensity scores for d1 and d2 (n,)
        eps: small constant for numerical stability
    
    Returns:
        hT1: stacked tilde1 vectors (n x 2)
        hTX: stacked tildeX matrices (n x 2p)
        dHY: difference in weighted outcomes (n,)
    """
    # w1 is hat1_d1, w2 is hat1_d2
    hat1_d1 = _weights(D, d1, pi1, eps)
    hat1_d2 = _weights(D, d2, pi2, eps)
    # hatX_d1 is Xw1, hatX_d2 is Xw2
    hatX_d1 = X * hat1_d1[:, None]
    hatX_d2 = X * hat1_d2[:, None]
    # hatY_d1 is w1Y, hatY_d2 is w2Y
    hatY_d1 = hat1_d1 * Y
    hatY_d2 = hat1_d2 * Y

    hT1 = np.column_stack([hat1_d1 - 1.0, -hat1_d2 + 1.0])
    hTX = np.hstack([hatX_d1 - X, -hatX_d2 + X])
    dHY = hatY_d1 - hatY_d2

    return hT1, hTX, dHY


# ---------- (2×2) matrix choices for constants ----------
def _Omega_const(matrix_choice, L1, L2, L12, Kn, w1=None, w2=None, hT1=None):
    """
    Build Omega matrix for constant HAIPW estimation.
    
    matrix_choice:
      - 'truth'      : sums of L's (uses true variance structure)
      - 'one_sided'  : weighted sums (needs w1, w2)
      - 'regression' : tilde1^T Kn tilde1 (second strategy, not using omega)
    
    Args:
        matrix_choice: one of 'truth', 'one_sided', 'regression'
        L1, L2, L12: Lambda matrices
        Kn: neighbor indicator matrix
        w1, w2: IPW weights for d1 and d2 (required for 'one_sided')
        hT1: stacked tilde1 vectors (required for 'regression')
    
    Returns:
        Omega: 2x2 matrix
    """
    if matrix_choice == 'truth':
        sL1 = float(L1.sum())
        sL2 = float(L2.sum())
        sL12 = float(L12.sum())
        return np.array([[sL1, -sL12],
                         [-sL12, sL2]], float)
    elif matrix_choice == 'one_sided':
        assert w1 is not None and w2 is not None
        sL1 = float((L1 @ w1).sum())
        sL2 = float((L2 @ w2).sum())
        sL12_21 = float((L12 @ w1).sum())
        sL12_12 = float((L12 @ w2).sum())
        return np.array([[sL1, -sL12_12],
                         [-sL12_21, sL2]], float)
    elif matrix_choice == 'regression':
        assert hT1 is not None
        return hT1.T @ Kn @ hT1
    else:
        raise ValueError("Unknown matrix_choice for constants.")


# ---------- (vector) choices for constants ----------
def _vec_const(vector_choice, L1, L2, L12, Kn, w1Y=None, w2Y=None, hT1=None, dHY=None):
    """
    Build vector for constant HAIPW estimation.
    
    vector_choice:
      - 'one_sided'  : Λ-weighted vector (needs w1Y, w2Y)
      - 'regression' : tilde1^T Kn dHY (needs hT1, dHY)
    
    Args:
        vector_choice: one of 'one_sided', 'regression'
        L1, L2, L12: Lambda matrices
        Kn: neighbor indicator matrix
        w1Y, w2Y: weighted outcomes (required for 'one_sided')
        hT1: stacked tilde1 vectors (required for 'regression')
        dHY: difference in weighted outcomes (required for 'regression')
    
    Returns:
        vec: length-2 vector
    """
    if vector_choice == 'one_sided':
        assert w1Y is not None and w2Y is not None
        v1 = float((L1 @ w1Y).sum())
        v2 = float((L2 @ w2Y).sum())
        v12_d1 = float((L12 @ w1Y).sum())
        v12_d2 = float((L12 @ w2Y).sum())
        return np.array([v1 - v12_d2, v2 - v12_d1], float)
    elif vector_choice == 'regression':
        assert hT1 is not None and dHY is not None
        return hT1.T @ Kn @ dHY
    else:
        raise ValueError("Unknown vector_choice for constants.")


# ---------- (2p×2p) matrix choices for covariates ----------
def _Omega_cov(matrix_choice, L1, L2, L12, Kn, X=None, Xw1=None, Xw2=None, hTX=None):
    """
    Build Omega matrix for covariate HAIPW estimation.
    
    matrix_choice:
      - 'truth'      : OmegaX_true = [[X^T L1 X, -X^T L12 X],
                                      [-X^T L12^T X, X^T L2 X]]
      - 'one_sided'  : weighted blocks with Xw1, Xw2
      - 'regression' : tildeX^T Kn tildeX (second strategy)
    
    Args:
        matrix_choice: one of 'truth', 'one_sided', 'regression'
        L1, L2, L12: Lambda matrices
        Kn: neighbor indicator matrix
        X: covariate matrix (required for 'truth', 'one_sided')
        Xw1, Xw2: weighted covariates (required for 'one_sided')
        hTX: stacked tildeX matrices (required for 'regression')
    
    Returns:
        OmegaX: 2p x 2p matrix
    """
    if matrix_choice == 'truth':
        Omega11 = X.T @ L1 @ X
        Omega22 = X.T @ L2 @ X
        Omega12 = X.T @ L12 @ X
        return np.block([[Omega11, -Omega12],
                         [-Omega12.T, Omega22]])
    elif matrix_choice == 'one_sided':
        assert Xw1 is not None and Xw2 is not None
        Omega11 = X.T @ (L1 @ Xw1)
        Omega22 = X.T @ (L2 @ Xw2)
        Omega12 = X.T @ (L12 @ Xw2)
        Omega21 = X.T @ (L12 @ Xw1)
        return np.block([[Omega11, -Omega12],
                         [-Omega21, Omega22]])
    elif matrix_choice == 'regression':
        assert hTX is not None
        return hTX.T @ Kn @ hTX
    else:
        raise ValueError("Unknown matrix_choice for covariates.")


# ---------- (vector) choices for covariates ----------
def _vec_cov(vector_choice, L1, L2, L12, Kn, X=None, w1Y=None, w2Y=None, hTX=None, dHY=None):
    """
    Build vector for covariate HAIPW estimation.
    
    vector_choice:
      - 'one_sided'  : vecX = [X^T L1 (w1Y) - X^T L12 (w2Y);  
                               X^T L2 (w2Y) - X^T L12 (w1Y)]
      - 'regression' : tildeX^T Kn (hatY(d1) - hatY(d2))
    
    Args:
        vector_choice: one of 'one_sided', 'regression'
        L1, L2, L12: Lambda matrices
        Kn: neighbor indicator matrix
        X: covariate matrix (required for 'one_sided')
        w1Y, w2Y: weighted outcomes (required for 'one_sided')
        hTX: stacked tildeX matrices (required for 'regression')
        dHY: difference in weighted outcomes (required for 'regression')
    
    Returns:
        vecX: length-2p vector
    """
    if vector_choice == 'one_sided':
        assert X is not None and w1Y is not None and w2Y is not None
        sXY_d1 = X.T @ (L1 @ w1Y)
        sXY_d2 = X.T @ (L2 @ w2Y)
        sXY12_d1 = X.T @ (L12 @ w1Y)
        sXY12_d2 = X.T @ (L12 @ w2Y)
        return np.concatenate([sXY_d1 - sXY12_d2,
                               sXY_d2 - sXY12_d1], axis=0)
    elif vector_choice == 'regression':
        assert hTX is not None and dHY is not None
        return hTX.T @ Kn @ dHY
    else:
        raise ValueError("Unknown vector_choice for covariates.")


# ---------- Compute all estimators for one replication ----------
def compute_all_estimators_once(Y, D, X, d1, d2, L1, L2, L12, Kn, pi_id_all, eps=1e-12):
    """
    Compute all estimators for a single treatment assignment.
    
    Estimators computed:
    - HT (Horvitz-Thompson)
    - Hájek (normalized HT)
    - HAIPW with 3 variants: (truth, one_sided), (one_sided, one_sided), (regression, regression)
    - HAIPW-cov with 3 variants: same combinations
    
    Args:
        Y: observed outcomes (n,)
        D: observed exposure (n,)
        X: covariate matrix with intercept (n x p)
        d1, d2: exposure levels to compare
        L1, L2, L12: Lambda matrices
        Kn: neighbor indicator matrix
        pi_id_all: propensity score matrix (n x k)
        eps: small constant for numerical stability
    
    Returns:
        out: dict of estimator names -> tau estimates
        beta_out: dict of estimator names -> beta coefficients (for HAIPW-cov only)
    """
    n, p = X.shape
    pi1 = pi_id_all[:, d1]
    pi2 = pi_id_all[:, d2]
    w1 = _weights(D, d1, pi1, eps)
    w2 = _weights(D, d2, pi2, eps)
    w1Y = w1 * Y
    w2Y = w2 * Y

    # HT estimator
    mu1_ht = mu_ht(Y, D, d1, pi_id_all, eps)
    mu2_ht = mu_ht(Y, D, d2, pi_id_all, eps)

    # Hajek estimator
    mu1_haj = mu_haj(Y, D, d1, pi_id_all, eps)
    mu2_haj = mu_haj(Y, D, d2, pi_id_all, eps)

    # Building blocks for the second strategy
    hT1, hTX, dHY = _stack_for_reg(Y, D, X, d1, d2, pi1, pi2, eps)

    # Containers
    out = dict()
    beta_out = dict()
    out["tau_ht"] = float(mu1_ht - mu2_ht)
    out["tau_hajek"] = float(mu1_haj - mu2_haj)

    # --- HAIPW (constants) 3 variants ---
    estimator_configs = [
        ("truth", "one_sided"),
        ("one_sided", "one_sided"),
        ("regression", "regression")
    ]

    for M, V in estimator_configs:
        Om = _Omega_const(M, L1, L2, L12, Kn, w1=w1, w2=w2, hT1=hT1)
        vec = _vec_const(V, L1, L2, L12, Kn, w1Y=w1Y, w2Y=w2Y, hT1=hT1, dHY=dHY)
        c_hat = _solve(Om, vec)  # (2,)
        c1, c2 = float(c_hat[0]), float(c_hat[1])
        mu1 = _mu_from_c(Y, D, d1, pi1, c1, eps)
        mu2 = _mu_from_c(Y, D, d2, pi2, c2, eps)
        key = f"tau_haipw[M={M},V={V}]"
        out[key] = float(mu1 - mu2)

    # --- HAIPW-cov (with X) 3 variants ---
    Xw1 = X * w1[:, None]  # reshape w1 to (n,1), Xw1 has (n,p)
    Xw2 = X * w2[:, None]
    for M, V in estimator_configs:
        OmX = _Omega_cov(M, L1, L2, L12, Kn, X=X, Xw1=Xw1, Xw2=Xw2, hTX=hTX)
        vecX = _vec_cov(V, L1, L2, L12, Kn, X=X, w1Y=w1Y, w2Y=w2Y, hTX=hTX, dHY=dHY)
        beta_hat = _solve(OmX, vecX)  # (2p,)
        beta1, beta2 = beta_hat[:p], beta_hat[p:]
        mu1 = _mu_from_beta(Y, D, d1, pi1, X, beta1, eps)
        mu2 = _mu_from_beta(Y, D, d2, pi2, X, beta2, eps)
        key = f"tau_haipw_cov[M={M},V={V}]"
        out[key] = float(mu1 - mu2)
        beta_out[key] = beta_hat

    return (out, beta_out)

