"""
Propensity scores and Lambda matrices module for HAIPW simulation.

This module contains functions for:
- Monte Carlo estimation of propensity scores (pi)
- Building Lambda matrices for variance estimation
"""

import numpy as np

from dgp import compute_exposure


# ---------- Step 7: MC probabilities ----------
def monte_carlo_probabilities(G, treated_fraction, k, mc_reps, seed_mc, d1, d2):
    """
    Estimate propensity scores and joint probabilities via Monte Carlo.
    
    Estimates:
      - pi_id_all[i,d] = P(D_i=d)  for all d=0..k-1
      - pi_ij_d1d2[i,j] = P(D_i=d1, D_j=d2)
      - pi_ij_d1d1, pi_ij_d2d2   (for Λ matrices)
    
    Args:
        G: adjacency matrix with self-loops
        treated_fraction: fraction of nodes to treat
        k: number of exposure categories (exposures in {0, 1, ..., k-1})
        mc_reps: number of Monte Carlo replicates
        seed_mc: random seed
        d1, d2: exposure levels to compare (must be in {0, 1, ..., k-1})
    
    Returns:
        pi_id_all: marginal probabilities (n x k)
        pi_ij_d1d2: joint probability P(D_i=d1, D_j=d2) (n x n)
        pi_ij_d1d1: joint probability P(D_i=d1, D_j=d1) (n x n)
        pi_ij_d2d2: joint probability P(D_i=d2, D_j=d2) (n x n)
    """
    rng = np.random.default_rng(seed_mc)
    n = G.shape[0]
    treated_count = int(round(treated_fraction * n))

    counts_id = np.zeros((n, k), dtype=np.int64)
    joint_d1d2 = np.zeros((n, n), dtype=np.int64)
    joint_d1d1 = np.zeros((n, n), dtype=np.int64)
    joint_d2d2 = np.zeros((n, n), dtype=np.int64)

    for _ in range(mc_reps):
        A = np.zeros(n, dtype=np.int8)
        A[rng.choice(n, size=treated_count, replace=False)] = 1
        D = compute_exposure(G, A, k)
        if np.any((D < 0) | (D >= k)):
            raise ValueError(f"Exposure out of range during MC; got values {np.unique(D)}, expected 0..{k-1}.")
        counts_id[np.arange(n), D] += 1

        m1 = (D == d1).astype(np.int8)
        m2 = (D == d2).astype(np.int8)
        joint_d1d2 += np.outer(m1, m2)
        joint_d1d1 += np.outer(m1, m1)
        joint_d2d2 += np.outer(m2, m2)

    pi_id_all = counts_id / float(mc_reps)
    pi_ij_d1d2 = joint_d1d2 / float(mc_reps)
    pi_ij_d1d1 = joint_d1d1 / float(mc_reps)
    pi_ij_d2d2 = joint_d2d2 / float(mc_reps)
    np.fill_diagonal(pi_ij_d1d2, 0.0)
    np.fill_diagonal(pi_ij_d1d1, 0.0)
    np.fill_diagonal(pi_ij_d2d2, 0.0)

    return pi_id_all, pi_ij_d1d2, pi_ij_d1d1, pi_ij_d2d2


# ---------- Step 8: Lambda matrices ----------
def build_lambdas(pi_id_all, pi_ij_d1d2, pi_ij_d1d1, pi_ij_d2d2, d1, d2, eps=1e-12):
    """
    Build Lambda matrices for variance estimation.
    
    These matrices capture the covariance structure of the IPW estimators
    under the randomization distribution.
    
    Args:
        pi_id_all: marginal probabilities (n x k)
        pi_ij_d1d2: joint probability P(D_i=d1, D_j=d2) (n x n)
        pi_ij_d1d1: joint probability P(D_i=d1, D_j=d1) (n x n)
        pi_ij_d2d2: joint probability P(D_i=d2, D_j=d2) (n x n)
        d1, d2: exposure levels being compared
        eps: small constant for numerical stability
    
    Returns:
        L1: Lambda matrix for exposure d1 (n x n)
        L2: Lambda matrix for exposure d2 (n x n)
        L12: Cross Lambda matrix (n x n)
    """
    pi1 = pi_id_all[:, d1]
    pi2 = pi_id_all[:, d2]

    # den11_{i,j} = pi_i(d1) * pi_j(d1)
    den11 = np.outer(pi1, pi1)
    # L1 off-diagonal items
    L1 = (den11 > eps) * ((pi_ij_d1d1 - den11) / (den11 + eps))
    np.fill_diagonal(L1, (pi1 > eps) * (1.0 - pi1) / (pi1 + eps))
    
    den22 = np.outer(pi2, pi2)
    L2 = (den22 > eps) * ((pi_ij_d2d2 - den22) / (den22 + eps))
    np.fill_diagonal(L2, (pi2 > eps) * (1.0 - pi2) / (pi2 + eps))
    
    den12 = np.outer(pi1, pi2)
    L12 = (den12 > eps) * ((pi_ij_d1d2 - den12) / (den12 + eps))
    np.fill_diagonal(L12, -1.0)
    
    return L1, L2, L12


def build_kn_matrix(G):
    """
    Build the Kn matrix: indicator matrix where (i,j) = 1 if nodes i and j 
    have common neighbors (used in regression-based estimators).
    
    Args:
        G: adjacency matrix (n x n)
    
    Returns:
        Kn: indicator matrix (n x n)
    """
    return (G @ G.T > 0).astype(int)

