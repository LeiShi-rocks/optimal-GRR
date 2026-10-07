"""
Data Generating Process (DGP) module for HAIPW simulation.

This module contains functions for:
- Generating network graphs with self-loops
- Generating covariates based on degree
- Generating potential outcomes
- Assigning treatment
- Computing exposure based on network structure
- Extracting observed outcomes
"""

import numpy as np


# ---------- Step 1: Graph (min degree ≥ 2, with self-loops) ----------
def generate_graph(n=200, avg_degree=3.0, max_degree=9, seed=123):
    """
    Simple undirected graph with:
      • WITH self-loops (diagonal elements = 1),
      • degree_i <= max_degree + 1 for all i (including self-loop),
      • degree_i >= 2 for all i (at least one neighbor + self-loop).

    Note: If max_degree == 1 and n is odd, this is impossible (a 1-regular graph
    exists only for even n). We raise a ValueError in that case.
    
    Args:
        n: number of nodes
        avg_degree: target average degree (before self-loops)
        max_degree: maximum degree per node (before self-loops)
        seed: random seed
    
    Returns:
        G: adjacency matrix (n x n) with self-loops
        deg: degree vector (n,) including self-loops
    """
    if n < 2:
        raise ValueError("n must be >= 2 to give each unit at least one connection without self-loops.")
    if max_degree < 1:
        raise ValueError("max_degree must be >= 1 to ensure at least one connection per unit.")
    if max_degree == 1 and (n % 2 == 1):
        raise ValueError("Impossible: max_degree=1 with odd n cannot give everyone degree ≥ 1.")

    rng = np.random.default_rng(seed)

    # target edges for desired average degree, but at least ceil(n/2) to support min degree ≥ 1
    target_edges = int(n * avg_degree / 2.0)
    min_edges = (n + 1) // 2  # ceil(n/2)
    target_edges = max(target_edges, min_edges)

    G = np.zeros((n, n), dtype=np.int8)
    deg = np.zeros(n, dtype=np.int32)

    # --- Phase A: give every node at least one neighbor (random matching) ---
    perm = rng.permutation(n)
    # pair consecutive nodes
    for i in range(0, n - 1, 2):
        u, v = perm[i], perm[i + 1]
        if G[u, v] == 0:
            G[u, v] = 1
            G[v, u] = 1
            deg[u] += 1
            deg[v] += 1

    # if n is odd, connect the leftover node to someone with spare degree
    if n % 2 == 1:
        w = perm[-1]
        # partner must be != w, not already connected, and have spare degree
        candidates = np.where((np.arange(n) != w) & (G[w] == 0) & (deg < max_degree))[0]
        if candidates.size == 0:
            raise RuntimeError("Cannot ensure degree ≥ 1 under the current max_degree cap.")
        v = rng.choice(candidates)
        G[w, v] = 1
        G[v, w] = 1
        deg[w] += 1
        deg[v] += 1

    # --- Phase B: add random edges up to target_edges while respecting caps ---
    added = int(G.sum() // 2)
    attempts, max_attempts = 0, 20 * max(target_edges, 1) + 1000
    while added < target_edges and attempts < max_attempts:
        i, j = rng.integers(0, n, size=2)
        if i == j or G[i, j] == 1:
            attempts += 1
            continue
        if deg[i] >= max_degree or deg[j] >= max_degree:
            attempts += 1
            continue
        G[i, j] = 1
        G[j, i] = 1
        deg[i] += 1
        deg[j] += 1
        added += 1

    # --- Final safety checks ---
    if np.any(deg == 0):
        # Try one last greedy patch (should not happen with Phase A)
        zeros = np.where(deg == 0)[0]
        for u in zeros:
            cand = np.where((np.arange(n) != u) & (G[u] == 0) & (deg < max_degree))[0]
            if cand.size == 0:
                raise RuntimeError("Could not ensure degree ≥ 1 due to max_degree cap.")
            v = rng.choice(cand)
            G[u, v] = 1
            G[v, u] = 1
            deg[u] += 1
            deg[v] += 1

    # --- Add self-loops ---
    np.fill_diagonal(G, 1)  # add self-loops
    deg += 1  # each node's degree increases by 1 due to self-loop

    return G, deg


# ---------- Step 2: Covariates ----------
def generate_covariates(H, a1=0.5, a2=0.02, sigma1=1.0, sigma2=1.0,
                        use_poisson_x3=True, sigma3=1.0, seed=1234):
    """
    Generate covariates based on degree.
    
    X1 ~ N(a1*H, sigma1^2), X2 ~ N(a2*H^2, sigma2^2),
    X3 ~ Poisson(1{H>np.mean(H)}) by default; else Normal(mean=1{H>8}, sd=sigma3).
    
    Args:
        H: degree vector (n,)
        a1, a2: coefficients for X1 and X2 means
        sigma1, sigma2, sigma3: standard deviations
        use_poisson_x3: if True, X3 ~ Poisson; else Normal
        seed: random seed
    
    Returns:
        X: covariate matrix with intercept (n x 4) = [1, X1, X2, X3]
        X_no_intercept: covariate matrix without intercept (n x 3)
    """
    rng = np.random.default_rng(seed)
    H = np.asarray(H, float)
    n = H.size

    X1 = rng.normal(loc=a1 * H, scale=sigma1, size=n)
    X2 = rng.normal(loc=a2 * (H ** 2), scale=sigma2, size=n)
    if use_poisson_x3:
        lam = (H >= np.mean(H)).astype(float)
        X3 = rng.poisson(lam=lam, size=n).astype(float)
    else:
        mu3 = (H >= np.mean(H)).astype(float)
        X3 = rng.normal(loc=mu3, scale=sigma3, size=n)

    X = np.column_stack([np.ones(n), X1, X2, X3])
    return X, X[:, 1:]


# ---------- Step 3: Potential outcomes ----------
def g_nonlinear(X_no_intercept, gamma1=0.6, gamma2=0.15, gamma3=0.4):
    """
    Nonlinear function of covariates for potential outcomes.
    
    Args:
        X_no_intercept: covariate matrix without intercept (n x 3)
        gamma1, gamma2, gamma3: coefficients
    
    Returns:
        g(X): nonlinear transformation (n,)
    """
    X1 = X_no_intercept[:, 0]
    X2 = X_no_intercept[:, 1]
    X3 = X_no_intercept[:, 2]
    return gamma1 * X1 + gamma2 * (X2 ** 2) + gamma3 * np.tanh(X3)


def generate_potential_outcomes(H, X_no_intercept, k=3, beta0=0.0, beta_h=0.5, beta_d=0.3,
                                noise_sd=1.0, seed=999):
    """
    Generate potential outcomes for each exposure category.
    For d=0..k-1: Y_i(d) ~ Normal(beta0 + beta_h H_i + beta_d d + g(X_i), noise_sd^2).
    
    Args:
        H: degree vector (n,)
        X_no_intercept: covariates without intercept (n x p)
        k: number of exposure categories (default=3)
        beta0, beta_h, beta_d: coefficients
        noise_sd: standard deviation of noise
        seed: random seed
    
    Returns:
        Y_pot: potential outcomes matrix of shape (n, k)
    """
    rng = np.random.default_rng(seed)
    H = np.asarray(H, float)
    n = H.size
    gX = g_nonlinear(X_no_intercept)
    Y = np.zeros((n, k), float)
    for d in range(k):
        mean = beta0 + beta_h * H + beta_d * d + gX
        Y[:, d] = rng.normal(loc=mean, scale=noise_sd, size=n)
    return Y


# ---------- Helpers: treatment / exposure / observed ----------
def assign_treatment(n, treated_fraction=1/3, seed=None):
    """
    Randomly assign treatment to units.
    
    Args:
        n: number of units
        treated_fraction: fraction of units to treat
        seed: random seed (None for random)
    
    Returns:
        A: treatment assignment vector (n,) with values in {0, 1}
    """
    rng = np.random.default_rng(seed)
    A = np.zeros(n, dtype=np.int8)
    treated_count = int(round(treated_fraction * n))
    A[rng.choice(n, size=treated_count, replace=False)] = 1
    return A


def compute_exposure(G, A, k=3):
    """
    Compute categorical exposure based on the proportion of treated neighbors.
    
    For each node i:
    - proportion_i = (number of treated neighbors including self) / (total degree including self)
    - d_i = categorical variable in {0, 1, ..., k-1} indicating which bin:
        0 if proportion in [0, 1/k)
        1 if proportion in [1/k, 2/k)
        ...
        k-1 if proportion in [(k-1)/k, 1]
    
    Args:
        G: adjacency matrix (n x n) with self-loops (diagonal = 1)
        A: treatment assignment vector (n,)
        k: number of bins (default=3)
    
    Returns:
        D: categorical exposure vector (n,) with values in {0, 1, ..., k-1}
    """
    # Number of treated neighbors (including self) for each node
    num_treated_neighbors = G @ A  # shape (n,)
    
    # Total degree (including self) for each node
    degree = G.sum(axis=1)  # shape (n,)
    
    # Proportion of treated neighbors
    proportion = num_treated_neighbors / degree  # shape (n,)
    
    # Discretize into k bins
    # bin i corresponds to proportion in [i/k, (i+1)/k)
    # Special case: proportion = 1.0 should be in bin k-1
    D = np.floor(proportion * k).astype(int)
    D = np.clip(D, 0, k - 1)  # Ensure D is in {0, 1, ..., k-1}
    
    return D


def observed_outcome(Y_pot, D, k=3):
    """
    Extract observed outcomes based on categorical exposure D.
    
    Args:
        Y_pot: potential outcomes matrix (n x k) where Y_pot[i, d] is outcome for unit i under exposure d
        D: observed exposure vector (n,) with values in {0, 1, ..., k-1}
        k: number of exposure categories
    
    Returns:
        Y: observed outcomes (n,)
    """
    if np.any(D >= k):
        raise ValueError(f"Exposure has values >= k={k}, expected values in {{0, ..., {k-1}}}")
    return Y_pot[np.arange(Y_pot.shape[0]), D]

