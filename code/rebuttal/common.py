"""Shared helpers for the AISTATS-2027 rebuttal experiments.

Everything reuses the paper's code in ../synthetic (DGP, exposure, propensities, PNA, variance estimator).
`ger_solve` re-implements LinearGRR(reg_type='ols').fit so the truncation level can vary; at rcond=1e-2
it reproduces grr_estimator(method='linear') exactly (checked by `check_ger_solve`).
"""
import importlib.util
import os
import sys

import numpy as np
from sklearn.linear_model import LinearRegression

HERE = os.path.dirname(os.path.abspath(__file__))
SYN = os.path.join(os.path.dirname(HERE), "synthetic")
sys.path.insert(0, SYN)

_spec = importlib.util.spec_from_file_location("syn_out2", os.path.join(SYN, "dgp-2", "simulation-out2-dgp2-parallel.py"))
syn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(syn)

from GRR import prepare_grr_inputs, compute_grr_tau, grr_estimator, PNA_embedding  # noqa: E402
from variance_estimators import grr_confidence_interval  # noqa: E402

EPS = 1e-12
SETTINGS = {"out1": (3.0, 9), "out2": (5.0, 10)}  # Table 1 and Table 6 networks (avg, max degree)


def population(setting):
    """Fixed finite population of the paper's synthetic studies (same seeds as the drivers)."""
    avg, mx = SETTINGS[setting]
    G, H = syn.generate_graph(n=syn.N, avg_degree=avg, max_degree=mx, seed=syn.SEED_GRAPH)
    X, X_noi = syn.generate_covariates(H, seed=syn.SEED_COV)
    Y_pot = syn.generate_potential_outcomes(G, H, X_noi, k=syn.K, beta_neighbor=1.0, seed=syn.SEED_OUT)
    return G, H, X[:, [0, 3]], Y_pot


def design(G, k, d1, d2, mc_reps, seed, exposure_fn, sampler):
    """Monte Carlo pi's, Lambda blocks and W_pi for an arbitrary assignment `sampler(rng, n)`."""
    rng = np.random.default_rng(seed)
    n = G.shape[0]
    M = np.zeros((k, n, mc_reps), dtype=np.float32)
    for r in range(mc_reps):
        D = exposure_fn(G, sampler(rng, n), k)
        M[D, np.arange(n), r] = 1.0
    pi = M.sum(2).T / mc_reps
    p12, p11, p22 = (M[a] @ M[b].T / mc_reps for a, b in ((d1, d2), (d1, d1), (d2, d2)))
    for p in (p12, p11, p22):
        np.fill_diagonal(p, 0.0)
    Kn = (G @ G.T > 0).astype(int)
    L = syn.build_lambdas(pi, p12.astype(float), p11.astype(float), p22.astype(float), d1, d2, EPS)
    W = syn.build_W_pi(pi, p11.astype(float), p22.astype(float), d1, d2, Kn)
    return pi, Kn, L, W


def complete_randomization(p):
    return lambda rng, n: _complete(rng, n, p)


def _complete(rng, n, p):
    A = np.zeros(n, dtype=np.int8)
    A[rng.choice(n, size=int(round(p * n)), replace=False)] = 1
    return A


def bernoulli(p):
    return lambda rng, n: (rng.random(n) < p).astype(np.int8)


def _ci(tau, inp, X1, X2, b1, b2, W):
    lo, hi = grr_confidence_interval(tau_hat=tau, hatY_d1=inp["hatY_d1"], hatY_d2=inp["hatY_d2"],
                                     hat1_d1=inp["hat1_d1"], hat1_d2=inp["hat1_d2"], X1=X1, X2=X2,
                                     beta1=b1, beta2=b2, W1_pi=W[0], W2_pi=W[1], alpha=0.05)
    return lo, hi


def aipw_fixed(Y, D, d1, d2, pi, W, f1, f2):
    """AIPW with fixed predictions (coefficient 1, no GER): the paper's '-raw' rows."""
    X1, X2 = f1.reshape(-1, 1), f2.reshape(-1, 1)
    inp = prepare_grr_inputs(Y, D, X1, d1, d2, pi, X1=X1, X2=X2, eps=EPS)
    tau = compute_grr_tau(f1, f2, inp["hat1_d1"], inp["hat1_d2"], inp["hatY_d1"], inp["hatY_d2"])[0]
    return (tau, *_ci(tau, inp, X1, X2, np.ones(1), np.ones(1), W), np.nan)


def hajek(Y, D, d1, d2, pi, W):
    """Hajek estimate and CI exactly as in the drivers (constant feature, beta = Hajek means)."""
    m1, m2 = syn.mu_haj(Y, D, d1, pi, EPS), syn.mu_haj(Y, D, d2, pi, EPS)
    ones = np.ones((len(Y), 1))
    inp = prepare_grr_inputs(Y, D, ones, d1, d2, pi, X1=ones, X2=ones, eps=EPS)
    tau = float(m1 - m2)
    return (tau, *_ci(tau, inp, ones, ones, np.array([m1]), np.array([m2]), W), np.nan)


def ger_solve(Y, D, d1, d2, pi, Kn, W, X1, X2, rcond=1e-2, ridge=0.0):
    """GER estimating equation solved by truncated least squares (or relative ridge); returns (tau, lo, hi, cond)."""
    inp = prepare_grr_inputs(Y, D, X1, d1, d2, pi, X1=X1, X2=X2, eps=EPS)
    Z = np.hstack([X1 * inp["tilde1_d1"][:, None], -X2 * inp["tilde1_d2"][:, None]])
    ZtKn = Z.T @ Kn
    A, b = ZtKn @ Z, ZtKn @ inp["dHY"]
    cond = np.linalg.cond(A)
    if ridge > 0:
        theta = np.linalg.solve(A + ridge * np.trace(A) / A.shape[0] * np.eye(A.shape[0]), b)
    else:
        theta = np.linalg.lstsq(A, b, rcond=rcond)[0]
    b1, b2 = theta[:X1.shape[1]], theta[X1.shape[1]:]
    tau = compute_grr_tau(X1 @ b1, X2 @ b2, inp["hat1_d1"], inp["hat1_d2"], inp["hatY_d1"], inp["hatY_d2"])[0]
    return (tau, *_ci(tau, inp, X1, X2, b1, b2, W), cond)


def per_exposure_ols(Y, D, X, d):
    """Per-exposure OLS predictions for all units (the paper's linear-raw fit)."""
    m = D == d
    if m.sum() > X.shape[1]:
        return X @ LinearRegression(fit_intercept=False).fit(X[m], Y[m]).coef_
    return np.full(len(Y), Y[m].mean() if m.any() else 0.0)


def train_pna(X, A_adj, D, Y, k, d1, d2, seed, config, predict_on=None):
    """Seeded PNA fit; returns (pred_d1, pred_d2, leaf_d1, leaf_d2), optionally evaluated on another graph."""
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    out = PNA_embedding(X, A_adj, D, Y, k=k, d1=d1, d2=d2, verbose=False, return_model=True, **config)
    model = out["model"]
    if predict_on is None:
        return model.predict(X, D=d1), model.predict(X, D=d2), out["X_PNA_leaf"][0], out["X_PNA_leaf"][1]
    Xf, Af = predict_on
    return model.predict(Xf, A=Af, D=d1), model.predict(Xf, A=Af, D=d2), None, None


def check_ger_solve():
    """ger_solve(rcond=1e-2) must equal the production grr_estimator(method='linear')."""
    G, H, X_run, Y_pot = population("out1")
    pi, Kn, L, W = design(G, 3, 0, 2, 200, 111, syn.compute_exposure, complete_randomization(1 / 3))
    rng = np.random.default_rng(0)
    D = syn.compute_exposure(G, _complete(rng, G.shape[0], 1 / 3), 3)
    Y = Y_pot[np.arange(len(D)), D]
    ref = grr_estimator(Y, D, X_run, 0, 2, pi, Kn, method="linear")["tau_hat"]
    mine = ger_solve(Y, D, 0, 2, pi, Kn, W, X_run, X_run)[0]
    assert abs(ref - mine) < 1e-10 * max(1, abs(ref)), (ref, mine)
    return ref, mine


if __name__ == "__main__":
    print("ger_solve matches grr_estimator:", check_ger_solve())
