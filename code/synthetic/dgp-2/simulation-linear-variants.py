"""Linear-backbone variants on shared draws, adding `linear-no-harm`: GER on linear predictions (Paradigm 1).

`linear-no-harm` mirrors `gat-no-harm`/`pna-no-harm`: the per-exposure OLS predictions of `linear-no-cal`
become the single GER feature. All other methods come unchanged from the original drivers.

Usage (from code/synthetic):
    python dgp-2/simulation-linear-variants.py --setting out1   # Table 1 network (avg/max degree 3/9)
    python dgp-2/simulation-linear-variants.py --setting out2   # Table 6 network (avg/max degree 5/10)
"""
import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

_spec = importlib.util.spec_from_file_location("syn_out2", os.path.join(HERE, "simulation-out2-dgp2-parallel.py"))
syn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(syn)

METHODS = ["hajek", "linear-no-cal", "linear", "lasso", "linear-no-harm"]
SETTINGS = {"out1": (3.0, 9), "out2": (5.0, 10)}


def per_exposure_ols(Y, D, X, d):
    """OLS of Y on [1, X] among units with D == d, predicted for all units (as in `linear-no-cal`)."""
    Xi = np.column_stack([np.ones(len(Y)), X])
    m = D == d
    if m.sum() > Xi.shape[1]:
        return Xi @ LinearRegression(fit_intercept=False).fit(Xi[m], Y[m]).coef_
    return np.full(len(Y), Y[m].mean() if m.any() else 0.0)


def linear_no_harm(Y, D, X_run, d1, d2, pi_id_all, Kn, eps):
    X1 = per_exposure_ols(Y, D, X_run, d1).reshape(-1, 1)
    X2 = per_exposure_ols(Y, D, X_run, d2).reshape(-1, 1)
    res = syn.grr_estimator(Y, D, X_run, d1, d2, pi_id_all, Kn, method="linear", eps=eps, X1=X1, X2=X2)
    inputs = syn.prepare_grr_inputs(Y, D, X_run, d1, d2, pi_id_all, X1=X1, X2=X2, eps=eps)
    beta1, beta2 = res["model"].get_coefficients()
    var_info = {k: inputs[k] for k in ("hatY_d1", "hatY_d2", "hat1_d1", "hat1_d2")}
    var_info.update(X1=X1, X2=X2, beta1=beta1, beta2=beta2)
    return res["tau_hat"], True, var_info


def run_rep(b, G, X_run, Y_pot, pi, Kn, L, W, true_tau, seed_base):
    rng = np.random.default_rng(seed_base + b)
    n = G.shape[0]
    A = np.zeros(n, dtype=np.int8)
    A[rng.choice(n, size=int(round(syn.TREATED_FRACTION * n)), replace=False)] = 1  # complete randomization, as in dgp-2
    D = syn.compute_exposure(G, A, syn.K)
    Y = syn.observed_outcome(Y_pot, D, k=syn.K)
    d1, d2 = syn.D1, syn.D2
    est, cov, pw = {}, {}, {}
    for m in METHODS:
        if m == "linear-no-harm":
            tau, has_var, v = linear_no_harm(Y, D, X_run, d1, d2, pi, Kn, syn.EPS)
        else:
            tau, has_var, v = syn._run_method(m, Y, D, X_run, d1, d2, n, pi[:, d1], pi[:, d2], pi, Kn, syn.EPS,
                                              *L, None, None, None, None, {"alpha": 0.01, "l1_ratio": 0.5})
        est[m] = tau
        lo, hi = syn.grr_confidence_interval(tau_hat=tau, W1_pi=W[0], W2_pi=W[1], alpha=0.05, **v)
        cov[m], pw[m] = float(lo <= true_tau <= hi), float(lo > 0 or hi < 0)
    return b, est, cov, pw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=SETTINGS, default="out1")
    ap.add_argument("--B", type=int, default=100)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--seed-base", type=int, default=20261123)
    args = ap.parse_args()

    avg_deg, max_deg = SETTINGS[args.setting]
    G, H = syn.generate_graph(n=syn.N, avg_degree=avg_deg, max_degree=max_deg, seed=syn.SEED_GRAPH)
    X, X_noi = syn.generate_covariates(H, seed=syn.SEED_COV)
    X_run = X[:, [0, 3]]
    Y_pot = syn.generate_potential_outcomes(G, H, X_noi, k=syn.K, beta_neighbor=1.0, seed=syn.SEED_OUT)
    true_tau = float(np.mean(Y_pot[:, syn.D1] - Y_pot[:, syn.D2]))
    pi, p12, p11, p22 = syn.monte_carlo_probabilities(G, syn.TREATED_FRACTION, syn.K, syn.MC_REPS, syn.SEED_MC,
                                                      syn.D1, syn.D2)
    Kn = (G @ G.T > 0).astype(int)
    L = syn.build_lambdas(pi, p12, p11, p22, syn.D1, syn.D2, syn.EPS)
    W = syn.build_W_pi(pi, p11, p22, syn.D1, syn.D2, Kn)

    out = Parallel(n_jobs=args.n_jobs)(delayed(run_rep)(b, G, X_run, Y_pot, pi, Kn, L, W, true_tau, args.seed_base)
                                       for b in range(args.B))
    rows = []
    for m in METHODS:
        v = np.array([o[1][m] for o in out])
        bias, sd = float(v.mean() - true_tau), float(v.std(ddof=1))
        rows.append({"method": m, "bias": bias, "sd": sd, "rmse": float(np.hypot(bias, sd)),
                     "coverage": float(np.mean([o[2][m] for o in out])), "power": float(np.mean([o[3][m] for o in out]))})
    summary = pd.DataFrame(rows)
    print(f"setting={args.setting}, true_tau={true_tau:.4f}, B={args.B}")
    print(summary.to_string(index=False))

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "results", "synthetic", "dgp2")
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, f"{args.setting}_linear_variants_{datetime.now():%Y%m%d_%H%M%S}")
    summary.to_csv(f"{prefix}_summary.csv", index=False)
    np.savez(f"{prefix}_full.npz", **{f"results_{m}": np.array([o[1][m] for o in out]) for m in METHODS})
    with open(f"{prefix}_config.json", "w") as f:
        json.dump({**vars(args), "true_tau": true_tau, "avg_degree": avg_deg, "max_degree": max_deg}, f, indent=2)


if __name__ == "__main__":
    main()
