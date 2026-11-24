"""Student-network semi-synthetic study (paper Section 6, Table 2), rebuilt.

Reuses the synthetic-study estimators verbatim: `_run_method` and the GNN/GER code are imported from
../synthetic. Differences from the synthetic drivers: i.i.d. Bernoulli(1/3) assignment (as the paper
states for this study) and per-replication seeds for NumPy and PyTorch.

Usage (from code/semisynthetic):
    python run_semisynthetic.py                 # B = 200, all methods
    python run_semisynthetic.py --B 2 --methods ht hajek linear --mc-reps 50   # smoke test
"""
import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
SYN = HERE.parent / "synthetic"
sys.path.insert(0, str(SYN))
sys.path.insert(0, str(HERE))

from dgp import compute_exposure  # noqa: E402
from propensity import build_lambdas, build_W_pi  # noqa: E402
from GRR import GAT_embedding, PNA_embedding  # noqa: E402
from variance_estimators import grr_confidence_interval  # noqa: E402
from semisynthetic_data import (  # noqa: E402
    build_student_network, generate_potential_outcomes, adjustment_covariates, bernoulli_mc_probabilities,
)

# Method implementations and GNN settings are shared with the medium-density synthetic driver.
_spec = importlib.util.spec_from_file_location("syn_out2", SYN / "dgp-2" / "simulation-out2-dgp2-parallel.py")
_syn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_syn)
_run_method, GAT_CONFIG, PNA_CONFIG = _syn._run_method, _syn.GAT_CONFIG, _syn.PNA_CONFIG

ALL_METHODS = ["ht", "hajek", "linear", "lasso", "linear-no-cal",
               "gat-linear", "gat-no-harm", "gat-pred", "pna-linear", "pna-no-harm", "pna-pred"]
K, D1, D2, P_TREAT, EPS = 3, 0, 2, 1 / 3, 1e-12


def run_replication(b, G, X_run, Y_pot, pi, Kn, L, W, true_tau, methods, seed_base):
    import torch
    rng = np.random.default_rng(seed_base + b)
    torch.manual_seed(seed_base + b)
    n = G.shape[0]
    D = compute_exposure(G, (rng.random(n) < P_TREAT).astype(np.int8), K)
    Y = Y_pot[np.arange(n), D]
    A_adj = G - np.eye(n)

    gat_emb = gat_model = pna_emb = pna_model = None
    if any(m.startswith("gat") for m in methods):
        gat_emb = GAT_embedding(X_run, A_adj, D, Y, k=K, d1=D1, d2=D2, verbose=False, return_model=True, **GAT_CONFIG)
        gat_model = gat_emb.get("model")
    if any(m.startswith("pna") for m in methods):
        pna_emb = PNA_embedding(X_run, A_adj, D, Y, k=K, d1=D1, d2=D2, verbose=False, return_model=True, **PNA_CONFIG)
        pna_model = pna_emb.get("model")

    est, cov, pw = {}, {}, {}
    for m in methods:
        try:
            tau_hat, has_var, v = _run_method(m, Y, D, X_run, D1, D2, n, pi[:, D1], pi[:, D2], pi, Kn, EPS,
                                              *L, gat_emb, pna_emb, gat_model, pna_model,
                                              {"alpha": 0.01, "l1_ratio": 0.5})
            est[m] = tau_hat
            if has_var:
                lo, hi = grr_confidence_interval(tau_hat=tau_hat, W1_pi=W[0], W2_pi=W[1], alpha=0.05, **v)
                cov[m], pw[m] = float(lo <= true_tau <= hi), float(lo > 0 or hi < 0)
        except Exception as e:  # keep other methods running; NaN is reported via n_valid
            print(f"rep {b} {m}: {e}")
    return b, est, cov, pw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS)
    ap.add_argument("--mc-reps", type=int, default=1000)
    ap.add_argument("--n-jobs", type=int, default=3)
    ap.add_argument("--seed-outcome", type=int, default=999)
    ap.add_argument("--seed-mc", type=int, default=111)
    ap.add_argument("--seed-base", type=int, default=20260000)
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()

    start = datetime.now()
    net = build_student_network()
    G = net["G"]
    Y_pot = generate_potential_outcomes(net, k=K, seed=args.seed_outcome)
    true_tau = float(np.mean(Y_pot[:, D2] - Y_pot[:, D1]))  # paper: E[Y(2)] - E[Y(0)]
    X_run = adjustment_covariates(net)
    print(f"n={G.shape[0]}, mean self-inclusive degree={G.sum(1).mean():.2f}, tau*={true_tau:.4f}")

    pi, p12, p11, p22 = bernoulli_mc_probabilities(G, P_TREAT, K, args.mc_reps, args.seed_mc, D1, D2, compute_exposure)
    Kn = (G @ G.T > 0).astype(int)
    L = build_lambdas(pi, p12, p11, p22, D1, D2, EPS)
    W = build_W_pi(pi, p11, p22, D1, D2, Kn)

    # The shared estimators target mu(d1) - mu(d2); report tau* = mu(2) - mu(0) by flipping sign below.
    out = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(run_replication)(b, G, X_run, Y_pot, pi, Kn, L, W, -true_tau, args.methods, args.seed_base)
        for b in range(args.B))

    res = {m: np.full(args.B, np.nan) for m in args.methods}
    cov = {m: np.full(args.B, np.nan) for m in args.methods}
    pw = {m: np.full(args.B, np.nan) for m in args.methods}
    for b, e, c, p in out:
        for m in args.methods:
            res[m][b] = -e.get(m, np.nan)
            cov[m][b], pw[m][b] = c.get(m, np.nan), p.get(m, np.nan)

    rows = []
    for m in args.methods:
        v = res[m][~np.isnan(res[m])]
        bias, sd = (float(v.mean() - true_tau), float(v.std(ddof=1))) if len(v) > 1 else (np.nan, np.nan)
        rows.append({"method": m, "bias": bias, "sd": sd, "rmse": float(np.hypot(bias, sd)), "n_valid": len(v),
                     "coverage": float(np.nanmean(cov[m])) if np.isfinite(cov[m]).any() else np.nan,
                     "power": float(np.nanmean(pw[m])) if np.isfinite(pw[m]).any() else np.nan})
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))

    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{args.out}/semisyn_{stamp}"
    summary.to_csv(f"{prefix}_summary.csv", index=False)
    np.savez(f"{prefix}_full.npz", **{f"results_{m}": res[m] for m in args.methods},
             **{f"coverage_{m}": cov[m] for m in args.methods}, **{f"power_{m}": pw[m] for m in args.methods})
    with open(f"{prefix}_config.json", "w") as f:
        json.dump({**vars(args), "true_tau": true_tau, "n": int(G.shape[0]), "gat_config": GAT_CONFIG,
                   "pna_config": PNA_CONFIG, "elapsed_seconds": (datetime.now() - start).total_seconds()}, f, indent=2)
    print(f"saved {prefix}_*")


if __name__ == "__main__":
    main()
