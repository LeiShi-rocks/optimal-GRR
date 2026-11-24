"""Rebuttal experiment 2 (CPU): positivity and coverage as hubs appear (AC, XK3s, AnSZ).

Chung-Lu graphs (n = 2000, mean degree 3, power-law weights with exponent 2.5) whose expected degrees are
capped at `cap`; larger caps create hubs. Bernoulli(1/3) assignment, as in the reviewer's example. Two
exposure mappings, both on the self-inclusive neighbourhood:
  * fraction: share of treated units cut into 3 equal bins (the paper's mapping), contrast d=0 vs d=2;
  * any: 1{at least one treated unit}, contrast d=0 (none treated) vs d=1.
First-order probabilities are exact (binomial); joint probabilities come from Monte Carlo.
Outcomes follow the paper's DGP, with degree truncated at 10 inside the outcome model so that outcomes stay
bounded (Assumption C.1(c)) and only positivity and dependence change with `cap`.
Estimators: Hajek and the linear-backbone rows of Tables 1/6 (linear-raw, linear-linear, linear-pred).
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import binom

from common import (EPS, aipw_fixed, bernoulli, ger_solve, hajek, per_exposure_ols, syn)

P = 1 / 3
N = 2000
CAPS = [10, 30, 100, 300]


def chung_lu(n, mean_deg, cap, gamma=2.5, seed=0):
    rng = np.random.default_rng(seed)
    w = (np.arange(1, n + 1) / n) ** (-1 / (gamma - 1))
    w = np.minimum(w * mean_deg / w.mean(), cap)
    w *= mean_deg / w.mean()
    w = np.minimum(w, cap)
    prob = np.minimum(1.0, np.outer(w, w) / w.sum())
    U = np.triu(rng.random((n, n)) < prob, 1)
    G = (U | U.T).astype(np.int8)
    np.fill_diagonal(G, 1)
    return G


def exposure_fraction(G, A, k=3):
    return syn.compute_exposure(G, A, k)


def exposure_any(G, A, k=2):
    return (G @ A > 0).astype(int)


def exact_pi(G, mapping):
    H = G.sum(1).astype(int)
    if mapping == "any":
        p0 = (1 - P) ** H
        return np.column_stack([p0, 1 - p0])
    pi = np.zeros((len(H), 3))
    for i, h in enumerate(H):
        t = np.arange(h + 1)
        bins = np.clip(np.floor(3 * t / h).astype(int), 0, 2)
        np.add.at(pi[i], bins, binom.pmf(t, h, P))
    return pi


def joint_mc(G, mapping, k, d1, d2, reps, seed):
    rng = np.random.default_rng(seed)
    n = G.shape[0]
    f = exposure_any if mapping == "any" else exposure_fraction
    M1 = np.zeros((n, reps), np.float32)
    M2 = np.zeros((n, reps), np.float32)
    for r in range(reps):
        D = f(G, (rng.random(n) < P).astype(np.int8), k)
        M1[:, r], M2[:, r] = D == d1, D == d2
    out = [M1 @ M2.T / reps, M1 @ M1.T / reps, M2 @ M2.T / reps]
    for p in out:
        np.fill_diagonal(p, 0.0)
    return [p.astype(float) for p in out]


def run_draw(b, G, X, Y_pot, mapping, k, d1, d2, pi, Kn, W, seed_base):
    rng = np.random.default_rng(seed_base + b)
    n = G.shape[0]
    f = exposure_any if mapping == "any" else exposure_fraction
    D = f(G, bernoulli(P)(rng, n), k)
    Y = Y_pot[np.arange(n), D]
    f1, f2 = per_exposure_ols(Y, D, X, d1), per_exposure_ols(Y, D, X, d2)
    res = {"hajek": hajek(Y, D, d1, d2, pi, W),
           "linear-raw": aipw_fixed(Y, D, d1, d2, pi, W, f1, f2),
           "linear-linear": ger_solve(Y, D, d1, d2, pi, Kn, W, X, X),
           "linear-pred": ger_solve(Y, D, d1, d2, pi, Kn, W, f1[:, None], f2[:, None])}
    return [{"draw": b, "method": m, "tau": r[0], "lo": r[1], "hi": r[2], "cond": r[3]} for m, r in res.items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--mc-reps", type=int, default=5000)
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--seed-base", type=int, default=8_000_000)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows, graphs = [], []
    for cap in CAPS:
        G = chung_lu(N, 3.0, cap, seed=cap)
        H = G.sum(1)
        Hc = np.minimum(H, 10)  # bounded outcomes; degree enters the outcome model only up to 10
        X, X_noi = syn.generate_covariates(Hc, seed=syn.SEED_COV)
        X_run = X[:, [0, 3]]
        Kn = (G @ G.T > 0).astype(int)
        for mapping, k, d1, d2 in (("fraction", 3, 0, 2), ("any", 2, 0, 1)):
            t0 = time.time()
            Y_pot = syn.generate_potential_outcomes(G, Hc, X_noi, k=k, beta_neighbor=1.0, seed=syn.SEED_OUT)
            true_tau = float(np.mean(Y_pot[:, d1] - Y_pot[:, d2]))
            pi = exact_pi(G, mapping)
            p12, p11, p22 = joint_mc(G, mapping, k, d1, d2, args.mc_reps, seed=cap)
            L = syn.build_lambdas(pi, p12, p11, p22, d1, d2, EPS)
            W = syn.build_W_pi(pi, p11, p22, d1, d2, Kn)
            out = Parallel(n_jobs=args.n_jobs)(delayed(run_draw)(b, G, X_run, Y_pot, mapping, k, d1, d2, pi, Kn, W,
                                                                 args.seed_base) for b in range(args.B))
            for r in (r for rs in out for r in rs):
                rows.append({**r, "cap": cap, "mapping": mapping, "true_tau": true_tau})
            pmin = np.minimum(pi[:, d1], pi[:, d2])
            graphs.append({"cap": cap, "mapping": mapping, "max_deg": int(H.max() - 1), "mean_deg": float(H.mean() - 1),
                           "max_deg_Delta": int(Kn.sum(1).max() - 1), "min_pi": float(pmin.min()),
                           "share_pi_below_0.01": float(np.mean(pmin < 0.01)), "true_tau": true_tau})
            print(graphs[-1], f"{time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "exp2_degree.csv"), index=False)
    pd.DataFrame(graphs).to_csv(os.path.join(args.out, "exp2_degree_graphs.csv"), index=False)
    print("saved exp2_degree.csv", flush=True)


if __name__ == "__main__":
    main()
