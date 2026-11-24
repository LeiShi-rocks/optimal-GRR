"""Rebuttal experiment 5: Lu, Wang and Zhang (2024, arXiv:2411.19789) network-dependent regression adjustment.

Their estimator (Sections 3.3 and 4.2), with Hajek weights w_i = h1_i/mean(h1) - h2_i/mean(h2),
h_k = 1{T_i = t_k}/pi_i(t_k):
    tau(beta) = n^-1 sum_i (Y_i - phi_i' beta) w_i,
    V_i(beta) = (Y_i - phi_i'beta - mu_hat(t))h1_i - (Y_i - phi_i'beta - mu_hat(t'))h2_i   (Hajek means),
    sigma2(beta) = n^-1 sum_ij B_ij V_i V_j,  B_ij = 1{dist(i,j) < b_n}, b_n = 3 (their choice),
    beta_hat = argmin sigma2(beta);  CI = tau(beta_hat) -/+ 1.96 sqrt(sigma2(beta_hat)/n).
With b_n = 3, B is exactly our Delta (pairs within two hops). Auxiliary variables:
    ND-F       phi = X (own covariates);
    ND-phi0(G1) G1 = (D_i, treated share of neighbours, X_i, neighbour mean of X), normalized by
                phi0(G) = G - gamma_i w_HT,i, gamma_i = E[w_HT,i G_i]/E[w_HT,i^2] (Monte Carlo);
    ND-phi0(G2) G2 = (G1 1{T=t}, G1 1{T=t'}), same normalization.
Same population, design and draws as exp1 (seed_base 7,000,000; the first 50 draws are exp1's draws), so the
PNA rows of results/exp1_<setting>.csv can be compared on identical assignments.
Output: results/exp5_lwz_<setting>.csv
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import (EPS, SETTINGS, aipw_fixed, complete_randomization, design, ger_solve, hajek,
                    per_exposure_ols, population, syn)

K, D1, D2, P = 3, 0, 2, 1 / 3
Z975 = 1.959963984540054


def g1_features(A_adj, Atr, X):
    """LWZ's G1: own treatment, treated share of neighbours, own covariates, neighbour covariate means."""
    deg = A_adj.sum(1)
    nb = lambda v: np.where(deg > 0, (A_adj @ v) / np.maximum(deg, 1), 0.0)
    return np.column_stack([Atr, nb(Atr), X, np.column_stack([nb(X[:, j]) for j in range(X.shape[1])])])


def gamma_mc(G, A_adj, X, pi, reps, seed, interacted):
    """gamma_i = E[w_HT,i G_i] / E[w_HT,i^2] for the phi0 normalization, by Monte Carlo over assignments."""
    rng = np.random.default_rng(seed)
    n = G.shape[0]
    sampler = complete_randomization(P)
    num, den = 0.0, np.zeros(n)
    for _ in range(reps):
        Atr = sampler(rng, n).astype(float)
        D = syn.compute_exposure(G, Atr.astype(np.int8), K)
        h1, h2 = (D == D1) / pi[:, D1], (D == D2) / pi[:, D2]
        w = h1 - h2
        Gi = g1_features(A_adj, Atr, X)
        if interacted:
            Gi = np.hstack([Gi * (D == D1)[:, None], Gi * (D == D2)[:, None]])
        num = num + w[:, None] * Gi
        den += w ** 2
    return num / np.maximum(den, EPS)[:, None]


def lwz(Y, D, pi, B, phi):
    """Hajek network-dependent regression adjustment; returns (tau, lo, hi, cond)."""
    n = len(Y)
    h1, h2 = (D == D1) / pi[:, D1], (D == D2) / pi[:, D2]
    s1, s2 = h1.sum(), h2.sum()
    w = h1 / (s1 / n) - h2 / (s2 / n)
    hm = lambda v, h, s: (h @ v) / s  # Hajek mean of v (vector or matrix columns)
    a = (Y - hm(Y, h1, s1)) * h1 - (Y - hm(Y, h2, s2)) * h2
    Bm = (phi - hm(phi, h1, s1)) * h1[:, None] - (phi - hm(phi, h2, s2)) * h2[:, None]
    BtM = Bm.T @ B
    Amat = BtM @ Bm
    beta = np.linalg.lstsq(Amat, BtM @ a, rcond=None)[0]
    tau = float((Y - phi @ beta) @ w / n)
    V = a - Bm @ beta
    var = max(float(V @ B @ V) / n ** 2, 0.0)
    se = np.sqrt(var)
    return tau, tau - Z975 * se, tau + Z975 * se, np.linalg.cond(Amat)


def run_draw(b, seed_base, G, A_adj, X, Y_pot, pi, Kn, W, gam1, gam2):
    rng = np.random.default_rng(seed_base + b)
    n = G.shape[0]
    Atr = np.zeros(n, dtype=np.int8)
    Atr[rng.choice(n, size=int(round(P * n)), replace=False)] = 1  # identical to exp1.run_draw
    D = syn.compute_exposure(G, Atr, K)
    Y = Y_pot[np.arange(n), D]
    h1, h2 = (D == D1) / pi[:, D1], (D == D2) / pi[:, D2]
    w_ht = h1 - h2
    Xc = X[:, 1:]  # drop the intercept; Hajek weights make it irrelevant
    g1 = g1_features(A_adj, Atr.astype(float), Xc)
    g2 = np.hstack([g1 * (D == D1)[:, None], g1 * (D == D2)[:, None]])
    B = Kn.astype(float)
    f1, f2 = per_exposure_ols(Y, D, X, D1), per_exposure_ols(Y, D, X, D2)
    res = {
        "hajek": hajek(Y, D, D1, D2, pi, W),
        "linear-raw": aipw_fixed(Y, D, D1, D2, pi, W, f1, f2),
        "linear-linear": ger_solve(Y, D, D1, D2, pi, Kn, W, X, X),
        "linear-pred": ger_solve(Y, D, D1, D2, pi, Kn, W, f1[:, None], f2[:, None]),
        "lwz-ND-F": lwz(Y, D, pi, B, Xc),
        "lwz-ND-phi0(G1)": lwz(Y, D, pi, B, g1 - gam1 * w_ht[:, None]),
        "lwz-ND-phi0(G2)": lwz(Y, D, pi, B, g2 - gam2 * w_ht[:, None]),
        "lwz-ND-G1 (no phi0)": lwz(Y, D, pi, B, g1),
    }
    return [{"draw": b, "method": m, "tau": r[0], "lo": r[1], "hi": r[2], "cond": r[3]} for m, r in res.items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS))
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--gamma-reps", type=int, default=5000)
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--seed-base", type=int, default=7_000_000)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for setting in args.settings:
        t0 = time.time()
        G, H, X, Y_pot = population(setting)
        true_tau = float(np.mean(Y_pot[:, D1] - Y_pot[:, D2]))
        pi, Kn, L, W = design(G, K, D1, D2, syn.MC_REPS, syn.SEED_MC, syn.compute_exposure, complete_randomization(P))
        A_adj = G - np.eye(G.shape[0])
        gam1 = gamma_mc(G, A_adj, X[:, 1:], pi, args.gamma_reps, 2024, interacted=False)
        gam2 = gamma_mc(G, A_adj, X[:, 1:], pi, args.gamma_reps, 2024, interacted=True)
        out = Parallel(n_jobs=args.n_jobs)(
            delayed(run_draw)(b, args.seed_base, G, A_adj, X, Y_pot, pi, Kn, W, gam1, gam2) for b in range(args.B))
        df = pd.DataFrame([r for rows in out for r in rows])
        df["true_tau"] = true_tau
        df["setting"] = setting
        df.to_csv(os.path.join(args.out, f"exp5_lwz_{setting}.csv"), index=False)
        print(f"{setting}: saved {len(df)} rows in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
