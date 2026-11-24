"""Rebuttal experiment 7: graph cross-fitting (Appendix D.5) on the student-network semi-synthetic study (Table 2).

Same population, covariates, Bernoulli(1/3) design and contrast (d = 2 vs 0) as code/semisynthetic. Folds:
connected components of the two-hop graph are independent, so they are assigned whole to the smaller fold; a
component larger than a quarter of the sample is split by spectral bisection, and units with a two-hop
neighbour in the other fold are removed from the training set (the buffer). Each fold's PNA predicts the other.
Output: results/exp7_semisyn_crossfit.csv
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse.csgraph import connected_components

from common import aipw_fixed, bernoulli, design, ger_solve, hajek, per_exposure_ols, syn, train_pna

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "semisynthetic"))
from semisynthetic_data import adjustment_covariates, build_student_network, generate_potential_outcomes  # noqa: E402

K, D1, D2, P = 3, 2, 0, 1 / 3


def crossfit_split(G):
    Kn = (G @ G.T > 0).astype(float)
    np.fill_diagonal(Kn, 0)
    n = len(Kn)
    _, lab = connected_components(Kn, directed=False)
    V1 = np.zeros(n, bool)
    assigned = np.zeros(n, bool)
    sizes = np.bincount(lab)
    for c in np.argsort(-sizes):
        idx = np.flatnonzero(lab == c)
        if sizes[c] > n / 4:
            sub = Kn[np.ix_(idx, idx)]
            f = np.linalg.eigh(np.diag(sub.sum(1)) - sub)[1][:, 1]
            V1[idx[f <= np.median(f)]] = True
        elif V1.sum() <= (assigned & ~V1).sum():  # greedy: add to the smaller fold
            V1[idx] = True
        assigned[idx] = True
    V2 = ~V1
    T1 = V1 & ~(Kn[:, V2].sum(1) > 0)
    T2 = V2 & ~(Kn[:, V1].sum(1) > 0)
    return V1, V2, T1, T2


def run_draw(b, seed_base, G, X, Y_pot, pi, Kn, W, split):
    rng = np.random.default_rng(seed_base + b)
    n = G.shape[0]
    D = syn.compute_exposure(G, (rng.random(n) < P).astype(np.int8), K)
    Y = Y_pot[np.arange(n), D]
    A_adj = G - np.eye(n)  # as in run_semisynthetic.py
    f1, f2 = per_exposure_ols(Y, D, X, D1), per_exposure_ols(Y, D, X, D2)
    p1, p2, _, _ = train_pna(X, A_adj, D, Y, K, D1, D2, seed=seed_base + 1000 * b, config=syn.PNA_CONFIG)
    V1, V2, T1, T2 = split
    c1, c2 = np.zeros(n), np.zeros(n)
    for T, V in ((T1, V2), (T2, V1)):
        idx = np.flatnonzero(T)
        q1, q2, _, _ = train_pna(X[idx], A_adj[np.ix_(idx, idx)], D[idx], Y[idx], K, D1, D2,
                                 seed=seed_base + 1000 * b, config=syn.PNA_CONFIG, predict_on=(X, A_adj))
        c1[V], c2[V] = q1[V], q2[V]
    res = {"hajek": hajek(Y, D, D1, D2, pi, W),
           "linear-raw": aipw_fixed(Y, D, D1, D2, pi, W, f1, f2),
           "linear-linear": ger_solve(Y, D, D1, D2, pi, Kn, W, X, X),
           "linear-pred": ger_solve(Y, D, D1, D2, pi, Kn, W, f1[:, None], f2[:, None]),
           "pna-raw": aipw_fixed(Y, D, D1, D2, pi, W, p1, p2),
           "pna-pred": ger_solve(Y, D, D1, D2, pi, Kn, W, p1[:, None], p2[:, None]),
           "pna-raw-cf": aipw_fixed(Y, D, D1, D2, pi, W, c1, c2),
           "pna-pred-cf": ger_solve(Y, D, D1, D2, pi, Kn, W, c1[:, None], c2[:, None])}
    return [{"draw": b, "method": m, "tau": r[0], "lo": r[1], "hi": r[2]} for m, r in res.items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=50)
    ap.add_argument("--mc-reps", type=int, default=1000)
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--seed-base", type=int, default=20260000)  # run_semisynthetic.py's draws
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    t0 = time.time()
    net = build_student_network()
    G = net["G"]
    Y_pot = generate_potential_outcomes(net, k=K, seed=999)
    true_tau = float(np.mean(Y_pot[:, D1] - Y_pot[:, D2]))
    X = adjustment_covariates(net)
    pi, Kn, L, W = design(G, K, D1, D2, args.mc_reps, 111, syn.compute_exposure, bernoulli(P))
    split = crossfit_split(G)
    print(f"n={len(X)}, tau={true_tau:.4f}; |V1|={split[0].sum()}, |T1|={split[2].sum()}, "
          f"|V2|={split[1].sum()}, |T2|={split[3].sum()}", flush=True)
    out = Parallel(n_jobs=args.n_jobs)(delayed(run_draw)(b, args.seed_base, G, X, Y_pot, pi, Kn, W, split)
                                       for b in range(args.B))
    df = pd.DataFrame([r for rows in out for r in rows])
    df["true_tau"] = true_tau
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "exp7_semisyn_crossfit.csv"), index=False)
    print(f"saved ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
