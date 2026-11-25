"""Rebuttal experiment 1 (GPU): seed stability, untuned PNA configurations, GER conditioning, cross-fitting.

Synthetic populations of Table 1 ("out1") and Table 6 ("out2"); complete randomization with 1/3 treated,
contrast d=0 vs d=2, B seeded draws. Per draw:
  * linear backbone: Hajek, linear-raw, linear-linear and linear-pred (GER on per-exposure OLS predictions),
    each GER row at truncation levels rcond in RCONDS and with a relative ridge;
  * PNA with the paper configuration and S seeds: raw (no GER), pred (GER on predictions),
    linear (GER on representations, at every rcond / ridge), plus a seed-ensemble of the predictions;
  * PNA with two pre-specified, untuned configurations (package defaults; a small network), seed 0;
  * graph cross-fitting (Appendix D.5, Algorithm 2) with the paper configuration, seed 0.
Rows: one per (draw, method); written to results/rebuttal/exp1_<setting>.csv.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import (EPS, SETTINGS, aipw_fixed, complete_randomization, design, ger_solve, hajek,
                    per_exposure_ols, population, syn, train_pna)

RCONDS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-8]
RIDGE = 1e-3
K, D1, D2, P = 3, 0, 2, 1 / 3
CONFIGS = {
    "paper": syn.PNA_CONFIG,
    "default": {},  # PNA_embedding defaults, not tuned
    "small": {"pna_hidden_dims": [32], "rep_dim": 8, "head_hidden_dims": [16], "aggregators": ["mean"],
              "scalers": ["identity"], "dropout": 0.0, "n_epochs": 300},
}


def crossfit_split(G):
    """Balanced spectral bisection of the 2-hop graph and buffered training sets (Algorithm 2)."""
    Kn = (G @ G.T > 0).astype(float)
    np.fill_diagonal(Kn, 0)
    Lap = np.diag(Kn.sum(1)) - Kn
    fiedler = np.linalg.eigh(Lap)[1][:, 1]
    V1 = fiedler <= np.median(fiedler)
    V2 = ~V1
    T1 = V1 & ~((Kn[:, V2].sum(1)) > 0)
    T2 = V2 & ~((Kn[:, V1].sum(1)) > 0)
    return V1, V2, T1, T2


def run_draw(b, setting, seeds, seed_base, G, X, Y_pot, pi, Kn, W, split):
    rng = np.random.default_rng(seed_base + b)
    n = G.shape[0]
    A = np.zeros(n, dtype=np.int8)
    A[rng.choice(n, size=int(round(P * n)), replace=False)] = 1
    D = syn.compute_exposure(G, A, K)
    Y = Y_pot[np.arange(n), D]
    A_adj = G.astype(float)  # the drivers pass G with self-loops to the GNNs
    rows = []

    def add(method, res, **kw):
        rows.append({"draw": b, "method": method, "tau": res[0], "lo": res[1], "hi": res[2], "cond": res[3], **kw})

    def ger_grid(prefix, X1, X2, **kw):
        for rc in RCONDS:
            add(prefix, ger_solve(Y, D, D1, D2, pi, Kn, W, X1, X2, rcond=rc), rcond=rc, **kw)
        add(prefix, ger_solve(Y, D, D1, D2, pi, Kn, W, X1, X2, ridge=RIDGE), rcond=-1, **kw)  # -1 = ridge

    # Linear backbone (no randomness beyond the draw).
    add("hajek", hajek(Y, D, D1, D2, pi, W))
    f1, f2 = per_exposure_ols(Y, D, X, D1), per_exposure_ols(Y, D, X, D2)
    add("linear-raw", aipw_fixed(Y, D, D1, D2, pi, W, f1, f2))
    ger_grid("linear-linear", X, X)
    ger_grid("linear-pred", f1.reshape(-1, 1), f2.reshape(-1, 1))

    # PNA, paper configuration, several seeds.
    preds = []
    for s in seeds:
        p1, p2, h1, h2 = train_pna(X, A_adj, D, Y, K, D1, D2, seed=seed_base + 1000 * b + s, config=CONFIGS["paper"])
        preds.append((p1, p2))
        add("pna-raw", aipw_fixed(Y, D, D1, D2, pi, W, p1, p2), seed=s, config="paper")
        add("pna-pred", ger_solve(Y, D, D1, D2, pi, Kn, W, p1[:, None], p2[:, None]), seed=s, config="paper", rcond=1e-2)
        ger_grid("pna-linear", h1, h2, seed=s, config="paper")
    e1, e2 = np.mean([p[0] for p in preds], 0), np.mean([p[1] for p in preds], 0)
    add("pna-raw-ens", aipw_fixed(Y, D, D1, D2, pi, W, e1, e2), config="paper")
    add("pna-pred-ens", ger_solve(Y, D, D1, D2, pi, Kn, W, e1[:, None], e2[:, None]), config="paper", rcond=1e-2)

    # Untuned configurations.
    for name in ("default", "small"):
        p1, p2, h1, h2 = train_pna(X, A_adj, D, Y, K, D1, D2, seed=seed_base + 1000 * b, config=CONFIGS[name])
        add("pna-raw", aipw_fixed(Y, D, D1, D2, pi, W, p1, p2), seed=0, config=name)
        add("pna-pred", ger_solve(Y, D, D1, D2, pi, Kn, W, p1[:, None], p2[:, None]), seed=0, config=name, rcond=1e-2)
        add("pna-linear", ger_solve(Y, D, D1, D2, pi, Kn, W, h1, h2), seed=0, config=name, rcond=1e-2)

    # Graph cross-fitting: model trained on T_k predicts the other half V_{3-k}.
    V1, V2, T1, T2 = split
    cf1, cf2 = np.zeros(n), np.zeros(n)
    for T, V in ((T1, V2), (T2, V1)):
        idx = np.flatnonzero(T)
        q1, q2, _, _ = train_pna(X[idx], A_adj[np.ix_(idx, idx)], D[idx], Y[idx], K, D1, D2,
                                 seed=seed_base + 1000 * b, config=CONFIGS["paper"], predict_on=(X, A_adj))
        cf1[V], cf2[V] = q1[V], q2[V]
    add("pna-raw-cf", aipw_fixed(Y, D, D1, D2, pi, W, cf1, cf2), config="paper")
    add("pna-pred-cf", ger_solve(Y, D, D1, D2, pi, Kn, W, cf1[:, None], cf2[:, None]), config="paper", rcond=1e-2)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=SETTINGS, default="out1")
    ap.add_argument("--B", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--seed-base", type=int, default=7_000_000)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "rebuttal"))
    args = ap.parse_args()

    t0 = time.time()
    G, H, X, Y_pot = population(args.setting)
    true_tau = float(np.mean(Y_pot[:, D1] - Y_pot[:, D2]))
    pi, Kn, L, W = design(G, K, D1, D2, syn.MC_REPS, syn.SEED_MC, syn.compute_exposure, complete_randomization(P))
    split = crossfit_split(G)
    print(f"{args.setting}: tau={true_tau:.4f}; cross-fit |V1|={split[0].sum()}, |T1|={split[2].sum()}, "
          f"|V2|={split[1].sum()}, |T2|={split[3].sum()}", flush=True)

    out = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(run_draw)(b, args.setting, range(args.seeds), args.seed_base, G, X, Y_pot, pi, Kn, W, split)
        for b in range(args.B))
    df = pd.DataFrame([r for rows in out for r in rows])
    df["true_tau"] = true_tau
    df["setting"] = args.setting
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, f"exp1_{args.setting}.csv"), index=False)
    print(f"saved exp1_{args.setting}.csv ({len(df)} rows) in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
