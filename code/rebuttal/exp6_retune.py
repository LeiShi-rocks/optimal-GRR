"""Rebuttal experiment 6: re-tune PNA on a replicate population, then evaluate on the paper's population (AC, item c).

The paper tuned PNA on the evaluation population itself (code/synthetic/tuning/tune-pna-grr.py: seeds 123/1234/999,
objective = RMSE of GER on PNA representations over 20 draws). Here the same 16-configuration grid and objective are
run on a replicate population drawn with different graph/covariate/outcome seeds; the selected configuration is then
evaluated on the paper's population with exp1's draws (seed_base 7,000,000), so it can be compared with the paper
configuration row by row.
Output: results/exp6_retune_<setting>_grid.csv and results/exp6_retune_<setting>_eval.csv
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import (SETTINGS, aipw_fixed, complete_randomization, design, ger_solve, hajek, population, syn,
                    train_pna)

K, D1, D2, P = 3, 0, 2, 1 / 3
REPLICATE_SEEDS = dict(graph=20261, cov=20262, out=20263, mc=20264, draws=8_000_000)
# (name, pna_hidden_dims, rep_dim, head_hidden_dims, n_epochs, lr, weight_decay, dropout): the paper's grid
GRID = [
    ("shallow_small", [32], 8, [16], 300, 1e-3, 1e-4, 0.0),
    ("shallow_medium", [64], 16, [32], 300, 1e-3, 1e-4, 0.0),
    ("medium_small", [64, 32], 16, [32], 300, 1e-3, 1e-4, 0.0),
    ("medium_medium", [64, 32], 32, [64, 32], 300, 1e-3, 1e-4, 0.0),
    ("deep_small", [64, 64, 32], 16, [32], 300, 1e-3, 1e-4, 0.0),
    ("medium_lr_high", [64, 32], 16, [32], 300, 1e-2, 1e-4, 0.0),
    ("medium_lr_low", [64, 32], 16, [32], 300, 1e-4, 1e-4, 0.0),
    ("medium_lr_vlow", [64, 32], 16, [32], 500, 5e-5, 1e-4, 0.0),
    ("medium_wd_high", [64, 32], 16, [32], 300, 1e-3, 1e-3, 0.0),
    ("medium_wd_vhigh", [64, 32], 16, [32], 300, 1e-3, 1e-2, 0.0),
    ("medium_dropout", [64, 32], 16, [32], 300, 1e-3, 1e-4, 0.1),
    ("medium_dropout_wd", [64, 32], 16, [32], 300, 1e-3, 1e-3, 0.1),
    ("medium_long", [64, 32], 16, [32], 500, 1e-3, 1e-3, 0.0),
    ("medium_vlong", [64, 32], 16, [32], 1000, 5e-4, 1e-3, 0.0),
    ("tiny", [16], 4, [8], 200, 1e-3, 1e-3, 0.0),
    ("linear_head", [32], 8, [], 300, 1e-3, 1e-3, 0.0),
]


def to_config(g):
    name, hid, rep, head, ep, lr, wd, dr = g
    return {**syn.PNA_CONFIG, "pna_hidden_dims": hid, "rep_dim": rep, "head_hidden_dims": head,
            "n_epochs": ep, "lr": lr, "weight_decay": wd, "dropout": dr}


def replicate_population(setting):
    avg, mx = SETTINGS[setting]
    G, H = syn.generate_graph(n=syn.N, avg_degree=avg, max_degree=mx, seed=REPLICATE_SEEDS["graph"])
    X, X_noi = syn.generate_covariates(H, seed=REPLICATE_SEEDS["cov"])
    Y_pot = syn.generate_potential_outcomes(G, H, X_noi, k=K, beta_neighbor=1.0, seed=REPLICATE_SEEDS["out"])
    return G, X[:, [0, 3]], Y_pot


def draw(G, Y_pot, seed):
    rng = np.random.default_rng(seed)
    n = G.shape[0]
    A = np.zeros(n, dtype=np.int8)
    A[rng.choice(n, size=int(round(P * n)), replace=False)] = 1
    D = syn.compute_exposure(G, A, K)
    return D, Y_pot[np.arange(n), D]


def grid_task(gi, b, G, X, Y_pot, pi, Kn, W):
    D, Y = draw(G, Y_pot, REPLICATE_SEEDS["draws"] + b)
    _, _, h1, h2 = train_pna(X, G.astype(float), D, Y, K, D1, D2, seed=REPLICATE_SEEDS["draws"] + 1000 * b,
                             config=to_config(GRID[gi]))
    return gi, b, ger_solve(Y, D, D1, D2, pi, Kn, W, h1, h2)[0]


def eval_task(b, cfg, G, X, Y_pot, pi, Kn, W):
    D, Y = draw(G, Y_pot, 7_000_000 + b)  # exp1's draw b
    p1, p2, h1, h2 = train_pna(X, G.astype(float), D, Y, K, D1, D2, seed=7_000_000 + 1000 * b, config=cfg)
    rows = {"hajek": hajek(Y, D, D1, D2, pi, W),
            "pna-raw": aipw_fixed(Y, D, D1, D2, pi, W, p1, p2),
            "pna-pred": ger_solve(Y, D, D1, D2, pi, Kn, W, p1[:, None], p2[:, None]),
            "pna-linear": ger_solve(Y, D, D1, D2, pi, Kn, W, h1, h2)}
    return [{"draw": b, "method": m, "tau": r[0], "lo": r[1], "hi": r[2]} for m, r in rows.items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS))
    ap.add_argument("--B-tune", type=int, default=20)
    ap.add_argument("--B-eval", type=int, default=50)
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for setting in args.settings:
        t0 = time.time()
        G, X, Y_pot = replicate_population(setting)
        tau_rep = float(np.mean(Y_pot[:, D1] - Y_pot[:, D2]))
        pi, Kn, L, W = design(G, K, D1, D2, syn.MC_REPS, REPLICATE_SEEDS["mc"], syn.compute_exposure,
                              complete_randomization(P))
        out = Parallel(n_jobs=args.n_jobs)(delayed(grid_task)(gi, b, G, X, Y_pot, pi, Kn, W)
                                           for gi in range(len(GRID)) for b in range(args.B_tune))
        grid = pd.DataFrame(out, columns=["gi", "draw", "tau"])
        summ = grid.groupby("gi")["tau"].agg(bias=lambda v: v.mean() - tau_rep, sd="std").reset_index()
        summ["rmse"] = np.hypot(summ.bias, summ.sd)
        summ["config"] = [GRID[i][0] for i in summ.gi]
        summ["setting"] = setting
        summ.to_csv(os.path.join(args.out, f"exp6_retune_{setting}_grid.csv"), index=False)
        best = int(summ.loc[summ.rmse.idxmin(), "gi"])
        print(f"{setting}: replicate tau={tau_rep:.4f}; selected {GRID[best][0]} "
              f"({(time.time() - t0) / 60:.1f} min)\n{summ.sort_values('rmse').round(4).to_string(index=False)}", flush=True)

        G0, H0, X0, Y0 = population(setting)
        true_tau = float(np.mean(Y0[:, D1] - Y0[:, D2]))
        pi0, Kn0, L0, W0 = design(G0, K, D1, D2, syn.MC_REPS, syn.SEED_MC, syn.compute_exposure, complete_randomization(P))
        ev = Parallel(n_jobs=args.n_jobs)(delayed(eval_task)(b, to_config(GRID[best]), G0, X0, Y0, pi0, Kn0, W0)
                                          for b in range(args.B_eval))
        df = pd.DataFrame([r for rows in ev for r in rows])
        df["true_tau"], df["setting"], df["config"] = true_tau, setting, GRID[best][0]
        df.to_csv(os.path.join(args.out, f"exp6_retune_{setting}_eval.csv"), index=False)
        print(f"{setting}: evaluation saved ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
