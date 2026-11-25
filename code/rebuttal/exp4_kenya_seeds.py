"""Rebuttal experiment 4: seed stability of PNA on the Kenya cash-transfer data (Tables 3 and 8).

For each outcome, PNA (the paper's per-outcome configuration) is trained with S seeds; for every seed and
contrast we record PNA-no-harm (GER on PNA predictions, the paper's column) and PNA-raw (AIPW, beta = 1).
A seed ensemble averages the S models' predictions, a fixed feature map that involves no selection.
HT, Hajek and Linear are deterministic and computed once. Seed 50039 (the paper's wage-bill seed) is
always included so its rank among the seeds can be reported.
Requires the outputs of preprocess.py and compute_inclusion_prob.py in analysis-GAUGER/.
Output: results/rebuttal/exp4_kenya_<outcome>.csv
"""
import argparse
import importlib.util
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
GAUGER = os.path.join(os.path.dirname(HERE), "Interference-CashTransfer", "analysis-GAUGER")
_spec = importlib.util.spec_from_file_location("kenya", os.path.join(GAUGER, "GRR-analysis.py"))
kenya = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kenya)

K = 4
CONTRASTS = [(3, 1, "Direct(H=1)"), (2, 0, "Direct(H=0)"), (1, 0, "Spillover(A=0)"), (3, 2, "Spillover(A=1)")]
PAPER_SEED = 50039


class _Fixed:
    """Stands in for a trained model: returns stored (normalized) predictions per exposure level."""

    def __init__(self, preds):
        self.preds = preds

    def predict(self, X, D):
        return self.preds[D]


def setup():
    data, G, propensity_df, so = kenya.load_data()
    pi = kenya.build_propensity_matrix(data, propensity_df)
    Kn = kenya.build_kn_matrix(G)
    W = {}
    for d1, d2, _ in CONTRASTS:
        p11 = kenya.build_second_order_matrix(so, len(data), d1)
        p22 = kenya.build_second_order_matrix(so, len(data), d2)
        W[(d1, d2)] = kenya.build_W_pi(pi, p11, p22, d1, d2, Kn, robust=False)
    X = np.nan_to_num(data[[c for c in kenya.COVARIATE_COLS if c in data.columns]].values, nan=0.0)
    return data, G, pi, Kn, W, X


def fit_seed(seed, X, G, D, Yn, config):
    kenya.set_seed(seed)
    out = kenya.PNA_embedding(X, G.astype(float), D, Yn, k=K, verbose=False, return_model=True, **config)
    m = out["model"]
    return seed, {d: m.predict(X, D=d) for d in range(K)}


def estimate(method, Y, D, X, pi, Kn, G, W, model, Ym, Ys):
    rows = []
    for d1, d2, name in CONTRASTS:
        r = kenya.estimate_contrast(Y, D, X, d1, d2, pi, Kn, G, method=method, pna_model=model,
                                    W1_pi=W[(d1, d2)][0], W2_pi=W[(d1, d2)][1], Y_mean=Ym, Y_std=Ys)
        rows.append({"contrast": name, "tau": r["tau_hat"], "lo": r["ci_lower"], "hi": r["ci_upper"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcomes", nargs="+", default=["revenue_per_ent", "wagebill_per_ent"])
    ap.add_argument("--S", type=int, default=50)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "results", "rebuttal"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    data, G, pi, Kn, W, X = setup()
    D = data["D"].values
    seeds = list(range(args.S - 1)) + [PAPER_SEED]
    for outcome in args.outcomes:
        t0 = time.time()
        Y = data[outcome].values
        Ym, Ys = np.nanmean(Y), np.nanstd(Y)
        Yn = (Y - Ym) / Ys
        config = kenya.PNA_CONFIGS[outcome]
        rows = []
        for method in ("ht", "hajek", "linear"):
            rows += [{"method": method, "seed": np.nan, **r} for r in estimate(method, Y, D, X, pi, Kn, G, W, None, Ym, Ys)]

        fits = dict(Parallel(n_jobs=args.n_jobs)(delayed(fit_seed)(s, X, G, D, Yn, config) for s in seeds))
        for s in seeds:
            model = _Fixed(fits[s])
            for method, label in (("pna-no-harm", "pna-ger-pred"), ("pna-pred", "pna-raw")):
                rows += [{"method": label, "seed": s, **r} for r in estimate(method, Y, D, X, pi, Kn, G, W, model, Ym, Ys)]
        ens = _Fixed({d: np.mean([fits[s][d] for s in seeds], 0) for d in range(K)})
        for method, label in (("pna-no-harm", "pna-ger-pred-ens"), ("pna-pred", "pna-raw-ens")):
            rows += [{"method": label, "seed": np.nan, **r} for r in estimate(method, Y, D, X, pi, Kn, G, W, ens, Ym, Ys)]

        df = pd.DataFrame(rows)
        df["width"] = df["hi"] - df["lo"]
        df["outcome"] = outcome
        df.to_csv(os.path.join(args.out, f"exp4_kenya_{outcome}.csv"), index=False)
        print(f"{outcome}: {len(seeds)} seeds in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
