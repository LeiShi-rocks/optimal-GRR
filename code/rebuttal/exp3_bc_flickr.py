"""Rebuttal experiment 3 (GPU): semi-synthetic studies on the denser BlogCatalog and Flickr graphs.

Same construction as the student-network study (code/semisynthetic): the real graph is kept, covariates
come from the real node attributes, and potential outcomes follow semisynthetic_data.generate_potential_outcomes
with the same parameters. Graphs: PyG AttributedGraphDataset BlogCatalog (n = 5,196) and Flickr (n = 7,575),
largest connected component, undirected, no self-loops (mean degree ~ 65).

Deviations from the student-network study, forced by the much denser graphs:
  * covariates w and z (the GPA slot) are the first two standardized TruncatedSVD components of the
    bag-of-words attributes; r~ is the standardized log(1 + degree) (raw degree has max ~ 4,000);
  * contrast d = 1 vs d = 0 (treated share in [1/3, 2/3) vs < 1/3). Under Bernoulli(1/3) the share of a
    high-degree unit concentrates at 1/3, so P(D = 2) collapses for hubs; this is recorded in the graph table;
  * beta1 = 1.0 instead of 0.4, so the d = 1 vs 0 effect (~0.48) is comparable to the paper's 0 vs 2 effect.
Assignment: i.i.d. Bernoulli(1/3); fraction mapping with k = 3. Exposure probabilities by Monte Carlo.
Estimators: HT, Hajek; linear-raw/-linear/-pred; {PNA, GAT}-raw/-pred/-linear with the paper configurations.
Each GER row is also solved with diag(Delta 1) in place of Delta ("-ydiag"): Delta is indefinite here.
Output: results/rebuttal/exp3_<graph>.csv (one row per draw x method) and results/rebuttal/exp3_graphs.csv.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import EPS, aipw_fixed, bernoulli, design, ger_solve, hajek, per_exposure_ols, syn, train_pna, _ci

HERE = os.path.dirname(os.path.abspath(__file__))
SEMI = os.path.join(os.path.dirname(HERE), "semisynthetic")
sys.path.insert(0, SEMI)
from semisynthetic_data import PARAMS, _neighbor_mean, generate_potential_outcomes  # noqa: E402
from GRR import GAT_embedding, prepare_grr_inputs  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data", "attributed")
K, D1, D2, P = 3, 1, 0, 1 / 3
# d = 1 vs 0 spans only half the dose range (s = d/2), so the main effect is raised from 0.4 to 1.0.
EXP3_PARAMS = {**PARAMS, "beta1": 1.0}


def load_graph(name):
    """Largest connected component of the undirected attributed graph; returns (A, features)."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    from torch_geometric.datasets import AttributedGraphDataset
    d = AttributedGraphDataset(DATA, name)[0]
    n = d.num_nodes
    ei = d.edge_index.numpy()
    A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(float)
    A.setdiag(0)
    A.eliminate_zeros()
    _, lab = connected_components(A, directed=False)
    keep = np.flatnonzero(lab == np.bincount(lab).argmax())
    F = sp.csr_matrix(d.x.to_dense().numpy())
    return A[keep][:, keep].toarray(), F[keep]


def build_network(name):
    """Analogue of semisynthetic_data.build_student_network for an attributed graph."""
    from sklearn.decomposition import TruncatedSVD
    A, F = load_graph(name)
    deg = A.sum(1)
    S = TruncatedSVD(n_components=2, random_state=0).fit_transform(F)
    S = (S - S.mean(0)) / S.std(0)
    w, z = S[:, 0], S[:, 1]
    r = np.log1p(deg)
    r_tilde = (r - r.mean()) / r.std()
    two_hop = ((A + A @ A) > 0).astype(float)
    np.fill_diagonal(two_hop, 0.0)
    return {"A": A, "G": A + np.eye(len(deg)), "deg": deg, "w": w, "r_tilde": r_tilde, "gpa": z,
            "w_bar1": _neighbor_mean(A, w), "r_bar1": _neighbor_mean(A, r_tilde),
            "w_bar2": _neighbor_mean(two_hop, w), "gpa_bar2": _neighbor_mean(two_hop, z)}


def ht(Y, D, d1, d2, pi, W):
    """HT estimate with the drivers' CI construction (constant feature, beta = HT means)."""
    m1, m2 = syn.mu_ht(Y, D, d1, pi, EPS), syn.mu_ht(Y, D, d2, pi, EPS)
    ones = np.ones((len(Y), 1))
    inp = prepare_grr_inputs(Y, D, ones, d1, d2, pi, X1=ones, X2=ones, eps=EPS)
    tau = float(m1 - m2)
    return (tau, *_ci(tau, inp, ones, ones, np.array([m1]), np.array([m2]), W), np.nan)


def train_gat(X, A_adj, D, Y, seed):
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    out = GAT_embedding(X, A_adj, D, Y, k=K, d1=D1, d2=D2, verbose=False, return_model=True, **syn.GAT_CONFIG)
    m = out["model"]
    return m.predict(X, D=D1), m.predict(X, D=D2), out["X_GAT_leaf"][0], out["X_GAT_leaf"][1]


def run_draw(b, seed_base, A_adj, X, Y_pot, pi, Kn, W, Kd, gnns):
    rng = np.random.default_rng(seed_base + b)
    n = len(X)
    D = syn.compute_exposure(A_adj + np.eye(n), (rng.random(n) < P).astype(np.int8), K)
    Y = Y_pot[np.arange(n), D]
    rows = []

    def add(method, fn):
        try:
            r = fn()
        except Exception as e:  # keep the remaining methods; NaN rows are counted in the summary
            print(f"draw {b} {method}: {e}", flush=True)
            r = (np.nan,) * 4
        rows.append({"draw": b, "method": method, "tau": r[0], "lo": r[1], "hi": r[2], "cond": r[3]})

    add("ht", lambda: ht(Y, D, D1, D2, pi, W))
    add("hajek", lambda: hajek(Y, D, D1, D2, pi, W))
    f1, f2 = per_exposure_ols(Y, D, X, D1), per_exposure_ols(Y, D, X, D2)
    add("linear-raw", lambda: aipw_fixed(Y, D, D1, D2, pi, W, f1, f2))
    add("linear-linear", lambda: ger_solve(Y, D, D1, D2, pi, Kn, W, X, X))
    add("linear-pred", lambda: ger_solve(Y, D, D1, D2, pi, Kn, W, f1[:, None], f2[:, None]))
    add("linear-linear-ydiag", lambda: ger_solve(Y, D, D1, D2, pi, Kd, W, X, X))
    add("linear-pred-ydiag", lambda: ger_solve(Y, D, D1, D2, pi, Kd, W, f1[:, None], f2[:, None]))
    for gnn, fit in (("pna", lambda: train_pna(X, A_adj, D, Y, K, D1, D2, seed=seed_base + b, config=syn.PNA_CONFIG)),
                     ("gat", lambda: train_gat(X, A_adj, D, Y, seed=seed_base + b))):
        if gnn not in gnns:
            continue
        t = time.time()
        try:
            p1, p2, h1, h2 = fit()
        except Exception as e:
            print(f"draw {b} {gnn} training: {e}", flush=True)
            continue
        add(f"{gnn}-raw", lambda: aipw_fixed(Y, D, D1, D2, pi, W, p1, p2))
        add(f"{gnn}-pred", lambda: ger_solve(Y, D, D1, D2, pi, Kn, W, p1[:, None], p2[:, None]))
        add(f"{gnn}-linear", lambda: ger_solve(Y, D, D1, D2, pi, Kn, W, h1, h2))
        add(f"{gnn}-pred-ydiag", lambda: ger_solve(Y, D, D1, D2, pi, Kd, W, p1[:, None], p2[:, None]))
        add(f"{gnn}-linear-ydiag", lambda: ger_solve(Y, D, D1, D2, pi, Kd, W, h1, h2))
        print(f"draw {b} {gnn} {time.time() - t:.0f}s", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", nargs="+", default=["BlogCatalog", "Flickr"])
    ap.add_argument("--gnns", nargs="*", default=["pna", "gat"])
    ap.add_argument("--tag", default="", help="suffix for output files")
    ap.add_argument("--B", type=int, default=50)
    ap.add_argument("--mc-reps", type=int, default=2000)
    ap.add_argument("--n-jobs", type=int, default=3)
    ap.add_argument("--seed-outcome", type=int, default=999)
    ap.add_argument("--seed-mc", type=int, default=111)
    ap.add_argument("--seed-base", type=int, default=9_000_000)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "results", "rebuttal"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    graphs = []
    for name in args.graphs:
        t0 = time.time()
        net = build_network(name)
        G, n = net["G"], len(net["w"])
        Y_pot = generate_potential_outcomes(net, k=K, seed=args.seed_outcome, params=EXP3_PARAMS)
        true_tau = float(np.mean(Y_pot[:, D1] - Y_pot[:, D2]))
        X = np.column_stack([np.ones(n), net["w"], net["r_tilde"], net["gpa"]])
        pi, Kn, L, W = design(G, K, D1, D2, args.mc_reps, args.seed_mc, syn.compute_exposure, bernoulli(P))
        # Delta is indefinite on dense graphs; diag(Delta 1) >= Delta (Young's inequality) is a PSD bound.
        Kd = np.diag(Kn.sum(1).astype(float))
        g = {"graph": name, "n": n, "mean_deg": net["deg"].mean(), "max_deg": net["deg"].max(),
             "Delta_density": (Kn.sum() - n) / (n * (n - 1)), "true_tau": true_tau,
             "mc_reps": args.mc_reps, "setup_min": (time.time() - t0) / 60,
             "Delta_min_eig": np.linalg.eigvalsh(Kn.astype(float))[0]}
        for d in range(K):
            g[f"min_pi_d{d}"] = pi[:, d].min()
            g[f"share_pi_d{d}_below_0.01"] = np.mean(pi[:, d] < 0.01)
        graphs.append(g)
        print(g, flush=True)

        out = Parallel(n_jobs=args.n_jobs, verbose=5)(
            delayed(run_draw)(b, args.seed_base, net["A"], X, Y_pot, pi, Kn, W, Kd, args.gnns) for b in range(args.B))
        df = pd.DataFrame([r for rows in out for r in rows])
        df["true_tau"] = true_tau
        df["graph"] = name
        df.to_csv(os.path.join(args.out, f"exp3_{name}{args.tag}.csv"), index=False)
        pd.DataFrame([g]).to_csv(os.path.join(args.out, f"exp3_graph_{name}.csv"), index=False)
        print(f"saved exp3_{name}{args.tag}.csv ({len(df)} rows) in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
