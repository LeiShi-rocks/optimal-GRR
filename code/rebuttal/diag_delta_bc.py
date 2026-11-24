"""Diagnostic: PSD dependence-weighted replacements for Delta in the GER objective (BlogCatalog, 20 draws).

xi_i = 1{D_i=d1}/pi_i(d1) - 1{D_i=d2}/pi_i(d2) is the design score of unit i; C = Corr(xi) from Monte Carlo.
Candidates: Delta (paper), diag(Delta 1), Delta o C (correlation-weighted), Delta o 1{|C| >= c0}
(strong pairs only); each indefinite candidate is projected onto the PSD cone.
"""
import numpy as np

from common import design, ger_solve, hajek, aipw_fixed, per_exposure_ols, bernoulli, syn
from exp3_bc_flickr import D1, D2, EXP3_PARAMS, K, P, build_network, generate_potential_outcomes

MC = 2000


def score_corr(G, seed=111):
    rng = np.random.default_rng(seed)
    n = G.shape[0]
    E = np.array([syn.compute_exposure(G, (rng.random(n) < P).astype(np.int8), K) for _ in range(MC)])
    p1, p2 = (E == D1).mean(0), (E == D2).mean(0)
    S = (E == D1) / p1 - (E == D2) / p2
    S = (S - S.mean(0)).astype(np.float64)
    C = S.T @ S / MC
    s = np.sqrt(np.diag(C))
    return C / np.outer(s, s)


def psd(M):
    ev, V = np.linalg.eigh(M)
    return (V * np.clip(ev, 0, None)) @ V.T, ev[0]


net = build_network("BlogCatalog")
n = len(net["w"])
Y_pot = generate_potential_outcomes(net, k=K, seed=999, params=EXP3_PARAMS)
tau = float(np.mean(Y_pot[:, D1] - Y_pot[:, D2]))
X = np.column_stack([np.ones(n), net["w"], net["r_tilde"], net["gpa"]])
pi, Kn, L, W = design(net["G"], K, D1, D2, 500, 111, syn.compute_exposure, bernoulli(P))
Kf = Kn.astype(float)
C = score_corr(net["G"])
off = Kf.astype(bool) & ~np.eye(n, dtype=bool)
print(f"tau {tau:.4f}; 2-hop pairs: {off.sum() / 2:.0f}; |corr| quantiles over 2-hop pairs "
      f"(50/90/99%): {np.quantile(np.abs(C[off]), [.5, .9, .99]).round(3)}")

cands = {"Delta": Kf, "diag(Delta1)": np.diag(Kf.sum(1))}
M, e0 = psd(Kf * C)
cands["Delta o C"] = M
print(f"Delta o C min eig before projection {e0:.3f}")
for c0 in (0.05, 0.1, 0.2, 0.3):
    T = Kf * (np.abs(C) >= c0)
    np.fill_diagonal(T, 1.0)
    M, e0 = psd(T)
    cands[f"Delta o 1(|C|>={c0})"] = M
    print(f"c0={c0}: kept {(T.astype(bool) & off).sum() / off.sum():.3%} of 2-hop pairs, min eig {e0:.2f}")

res = {}
for b in range(20):
    rng = np.random.default_rng(9_000_000 + b)
    D = syn.compute_exposure(net["G"], (rng.random(n) < P).astype(np.int8), K)
    Y = Y_pot[np.arange(n), D]
    f1, f2 = per_exposure_ols(Y, D, X, D1), per_exposure_ols(Y, D, X, D2)
    res.setdefault("hajek", []).append(hajek(Y, D, D1, D2, pi, W)[:3])
    res.setdefault("linear-raw", []).append(aipw_fixed(Y, D, D1, D2, pi, W, f1, f2)[:3])
    for cn, M in cands.items():
        rc = 1e-2 if cn == "Delta" else 1e-8
        res.setdefault(f"lin-linear | {cn}", []).append(ger_solve(Y, D, D1, D2, pi, M, W, X, X, rcond=rc)[:3])
        res.setdefault(f"lin-pred   | {cn}", []).append(
            ger_solve(Y, D, D1, D2, pi, M, W, f1[:, None], f2[:, None], rcond=rc)[:3])
for k, v in res.items():
    v = np.array(v)
    e = v[:, 0]
    print(f"{k:34s} bias={e.mean() - tau:+.4f} sd={e.std(ddof=1):.4f} rmse={np.sqrt(((e - tau) ** 2).mean()):.4f} "
          f"cov={np.mean((v[:, 1] <= tau) & (tau <= v[:, 2])):.2f} CIw={np.median(v[:, 2] - v[:, 1]):.3f}")
