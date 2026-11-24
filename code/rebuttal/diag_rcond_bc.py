"""Diagnostic: GER truncation level on the BlogCatalog smoke draws (same draws/seeds as exp3)."""
import numpy as np

from common import bernoulli, design, ger_solve, per_exposure_ols, syn
from exp3_bc_flickr import D1, D2, EXP3_PARAMS, K, P, build_network, generate_potential_outcomes

net = build_network("BlogCatalog")
n = len(net["w"])
Y_pot = generate_potential_outcomes(net, k=K, seed=999, params=EXP3_PARAMS)
tau = float(np.mean(Y_pot[:, D1] - Y_pot[:, D2]))
X = np.column_stack([np.ones(n), net["w"], net["r_tilde"], net["gpa"]])
pi, Kn, L, W = design(net["G"], K, D1, D2, 500, 111, syn.compute_exposure, bernoulli(P))
print(f"true tau {tau:.4f}")
for b in range(6):
    rng = np.random.default_rng(9_000_000 + b)
    D = syn.compute_exposure(net["G"], (rng.random(n) < P).astype(np.int8), K)
    Y = Y_pot[np.arange(n), D]
    f1, f2 = per_exposure_ols(Y, D, X, D1), per_exposure_ols(Y, D, X, D2)
    for name, X1, X2 in (("linear-linear", X, X), ("linear-pred", f1[:, None], f2[:, None])):
        out = [ger_solve(Y, D, D1, D2, pi, Kn, W, X1, X2, rcond=rc) for rc in (1e-2, 1e-4, 1e-6, 1e-10)]
        print(b, f"{name:14s} cond={out[0][3]:.3g} ",
              "  ".join(f"rc={rc:g}: {o[0]:+.4f} [{o[1]:+.3f},{o[2]:+.3f}]" for rc, o in zip((1e-2, 1e-4, 1e-6, 1e-10), out)))
