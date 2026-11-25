"""Student-network semi-synthetic study (paper Section 6, Table 2): network, covariates and DGP.

Rebuilt from the paper text and the ICML-2026 rebuttal (writeup/icml2026/rebuttal.tex); the original
driver was not recovered. Choices that the written record does not pin down are marked "RECONSTRUCTED"
and are justified in README.md.

Data: ICPSR 37070 (Paluck, Shepherd & Aronow, "Changing Climates of Conflict"), restricted-use.
Place 37070-0001-Data.dta under <repo>/data/school-network/ICPSR_37070/DS0001/ (git-ignored; do not redistribute).
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DTA = HERE.parents[1] / "data" / "school-network" / "ICPSR_37070" / "DS0001" / "37070-0001-Data.dta"
NOMINATIONS = [f"ST{i}" for i in range(1, 11)]  # wave-1 "spent time with" nominations

# Rebuttal parameter values (writeup/icml2026/rebuttal.tex, eq. dgp).
PARAMS = dict(alpha=0.30, beta1=0.40, beta2=-0.10, gamma1=0.10, gamma2=0.03, gamma3=0.05,
              delta=0.15, gamma4=0.30, gamma5=0.10, delta2=0.20, gamma6=0.50, gamma7=0.20,
              delta3=0.30, sigma=0.20)


def _valid(s):
    """ICPSR user-missing codes are negative; treat them as missing."""
    s = pd.to_numeric(s, errors="coerce").astype(float)
    return s.where(s >= 0)


def _neighbor_mean(M, x):
    """Row-mean of x over M's nonzero entries; units with no neighbours get 0 (as in dgp-2.py)."""
    d = M.sum(1)
    return np.where(d > 0, (M @ x) / np.maximum(d, 1), 0.0)


def build_student_network(dta=DTA):
    """Eligible students (TREAT in {1, 2}; n = 2,983) with the undirected wave-1 ST network.

    RECONSTRUCTED: the eligible-student subset is the only natural subset with n = 2,983; the
    self-inclusive average degree H_i = 1 + deg_i is 2.45, matching "average degree approximately 2.5".
    """
    df = pd.read_stata(dta, convert_categoricals=False)
    df = df[df.TREAT.isin([1, 2])].copy()
    df["IDn"] = pd.to_numeric(df.ID, errors="coerce")
    df = df.reset_index(drop=True)
    n = len(df)
    index = {(s, i): k for k, (s, i) in enumerate(zip(df.SCHID, df.IDn))}
    A = np.zeros((n, n))
    for c in NOMINATIONS:
        for k, (s, v) in enumerate(zip(df.SCHID, df[c])):
            j = index.get((s, v)) if pd.notna(v) and v > 0 else None
            if j is not None and j != k:
                A[k, j] = A[j, k] = 1.0

    deg = A.sum(1)
    r_tilde = (deg - deg.mean()) / deg.std()
    # RECONSTRUCTED: "baseline wrist measurement" = WRISTW2 (wore the Roots wristband), missing -> mean.
    w = _valid(df.WRISTW2)
    w = w.fillna(w.mean()).to_numpy()
    gpa = _valid(df.GPA)
    gpa = gpa.fillna(gpa.mean()).to_numpy()

    two_hop = ((A + A @ A) > 0).astype(float)  # within two hops, excluding self
    np.fill_diagonal(two_hop, 0.0)
    return {
        "A": A,                                   # adjacency without self-loops
        "G": A + np.eye(n),                       # paper convention G_ii = 1
        "deg": deg,
        "w": w, "r_tilde": r_tilde, "gpa": gpa,
        "w_bar1": _neighbor_mean(A, w), "r_bar1": _neighbor_mean(A, r_tilde),
        "w_bar2": _neighbor_mean(two_hop, w), "gpa_bar2": _neighbor_mean(two_hop, gpa),
        "school": df.SCHID.to_numpy(),
    }


def generate_potential_outcomes(net, k=3, seed=999, params=PARAMS):
    """Y_i(d) for d = 0..k-1 from the rebuttal DGP; independent noise per level as in dgp-2.py."""
    p = params
    rng = np.random.default_rng(seed)
    n = net["w"].size
    Y = np.zeros((n, k))
    for d in range(k):
        s = d / (k - 1)  # d/2 for k = 3
        mean = (p["alpha"] + p["beta1"] * s + p["beta2"] * s ** 2
                + p["gamma1"] * net["w"] + p["gamma2"] * net["r_tilde"] + p["gamma3"] * net["r_tilde"] ** 2
                + p["delta"] * s * net["w"] + p["gamma4"] * net["w_bar1"] + p["gamma5"] * net["r_bar1"]
                + p["delta2"] * s * net["w_bar1"] + p["gamma6"] * net["w_bar2"]
                + p["gamma7"] * net["gpa_bar2"] + p["delta3"] * s * net["w_bar2"])
        Y[:, d] = rng.normal(mean, p["sigma"])
    return Y


def adjustment_covariates(net):
    """Unit-level features passed to linear GER and the GNN encoders (intercept + own covariates).

    RECONSTRUCTED: the paper does not list them; neighbourhood information is left to the GNN.
    """
    return np.column_stack([np.ones_like(net["w"]), net["w"], net["r_tilde"], net["gpa"]])


def bernoulli_mc_probabilities(G, p_treat, k, mc_reps, seed, d1, d2, exposure_fn):
    """Monte Carlo exposure probabilities under i.i.d. Bernoulli(p_treat) assignment.

    Same outputs as propensity.monte_carlo_probabilities, which instead uses complete randomization.
    """
    rng = np.random.default_rng(seed)
    n = G.shape[0]
    counts = np.zeros((n, k))
    j12, j11, j22 = (np.zeros((n, n)) for _ in range(3))
    for _ in range(mc_reps):
        D = exposure_fn(G, (rng.random(n) < p_treat).astype(np.int8), k)
        counts[np.arange(n), D] += 1
        m1, m2 = (D == d1).astype(float), (D == d2).astype(float)
        j12 += np.outer(m1, m2)
        j11 += np.outer(m1, m1)
        j22 += np.outer(m2, m2)
    out = [counts / mc_reps] + [j / mc_reps for j in (j12, j11, j22)]
    for j in out[1:]:
        np.fill_diagonal(j, 0.0)
    return tuple(out)
