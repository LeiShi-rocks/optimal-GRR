"""Toy illustration: the adjustment coefficient that minimizes prediction error is not the one that
minimizes the design variance of the AIPW estimator under interference.

Self-contained (numpy + matplotlib). Writes figures/fig1_mismatch.svg, figures/fig2_weights.svg and
data/toy.json (curves for the interactive figure). Everything here is an illustration, not a paper result.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from style import BLUE, GRAY, ORANGE, save, setup

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
DATA = os.path.join(HERE, "..", "data")
rng = np.random.default_rng(7)

# ---- population: a sparse graph with heterogeneous degree --------------------------------------
n, p_treat, k = int(os.environ.get("TOY_N", 1000)), 1 / 3, 3
w = rng.gamma(1.2, 2.5, n)                     # Chung-Lu weights -> mean degree about 3, some hubs
P = np.minimum(np.outer(w, w) / w.sum(), 1.0)
A = np.triu(rng.random((n, n)) < P, 1)
A = (A | A.T).astype(float)
G = A + np.eye(n)                              # self-inclusive, as in the paper
deg = G.sum(1)


def exposure(a):
    """Share of treated units in the self-inclusive neighbourhood, cut into k equal bins."""
    frac = (G @ a.T).T / deg
    return np.minimum((frac * k).astype(int), k - 1)


# ---- potential outcomes: the covariate matters more for well-connected units --------------------
x = rng.normal(0, 1, n)
slope = np.where(deg >= 5, 3.0, 0.3)
base = slope * x + rng.normal(0, 0.3, n)
d1, d0 = 2, 0
Y1, Y0 = base + 1.0, base.copy()               # constant effect tau = 1 between d = 2 and d = 0
xc = x - x.mean()

# ---- design quantities by Monte Carlo ----------------------------------------------------------
R = int(os.environ.get("TOY_R", 20000))
draws = (rng.random((R, n)) < p_treat).astype(np.float32)
E = np.vstack([exposure(draws[i:i + 2000]) for i in range(0, R, 2000)])
I1, I0 = (E == d1).astype(np.float32), (E == d0).astype(np.float32)
pi1, pi0 = I1.mean(0), I0.mean(0)
keep = (pi1 > 0.002) & (pi0 > 0.002)           # drop units with (near) no support for this contrast
print(f"n = {n}, kept {keep.sum()} units with support for d = 2 vs d = 0; mean degree {deg.mean() - 1:.2f}")
I1, I0, pi1, pi0 = I1[:, keep], I0[:, keep], pi1[keep], pi0[keep]
Y1, Y0, xc, degk, slopek = Y1[keep], Y0[keep], xc[keep], deg[keep], slope[keep]
m = keep.sum()
S = np.hstack([I1 / pi1, -I0 / pi0])            # signed reweighted indicators
S = S - S.mean(0)
Lam = (S.T @ S) / (R - 1)                       # covariance of the signed vector = Lambda


def design_sd(b):
    r = np.concatenate([Y1 - b * xc, Y0 - b * xc])
    return float(np.sqrt(r @ Lam @ r) / m)


def pred_mse(b):
    return float(np.mean(np.concatenate([(Y1 - Y1.mean()) - b * xc, (Y0 - Y0.mean()) - b * xc]) ** 2))


bs = np.linspace(-0.5, 3.0, 141)
sd = np.array([design_sd(b) for b in bs])
mse = np.array([pred_mse(b) for b in bs])
b_var = bs[sd.argmin()]
b_mse = float(np.sum(xc * ((Y1 - Y1.mean()) + (Y0 - Y0.mean()))) / (2 * np.sum(xc * xc)))
print(f"b minimizing prediction MSE {b_mse:.2f}; b minimizing design SD {b_var:.2f}")
print(f"design SD at b_mse {design_sd(b_mse):.4f}, at b_var {design_sd(b_var):.4f}, at b=0 {design_sd(0):.4f}")

# ---- GER from single realized assignments (two-hop Delta), versus the MSE fit -------------------
Delta = ((G @ G) > 0)[np.ix_(keep, keep)].astype(float)
b_ger, b_ols, t_ger, t_ols = [], [], [], []
for r in range(int(os.environ.get("TOY_DRAWS", 1000))):
    a = (rng.random(n) < p_treat).astype(float)
    e = exposure(a[None, :])[0][keep]
    i1, i0 = (e == d1) / pi1, (e == d0) / pi0
    yobs = np.where(e == d1, Y1, np.where(e == d0, Y0, np.nan))
    X = (i1 - 1) * xc - (i0 - 1) * xc               # R1 - R2 for a shared coefficient
    yhat = i1 * np.nan_to_num(yobs) - i0 * np.nan_to_num(yobs)
    b_ger.append(float(X @ Delta @ yhat / (X @ Delta @ X)))
    s = (e == d1) | (e == d0)                       # the MSE fit uses observed outcomes, level intercepts
    Z = np.column_stack([e[s] == d1, e[s] == d0, xc[s]]).astype(float)
    b_ols.append(float(np.linalg.lstsq(Z, yobs[s], rcond=None)[0][2]))
    y0 = np.nan_to_num(yobs)
    for b, out in ((b_ger[-1], t_ger), (b_ols[-1], t_ols)):
        out.append(np.mean(i1 * (y0 - b * xc) + b * xc) - np.mean(i0 * (y0 - b * xc) + b * xc))
b_ger, b_ols = np.array(b_ger), np.array(b_ols)
sd_ger, sd_ols = float(np.std(t_ger)), float(np.std(t_ols))
print(f"realized SD of tau-hat: GER {sd_ger:.4f}, OLS {sd_ols:.4f} (oracle {design_sd(b_var):.4f})")
print(f"GER b-hat: median {np.median(b_ger):.2f} IQR [{np.percentile(b_ger, 25):.2f}, {np.percentile(b_ger, 75):.2f}]")
print(f"OLS b-hat: median {np.median(b_ols):.2f} IQR [{np.percentile(b_ols, 25):.2f}, {np.percentile(b_ols, 75):.2f}]")

os.makedirs(FIG, exist_ok=True)
os.makedirs(DATA, exist_ok=True)
toy = {"b": bs.round(4).tolist(), "sd": sd.round(5).tolist(), "mse": mse.round(5).tolist(),
       "b_var": round(float(b_var), 3), "b_mse": round(b_mse, 3),
       "sd_unadjusted": round(design_sd(0), 5), "sd_ger": round(sd_ger, 5), "sd_ols": round(sd_ols, 5),
       "n": int(m), "mean_degree": round(float(deg.mean() - 1), 2)}
json.dump(toy, open(os.path.join(DATA, "toy.json"), "w"))
with open(os.path.join(DATA, "toy.js"), "w") as f:  # loadable from file:// without fetch
    f.write("window.TOY = " + json.dumps(toy) + ";\n")

# ---- Figure 1: two objectives, two minimizers --------------------------------------------------
setup()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.4), gridspec_kw={"width_ratios": [1.35, 1]})
ax1.plot(bs, mse / mse.min(), color=GRAY, lw=2, label="prediction error (MSE)")
ax1.plot(bs, sd / sd.min(), color=BLUE, lw=2, label="estimator SD (design)")
ax1.axvline(b_mse, color=GRAY, ls=":", lw=1.2)
ax1.axvline(b_var, color=BLUE, ls=":", lw=1.2)
ax1.set_ylim(0.95, 2.2)
ax1.set_xlabel("adjustment coefficient $b$")
ax1.set_ylabel("relative to its own minimum")
ax1.legend(frameon=False, loc="upper center")
ax1.set_title("Two objectives, two minimizers", loc="left")
bins = np.linspace(-0.5, 3.0, 50)
ax2.hist(b_ols, bins=bins, color=GRAY, alpha=0.6, label="MSE fit (OLS)")
ax2.hist(b_ger, bins=bins, color=ORANGE, alpha=0.7, label="GER")
ax2.axvline(b_var, color=BLUE, ls=":", lw=1.2)
ax2.axvline(b_mse, color=GRAY, ls=":", lw=1.2)
ax2.set_xlabel("estimated $\\hat b$ from one experiment")
ax2.set_yticks([])
ax2.legend(frameon=False, loc="upper right")
ax2.text(0.98, 0.62, f"SD of $\\hat\\tau$:\nMSE fit {sd_ols:.3f}\nGER {sd_ger:.3f}", transform=ax2.transAxes,
         ha="right", va="top", fontsize=9)
ax2.set_title("What each fit lands on", loc="left")
fig.tight_layout()
save(fig, os.path.join(FIG, "fig1_mismatch.svg"))

# ---- Figure 2: where the design puts its weight ------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 3.2))
wdiag = np.diag(Lam)[:m]
jit = rng.uniform(-0.25, 0.25, m)
ax.scatter(degk - 1 + jit, wdiag, s=10, c=np.where(slopek > 1, ORANGE, GRAY), alpha=0.8, lw=0)
ax.set_yscale("log")
ax.set_xlabel("number of neighbours")
ax.set_ylabel("weight $(1-\\pi_i)/\\pi_i$ at $d=2$")
ax.set_title("The design weights well-connected units heavily", loc="left")
ax.text(0.98, 0.08, "orange: units where $x$ matters more", transform=ax.transAxes, ha="right", color=ORANGE)
fig.tight_layout()
save(fig, os.path.join(FIG, "fig2_weights.svg"))
print("wrote figures/fig1_mismatch.svg, figures/fig2_weights.svg, data/toy.json")
