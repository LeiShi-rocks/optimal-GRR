"""Re-plot experimental results from the project's own result files (no values typed in by hand).

Inputs (under <repo>/results/):
  synthetic/dgp2/out{1,2}_dgp2_parallel_*_summary.csv, out1_dgp2_linear_no_cal_*  -> paper Tables 1 and 6
  rebuttal/exp1_out{1,2}.csv   independent 50-draw rerun of the same designs, 5 PNA seeds
  rebuttal/exp2_degree.csv     heavy-tailed degrees
  rebuttal/exp3_{BlogCatalog,Flickr}[_gat].csv   dense graphs
  rebuttal/exp4_kenya_*.csv    Kenya cash transfers, 50 PNA seeds
  rebuttal/exp7_semisyn_crossfit.csv   student network, with cross-fitting
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style import BLUE, DARK, GRAY, ORANGE, RED, save, setup

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results")
FIG = os.path.join(HERE, "..", "figures")
setup()


def summarize(df, truth_col="true_tau"):
    err = df["tau"] - df[truth_col]
    cover = (df["lo"] <= df[truth_col]) & (df[truth_col] <= df["hi"])
    return pd.Series({"bias": err.mean(), "sd": df["tau"].std(ddof=0), "rmse": np.sqrt((err ** 2).mean()),
                      "coverage": cover.mean(), "B": len(df)})


# ---- Figure 3: synthetic studies, archived paper runs and an independent rerun ----------------------
SYN = os.path.join(RES, "synthetic", "dgp2")
paper = {
    "sparse (Table 1)": pd.concat([pd.read_csv(os.path.join(SYN, "out1_dgp2_parallel_20260128_054224_summary.csv")),
                                   pd.read_csv(os.path.join(SYN, "out1_dgp2_linear_no_cal_20260128_194808_summary.csv"))]),
    "denser (Table 6)": pd.read_csv(os.path.join(SYN, "out2_dgp2_parallel_20260128_222447_summary.csv")),
}
rows = [("Hájek", "hajek", None), ("linear, uncalibrated", "linear-no-cal", "raw"), ("linear + GER", "linear", "ger"),
        ("GAT, uncalibrated", "gat-pred", "raw"), ("GAT + GER", "gat-no-harm", "ger"),
        ("PNA, uncalibrated", "pna-pred", "raw"), ("PNA + GER", "pna-no-harm", "ger")]
rerun_key = {"hajek": ("hajek", None), "linear-no-cal": ("linear-raw", None), "linear": ("linear-linear", 0.01),
             "pna-pred": ("pna-raw", None), "pna-no-harm": ("pna-pred", 0.01)}


def rerun_rmse(setting, method, rcond):
    d = pd.read_csv(os.path.join(RES, "rebuttal", f"exp1_{setting}.csv"))
    d = d[d.method == method]
    if method.startswith("pna"):
        d = d[d.config == "paper"]
    d = d[d.rcond.isna()] if rcond is None else d[np.isclose(d.rcond, rcond)]
    by_seed = [summarize(g)["rmse"] for _, g in d.groupby(d.seed.fillna(-1))]
    return float(np.median(by_seed))


fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), sharey=True)
for ax, (title, tab), setting in zip(axes, paper.items(), ["out1", "out2"]):
    tab = tab.set_index("method")
    for y, (label, key, kind) in enumerate(rows[::-1]):
        c = DARK if kind is None else (GRAY if kind == "raw" else BLUE)
        ax.plot(tab.loc[key, "rmse"], y, "o", color=c, ms=7, zorder=3)
        if key in rerun_key:
            ax.plot(rerun_rmse(setting, *rerun_key[key]), y, "o", mfc="white", mec=c, ms=7, mew=1.5, zorder=4)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows[::-1]])
    ax.set_xlabel("RMSE (lower is better)")
    ax.set_title(title, loc="left")
    ax.grid(axis="x", alpha=0.3)
axes[1].plot([], [], "o", color=GRAY, label="paper run (100 draws)")
axes[1].plot([], [], "o", mfc="white", mec=GRAY, mew=1.5, label="independent rerun (50 draws, median of 5 seeds)")
axes[1].legend(frameon=False, loc="lower right", fontsize=8)
fig.tight_layout()
save(fig, os.path.join(FIG, "fig3_synthetic.svg"))

# ---- Figure 4: student network (semi-synthetic), including cross-fitting -------------------------
d = pd.read_csv(os.path.join(RES, "rebuttal", "exp7_semisyn_crossfit.csv"))
s7 = d.groupby("method").apply(summarize)
order = [("Hájek", "hajek", DARK), ("linear, uncalibrated", "linear-raw", GRAY), ("linear + GER", "linear-linear", BLUE),
         ("PNA, uncalibrated", "pna-raw", GRAY), ("PNA + GER", "pna-pred", BLUE),
         ("PNA + GER, cross-fitted", "pna-pred-cf", ORANGE)]
fig, ax = plt.subplots(figsize=(6.4, 2.9))
for y, (label, key, c) in enumerate(order[::-1]):
    ax.barh(y, s7.loc[key, "rmse"], color=c, height=0.6)
    ax.text(s7.loc[key, "rmse"] + 0.0008, y, f"{s7.loc[key, 'rmse']:.4f}", va="center", fontsize=8.5)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([o[0] for o in order[::-1]])
ax.set_xlabel("RMSE (lower is better)")
ax.set_title(f"Student network, n = 2,983 ({int(s7.loc['hajek', 'B'])} draws)", loc="left")
fig.tight_layout()
save(fig, os.path.join(FIG, "fig4_student.svg"))

# ---- Figure 5: heavy-tailed degrees ------------------------------------------------------------
d = pd.read_csv(os.path.join(RES, "rebuttal", "exp2_degree.csv"))
s2 = d.groupby(["mapping", "cap", "method"]).apply(summarize).reset_index()
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2), sharey=True)
for ax, (mapping, name) in zip(axes, [("fraction", "exposure = share of treated neighbours"),
                                      ("any", "exposure = any neighbour treated")]):
    s = s2[s2.mapping == mapping]
    haj = s[s.method == "hajek"].set_index("cap")["sd"]
    for method, label, c, ls in [("linear-raw", "linear, uncalibrated", GRAY, "-"),
                                 ("linear-pred", "GER on predictions", BLUE, "--"),
                                 ("linear-linear", "GER on covariates", BLUE, "-")]:
        r = s[s.method == method].set_index("cap")["sd"] / haj
        ax.plot(r.index, r.values, ls, marker="o", color=c, label=label)
    ax.axhline(1, color=DARK, lw=1, ls=":")
    ax.text(10, 1.03, "Hájek", color=DARK, fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xticks([10, 30, 100, 300])
    ax.set_xticklabels(["10", "30", "100", "300"])
    ax.set_xlabel("maximum degree allowed")
    ax.set_title(name, loc="left")
axes[0].set_ylabel("SD relative to Hájek")
axes[1].legend(frameon=False, fontsize=8.5, loc="upper right")
fig.tight_layout()
save(fig, os.path.join(FIG, "fig5_heavy_tails.svg"))

# ---- Figure 6: dense graphs -------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.0), sharey=True)
for ax, g in zip(axes, ["BlogCatalog", "Flickr"]):
    d = pd.read_csv(os.path.join(RES, "rebuttal", f"exp3_{g}.csv"))
    s3 = d.groupby("method").apply(summarize)
    tau = d.true_tau.iloc[0]
    items = [("Hájek", "hajek", DARK), ("PNA, uncalibrated", "pna-raw", GRAY),
             ("PNA + GER, $\\Delta$ weight", "pna-pred", RED), ("PNA + GER, PSD weight", "pna-pred-ydiag", BLUE)]
    for y, (label, key, c) in enumerate(items[::-1]):
        ax.barh(y, s3.loc[key, "rmse"] / tau * 100, color=c, height=0.6)
        ax.text(s3.loc[key, "rmse"] / tau * 100 * 1.08, y, f"bias {s3.loc[key, 'bias'] / tau * 100:+.0f}%",
                va="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([i[0] for i in items[::-1]])
    ax.set_xlabel("RMSE as % of the true effect (log scale)")
    ax.set_xlim(0.5, 200)
    meta = pd.read_csv(os.path.join(RES, "rebuttal", f"exp3_graph_{g}.csv")).iloc[0]
    ax.set_title(f"{g}: n = {int(meta.n):,}, mean degree {meta.mean_deg:.0f}", loc="left")
fig.tight_layout()
save(fig, os.path.join(FIG, "fig6_dense.svg"))

# ---- Figure 7: Kenya, 50 PNA seeds ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.0))
for ax, (outcome, name) in zip(axes, [("revenue_per_ent", "revenue per enterprise"),
                                      ("wagebill_per_ent", "wage bill per enterprise")]):
    d = pd.read_csv(os.path.join(RES, "rebuttal", f"exp4_kenya_{outcome}.csv"))
    det = d[d.seed.isna()].groupby("method")["width"].mean()
    rng = np.random.default_rng(0)
    for y, (key, label, c) in enumerate([("pna-raw", "PNA, uncalibrated", GRAY), ("pna-ger-pred", "PNA + GER", BLUE)]):
        w = d[(d.method == key) & d.seed.notna()].groupby("seed")["width"].mean()
        ax.scatter(w.values, y + rng.uniform(-0.15, 0.15, len(w)), s=14, color=c, alpha=0.7, lw=0)
        if 50039 in w.index:
            ax.scatter(w.loc[50039], y, s=60, facecolor="none", edgecolor=ORANGE, lw=1.6, zorder=4)
    for key, label, ls in [("hajek", "Hájek", ":"), ("linear", "linear", "--")]:
        ax.axvline(det[key], color=DARK, ls=ls, lw=1)
        ax.text(det[key], 1.42, label, ha="center", fontsize=8, color=DARK)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["PNA, uncalibrated", "PNA + GER"])
    ax.set_ylim(-0.5, 1.6)
    ax.set_xlabel("average 95% CI width (KES); one dot per seed")
    ax.set_title(name, loc="left")
fig.tight_layout()
save(fig, os.path.join(FIG, "fig7_kenya.svg"))
print("wrote fig3-fig7")
