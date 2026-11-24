"""Summarize the rebuttal experiments into plain-text tables (results/summary.txt)."""
import os

import numpy as np
import pandas as pd

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
pd.set_option("display.width", 200)


def metrics(g):
    t = g["true_tau"].iloc[0]
    v = g["tau"].to_numpy()
    bias, sd = v.mean() - t, v.std(ddof=1)
    cov = np.mean((g["lo"] <= t) & (t <= g["hi"]))
    return pd.Series({"B": len(v), "bias": bias, "sd": sd, "rmse": np.hypot(bias, sd), "coverage": cov})


def exp1(setting):
    path = os.path.join(R, f"exp1_{setting}.csv")
    if not os.path.exists(path):
        return
    d = pd.read_csv(path)
    d["config"] = d["config"].fillna("-")
    d["seed"] = d["seed"].fillna(-1)
    d["rcond"] = d["rcond"].fillna(0)
    main_rc = d["rcond"].isin([0, 1e-2])
    print(f"\n=== Experiment 1, setting {setting} (true tau {d.true_tau.iloc[0]:.4f}, B = {d.draw.nunique()} draws) ===")

    print("\n[A] Seed stability, paper PNA config, rcond 1e-2: per-seed metrics summarized over seeds")
    p = d[(d.config == "paper") & main_rc & d.method.isin(["pna-raw", "pna-pred", "pna-linear"])]
    per_seed = p.groupby(["method", "seed"]).apply(metrics).reset_index()
    print(per_seed.groupby("method")[["sd", "rmse", "coverage"]].agg(["median", "min", "max"]).round(4).to_string())
    within = p.groupby(["method", "draw"])["tau"].std(ddof=1).groupby("method").mean()
    design_sd = per_seed.groupby("method")["sd"].median()
    print("\nmean across-seed SD of tau-hat within a draw, and its ratio to the design SD:")
    print(pd.DataFrame({"within_draw_seed_sd": within, "ratio_to_design_sd": within / design_sd}).round(4).to_string())
    ens = d[d.method.isin(["pna-raw-ens", "pna-pred-ens"])].groupby("method").apply(metrics)
    print("\nseed-ensemble (average of the 5 seeds' predictions; no selection):")
    print(ens.round(4).to_string())

    print("\n[B] Untuned PNA configurations (seed 0) vs paper config (seed 0)")
    u = d[d.method.isin(["pna-raw", "pna-pred", "pna-linear"]) & main_rc & (d.seed == 0)]
    print(u.groupby(["config", "method"]).apply(metrics).round(4).to_string())
    lin = d[d.method.isin(["hajek", "linear-raw", "linear-linear", "linear-pred"]) & main_rc]
    print("\nreference rows (same draws):")
    print(lin.groupby("method").apply(metrics).round(4).to_string())

    print("\n[C] Conditioning of the GER system: condition number of A-hat (median / 90% / max over draws)")
    c = d[d.cond.notna() & (d.rcond == 1e-2) & ((d.seed <= 0))]
    q = lambda p: (lambda x: np.quantile(x, p, method="higher"))
    print(c.groupby(["method", "config"])["cond"].agg(median=q(0.5), p90=q(0.9), max="max")
          .to_string(float_format=lambda x: f"{x:.3g}"))
    print("\nsensitivity to truncation level (rcond; -1 = relative ridge 1e-3), seed 0 where applicable:")
    s = d[d.method.isin(["linear-linear", "linear-pred", "pna-linear"]) & (d.config.isin(["-", "paper"]))
          & (d.seed <= 0) & (d.rcond != 0)]
    print(s.groupby(["method", "rcond"]).apply(metrics)[["bias", "sd", "rmse", "coverage"]].round(4).to_string())

    print("\n[D] Graph cross-fitting (Algorithm 2) vs full-sample fit, paper config, seed 0")
    x = d[(d.method.isin(["pna-raw-cf", "pna-pred-cf"])) | (d.method.isin(["pna-raw", "pna-pred"]) & (d.config == "paper") & (d.seed == 0))]
    print(x.groupby("method").apply(metrics).round(4).to_string())


def exp2():
    path = os.path.join(R, "exp2_degree.csv")
    if not os.path.exists(path):
        return
    d = pd.read_csv(path)
    g = pd.read_csv(os.path.join(R, "exp2_degree_graphs.csv"))
    print("\n=== Experiment 2: hubs and positivity (Chung-Lu, Bernoulli(1/3)) ===")
    print(g.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    m = d.groupby(["mapping", "cap", "method"]).apply(metrics).reset_index()
    haj = m[m.method == "hajek"].set_index(["mapping", "cap"])["sd"]
    m["sd_ratio_vs_hajek"] = m.apply(lambda r: r["sd"] / haj[(r["mapping"], r["cap"])], axis=1)
    print(m.round(4).to_string(index=False))


def exp3():
    import glob
    gfiles = sorted(glob.glob(os.path.join(R, "exp3_graph_*.csv")))
    if not gfiles:
        return
    graphs = pd.concat([pd.read_csv(f) for f in gfiles])
    print("\n=== Experiment 3: BlogCatalog / Flickr semi-synthetic (Bernoulli(1/3), contrast d=1 vs d=0) ===")
    print(graphs.set_index("graph").T.to_string(float_format=lambda x: f"{x:.4g}"))
    for name in graphs["graph"]:
        path = os.path.join(R, f"exp3_{name}.csv")
        if os.path.exists(path):
            d = pd.read_csv(path).dropna(subset=["tau"])
            gat = os.path.join(R, f"exp3_{name}_gat.csv")
            if os.path.exists(gat):  # GAT rows come from a separate 20-draw run on the same draw seeds
                g = pd.read_csv(gat).dropna(subset=["tau"])
                d = pd.concat([d, g[g.method.str.startswith("gat")]])
            m = d.groupby("method", sort=False).apply(metrics)
            m["median_cond"] = d.groupby("method", sort=False)["cond"].median()
            print(f"\n{name} (true tau {d.true_tau.iloc[0]:.4f})")
            print(m.round(4).to_string(float_format=lambda x: f"{x:.4g}"))


def exp5():
    for setting in ("out1", "out2"):
        path = os.path.join(R, f"exp5_lwz_{setting}.csv")
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        print(f"\n=== Experiment 5: Lu-Wang-Zhang vs GER, setting {setting} (true tau {d.true_tau.iloc[0]:.4f}) ===")
        m = d.groupby("method", sort=False).apply(metrics)
        m["width"] = (d.hi - d.lo).groupby(d.method, sort=False).median()
        print(m.round(4).to_string())
        e1 = os.path.join(R, f"exp1_{setting}.csv")
        if os.path.exists(e1):  # exp1 used the same draws: compare PNA rows on the shared draws
            p = pd.read_csv(e1)
            p = p[(p.config == "paper") & (p.seed == 0) & p.method.isin(["pna-raw", "pna-pred"])
                  & (p.rcond.isna() | (p.rcond == 1e-2))]
            both = pd.concat([d[d.draw.isin(p.draw)], p[d.columns.intersection(p.columns)]])
            print(f"\nshared draws (B = {p.draw.nunique()}), adding PNA seed 0 from exp1:")
            mm = both.groupby("method", sort=False).apply(metrics)
            mm["width"] = (both.hi - both.lo).groupby(both.method, sort=False).median()
            print(mm.round(4).to_string())


def exp4():
    for outcome in ("revenue_per_ent", "wagebill_per_ent"):
        path = os.path.join(R, f"exp4_kenya_{outcome}.csv")
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        print(f"\n=== Experiment 4: Kenya {outcome}, PNA over {d.seed.nunique()} seeds (incl. 50039) ===")
        fixed = d[d.seed.isna()].pivot(index="contrast", columns="method", values="width")
        print("CI widths of deterministic and seed-ensemble estimators:")
        print(fixed.round(0).to_string())
        print("estimates [CI]:")
        fe = d[d.seed.isna()].assign(est=lambda x: x.apply(lambda r: f"{r.tau:.0f} [{r.lo:.0f}, {r.hi:.0f}]", axis=1))
        print(fe.pivot(index="contrast", columns="method", values="est").to_string())
        haj = d[d.method == "hajek"].set_index("contrast")
        for m in ("pna-ger-pred", "pna-raw"):
            s = d[d.method == m]
            q = lambda x, p: x.quantile(p)
            g = s.groupby("contrast").agg(tau_med=("tau", "median"), tau_q1=("tau", lambda x: q(x, .25)),
                                          tau_q3=("tau", lambda x: q(x, .75)), width_med=("width", "median"),
                                          width_q1=("width", lambda x: q(x, .25)), width_q3=("width", lambda x: q(x, .75)))
            g["share_narrower_than_hajek"] = s.groupby("contrast").apply(lambda x: (x.width < haj.loc[x.name, "width"]).mean())
            g["share_significant"] = s.groupby("contrast").apply(lambda x: ((x.lo > 0) | (x.hi < 0)).mean())
            avg = s.groupby("seed").width.mean()
            print(f"\n{m}: per-contrast distribution over seeds")
            print(g.round(2).to_string())
            print(f"average width over contrasts: median {avg.median():.0f} IQR [{avg.quantile(.25):.0f}, {avg.quantile(.75):.0f}]"
                  f"; Hajek {haj.width.mean():.0f}; share of seeds narrower than Hajek {(avg < haj.width.mean()).mean():.2f};"
                  f" seed 50039: {avg.get(50039.0, np.nan):.0f} (rank {int((avg < avg.get(50039.0, np.inf)).sum()) + 1} of {len(avg)})")


def exp6():
    for setting in ("out1", "out2"):
        gpath, epath = (os.path.join(R, f"exp6_retune_{setting}_{k}.csv") for k in ("grid", "eval"))
        if not os.path.exists(epath):
            continue
        grid = pd.read_csv(gpath).sort_values("rmse")
        e = pd.read_csv(epath)
        print(f"\n=== Experiment 6: re-tuning on a replicate population, {setting} ===")
        print("grid on the replicate population (objective: RMSE of GER on PNA representations), top 5:")
        print(grid.head(5)[["config", "bias", "sd", "rmse"]].round(4).to_string(index=False))
        print(f"selected: {e.config.iloc[0]}; evaluated on the paper's population (exp1 draws):")
        m = e.groupby("method", sort=False).apply(metrics)
        e1 = os.path.join(R, f"exp1_{setting}.csv")
        if os.path.exists(e1):
            p = pd.read_csv(e1)
            p = p[(p.config == "paper") & (p.seed == 0) & p.method.isin(["pna-raw", "pna-pred", "pna-linear"])
                  & (p.rcond.isna() | (p.rcond == 1e-2)) & p.draw.isin(e.draw)]
            pm = p.groupby("method").apply(metrics)
            pm.index = [f"{i} (paper config)" for i in pm.index]
            m = pd.concat([m, pm])
        print(m.round(4).to_string())


def exp7():
    path = os.path.join(R, "exp7_semisyn_crossfit.csv")
    if os.path.exists(path):
        d = pd.read_csv(path)
        print(f"\n=== Experiment 7: cross-fitting on the student network (true tau {d.true_tau.iloc[0]:.4f}) ===")
        m = d.groupby("method", sort=False).apply(metrics)
        m["width"] = (d.hi - d.lo).groupby(d.method, sort=False).median()
        print(m.round(4).to_string())


if __name__ == "__main__":
    for s in ("out1", "out2"):
        exp1(s)
    exp2()
    exp3()
    exp4()
    exp5()
    exp6()
    exp7()
