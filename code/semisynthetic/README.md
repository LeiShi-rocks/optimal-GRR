# Student-network semi-synthetic study (paper Section 6, Table 2), rebuilt

The original code was not recovered from this repository, the S3 bucket, git history, or editor history. This folder rebuilds the study from three sources:
- the paper;
- the ICML-2026 rebuttal (`writeup/icml2026/rebuttal.tex`), which lists all DGP parameter values;
- checks against the numbers the paper reports.

## Data (restricted use)

ICPSR 37070, Paluck, Shepherd & Aronow, "Changing Climates of Conflict". Put `37070-0001-Data.dta` under `<repo>/data/school-network/ICPSR_37070/DS0001/`. A copy is in `s3://epic-ace-leis-jobs/TabFN-CATE/data/paluck/` and in the project backup under `data/`. `data/` and `results/` are git-ignored; do not commit or redistribute the data.

## Run

```bash
python run_semisynthetic.py                     # B = 200, all methods, 1,000 Monte Carlo draws
python run_semisynthetic.py --B 2 --mc-reps 50  # smoke test
```

The driver imports the estimators, GNN code and GAT/PNA settings from `../synthetic` (via `_run_method` of the medium-density driver). Method keys mean the same as there; see `../synthetic/README.md`. Outputs are written to `<repo>/results/semisynthetic/semisyn_<timestamp>_{summary.csv,full.npz,config.json}`.

## Specification and evidence

| Item | Choice | Evidence |
| --- | --- | --- |
| Units | eligible students, `TREAT` in {1, 2}, n = 2,983 | the only natural subset with exactly n = 2,983 |
| Network | undirected union of wave-1 "spent time with" nominations ST1–ST10 | self-inclusive mean degree 2.45, the paper's "approximately 2.5" (the paper defines degree with G_ii = 1) |
| Exposure | share of treated units in the self-inclusive neighbourhood, cut into 3 equal bins (`dgp.compute_exposure`) | the same mapping as the synthetic studies, which the paper describes in the same words |
| Assignment | i.i.d. Bernoulli(1/3), also used for the Monte Carlo propensities | paper Section 6 |
| w_i | `WRISTW2`, missing set to the sample mean | see the τ* check below |
| r̃_i | standardized degree | rebuttal |
| Neighbour averages | 1-hop for w and r̃; within-2-hop (excluding self) for w and GPA; isolated units get 0, as in `dgp-2.py` | rebuttal |
| Parameters | α=0.30, β₁=0.40, β₂=−0.10, γ₁…γ₇ = 0.10, 0.03, 0.05, 0.30, 0.10, 0.50, 0.20; δ=0.15, δ₂=0.20, δ₃=0.30; σ=0.20 | rebuttal |
| Noise | independent N(0, σ²) per exposure level | same as `dgp-2.py` |
| Adjustment features | intercept, w, r̃, GPA | not stated anywhere; assumed |
| GNN settings | the synthetic-study GAT/PNA configurations | not stated anywhere; assumed |

**The τ* check.** τ* = mean[Y(2) − Y(0)] does not depend on the exposure cut-offs, so it can discriminate between candidate covariate definitions:
- With the choices above, the deterministic part of τ* is 0.4619, against the paper's 0.4623. Across noise seeds it is 0.457–0.475; seed 999 gives 0.4615.
- With `WRISTOW2` instead, τ* is about 0.36–0.37.
- Standardizing w gives about 0.30.

A smoke test with B = 20 reproduces the paper's magnitudes:

| Method | This rebuild (B = 20) | Paper |
| --- | --- | --- |
| Hájek SD | 0.050 | 0.052 |
| Linear GER SD | 0.027 | 0.030 |

## What is still not reproduced

- The paper calls w the "baseline wrist measurement", but `WRISTW2` is the wave-2 wristband variable. As a fixed covariate in a simulation this is harmless, but the wording in the paper should change.
- The adjustment features and GNN hyperparameters used originally are unknown.
- The external baselines (NetEstimator, GCN-Deconfounder, RRNet) are not rebuilt.
- The paper's Table 2 mixes runs. Its GER rows match the rebuttal's B = 25 table, while its external-baseline rows come from the B = 200 table.
