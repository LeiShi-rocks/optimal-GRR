# Kenya cash-transfer study (paper Section 6, Tables 3 and 7–10, Figure 2): GAUGER code

The rest of `code/Interference-CashTransfer/` (`analysis/`, `analysis-2/`) contains Hájek/IPW analyses of other network definitions: 3 nearest villages within 4 km, dist6/k8, and so on. None of it uses PNA or GER, and none of it uses the paper's network. The GAUGER code was missing from the repository. The scripts in this folder were recovered from the editor's local history (Cursor; originally `Interference-CashTransfer/analysis-GRR-third-try/`, last saved 2026-01-27/28). The only changes are the import paths, which now point at `../../synthetic`.

| Script | Role | Settings found in the code |
| --- | --- | --- |
| `preprocess.py` | builds the village network, exposure (A, H) and covariates | 5 nearest villages within 8 km, symmetric union, degree capped at 7 (closest kept); H = 1 if more than 1/3 of neighbours are treated |
| `compute_inclusion_prob.py` | Monte Carlo first- and second-order exposure probabilities | 100,000 draws, seed 42 |
| `tune_pna.py` | PNA hyperparameter search | objective: R² and average CI width on the analysis sample (revenue) |
| `GRR-analysis.py` | estimates for HT, Hájek, linear, GAT and PNA variants | per-outcome `PNA_CONFIGS`; wage bill config labelled "best found with seed search; use --seed 50039" |
| `earlier-attempts/` | first and second versions of the analysis and tuning scripts | kept for provenance only |

Expected run order: `preprocess.py`, then `compute_inclusion_prob.py`, then `tune_pna.py` (optional), then `GRR-analysis.py --outcome revenue_per_ent` (and `wagebill_per_ent --seed 50039`).

## Missing before this study can be rerun

1. **Raw data.** `preprocess.py` expects `../replication_materials/analysisdata/GE_VillageLevel_ECMA.dta` and `village_buffers_hhs_ge.dta` from the Egger et al. (2022) replication package. They are not on this machine or in S3.
2. **Intermediate files and results.** There are no `processed_data/`, `inclusion_probabilities/` or `results/` outputs, so Tables 3 and 7–10 and Figure 2 cannot be checked against archived numbers.
3. **The seed search itself.** Only the selected seed (50039) survives. The search loop and its other seeds were not recovered.
4. **Table and figure scripts.** No scripts produce the paper's tables or the propensity histograms.

## Issues found in the recovered code (relevant to the AISTATS reviews)

- **Tuning and seed selection on the realized data.** `tune_pna.py` scores configurations by average CI width on the analysis sample. The wage bill seed was chosen by a seed search. This is exactly the circularity the meta-review raises, so the rebuttal should not describe the protocol otherwise.
- **The design used for the probabilities.** `compute_inclusion_prob.py` holds each sublocation's realized saturation fixed and draws villages independently, with P(A = 1) = 2/3 (high saturation) or 1/3 (low). So inference is conditional on the first stage, with Bernoulli assignment within it. It is not the two-stage design itself. Same-sublocation pairs are listed as dependent, but they are independent under this simulation. Also, Δ in `GRR-analysis.py` is `G @ G.T > 0`, the two-hop network only. These points bear on Reviewer XK3s's question about the saturation design.
- **The degree cap versus Table 7.** The cap is 7 neighbours. Table 7 reports a degree range up to 8 and an average of 7.0, with 2,285 edges (2 × 2,285 / 653 = 7.0 without self-loops). Check whether Table 7 counts the village itself; this needs the data to resolve.
