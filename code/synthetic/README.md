# Synthetic studies (paper Table 1 and Appendix Table 6)

Run every command from this folder (`code/synthetic/`). Outputs go to `<repo>/results/synthetic/dgp2/`, which also holds the archived runs below.

| Paper result | Script | Archived output (`results/synthetic/dgp2/`) |
| --- | --- | --- |
| Table 1, all rows except linear-raw and the pure-GNN baselines | `dgp-2/simulation-out1-dgp2-parallel.py` | `out1_dgp2_parallel_20260128_054224_*` |
| Table 1, linear-raw row | `dgp-2/simulation-out1-linear-no-cal.py` | `out1_dgp2_linear_no_cal_20260128_194808_*` |
| Appendix Table 6 (medium density) | `dgp-2/simulation-out2-dgp2-parallel.py` | `out2_dgp2_parallel_20260128_222447_*` |
| Appendix G, GAT search | `tuning/tune-GAT-embedder*.py` | none |

Each driver runs B = 100 randomizations with 3 workers on GPU 0 (`N_JOBS` in the script). A run takes about 40 to 50 minutes on one A100.

## Provenance

- The sparse driver `simulation-out1-dgp2-parallel.py` is the original version, saved before it was narrowed to the linear-only ablation. It was recovered from the editor's local history (Cursor, saved 2026-01-28 19:39). Its method list (10 methods) matches the archived run. Its GAT and PNA settings are identical to the medium-density driver.
- `simulation-out1-linear-no-cal.py` is the repository's narrowed version, which runs only the `linear-no-cal` ablation.
- The core modules were byte-identical to the former repository-root copies (now removed) and to `GaugerREPLICATION/`.
- The tuning scripts were copied from the repository root and from `legacy/`. Each gained a single `sys.path` line so it runs from this folder.
- Older scripts with no counterpart here (the DGP-1 studies, DGP-2 out3/out4, notebooks) are kept in `../legacy/` for provenance; their imports assume the old flat layout.

## Method keys and paper labels

| Code key | Paper label | Meaning |
| --- | --- | --- |
| `linear-no-cal` | linear-raw | per-exposure OLS on covariates, plugged into AIPW (no GER) |
| `linear` | linear-linear | GER on covariates |
| `lasso` | linear-pred | GER on covariates, LASSO-penalized solve |
| `gat-pred`, `pna-pred` | GAT-raw, PNA-raw | AIPW with the network's predictions (beta = 1, no GER) |
| `gat-linear`, `pna-linear` | GAT-linear, PNA-linear | GER on learned representations (Paradigm 2) |
| `gat-no-harm`, `pna-no-harm` | GAT-pred, PNA-pred | GER on the network's predictions (Paradigm 1) |

## Known gaps

- The Table 1 pure-GNN baselines (NetEstimator, GCN-Deconfounder, RRNet) have no code in this repository.
- Runs are not seed-deterministic. Treatment draws use `default_rng(None)`, and no PyTorch seed is set, so fresh runs reproduce the archived results only in distribution.
- The archives do not record how the `±` Monte Carlo half-widths in the tables were computed.
- `tuning/tune-pna-grr.py` tunes an end-to-end PNA-GRR variant (`method='pna'`). That is not the PNA configuration used in the drivers.
- The Appendix G PNA search space, tuned by CI width, comes from the Kenya study. That code is in `../Interference-CashTransfer/analysis-GAUGER/tune_pna.py`.
