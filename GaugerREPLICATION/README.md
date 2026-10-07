# GAUGER numerical replication supplement

This lightweight package collects the available implementation and archived
synthetic results for **GNN-Assisted Generalized Regression for Network
Interference (GAUGER)**. It does not contain credentials, private datasets,
repository history, notebooks, or unrelated experiment outputs.

**Scope:** the package supports rerunning the proposed-method synthetic
experiments and inspecting archived outputs corresponding to the paper. It is
not yet a complete reproduction package for all numerical results: external
GNN baselines, the student-network experiment, and the cash-transfer application
require the additional materials listed below. No experiments, dependency
installation, or runtime tests were performed when assembling this package.

## Contents

- `GRR.py`: regression adjustment and GNN embedding interfaces. The source
  uses the historical name GRR for the paper's GER/GAUGER implementation.
- `GAT-embedder.py`, `PNA-embedder.py`: graph outcome models.
- `estimators.py`, `propensity.py`, `variance_estimators.py`: estimator,
  exposure-probability, and inference routines.
- `dgp.py`: shared exposure routine imported by `propensity.py`.
- `dgp-2/dgp-2.py`: synthetic potential outcomes with neighbor-covariate
  spillover, used by both experiment drivers.
- `dgp-2/simulation-out1-dgp2-parallel.py`: sparse-network synthetic study.
- `dgp-2/simulation-out2-dgp2-parallel.py`: medium-density synthetic study.
- `reference-results/`: three selected runs, each with a summary CSV,
  configuration JSON, and per-randomization NPZ arrays.
- `requirements.txt`: dependencies collected from the included source imports.
- `MANIFEST.sha256`: integrity checksums for the package contents.

The sparse runner was assembled from the repository's working medium-density
runner, changing only the network configuration and experiment/output names.
This is intentional: the repository's current original sparse parallel script
had been narrowed to a linear-only ablation and disabled GNN training. The
packaged sparse runner retains all eleven supported methods, including that
ablation. It is not claimed to be the exact historical source snapshot used
for the archived sparse run. Other Python files are copied unchanged. The
medium runner's introductory docstring mentions degrees 7/13; its executable
configuration is 5/10, matching the paper. The original repository is untouched.

## Environment

Use a separate Python environment and run commands from this package's root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The dependencies are NumPy, SciPy, pandas, scikit-learn, tqdm, joblib,
PyTorch, and PyTorch Geometric. Versions are intentionally not invented:
the original experimental environment's exact package versions were not
available in the collected files. Authors should supply a tested version lock
before claiming exact environment reproduction. For GPU execution, use a
PyTorch installation compatible with the machine's CUDA environment.

The authors report using an NVIDIA A100 GPU with 80 GB of GPU memory.
The included parallel drivers assign workers to GPU 0 and default to three
workers. Set `N_JOBS = 1` in the driver to avoid concurrent GPU training if
memory is limited. The GNN code falls back to CPU when CUDA is unavailable,
but CPU execution may be substantially slower. A fixed GPU device name does
not itself ensure deterministic training.

## Run the synthetic experiments

```bash
# Main sparse-network study.
python dgp-2/simulation-out1-dgp2-parallel.py

# Appendix medium-density study.
python dgp-2/simulation-out2-dgp2-parallel.py
```

These are full runs, not smoke tests. Both use 2,000 units, three exposure
levels, treated fraction 1/3, contrast `d1=0` versus `d2=2`, 500 propensity
Monte Carlo draws, and 100 outcome randomizations. Average/maximum degrees
are 3/9 for the main study and 5/10 for the appendix study. All model settings
are explicit in `GAT_CONFIG` and `PNA_CONFIG`; the data-generation and
propensity seeds are 123, 1234, 999, and 111 respectively. The generated
finite-population effect is computed and printed by each run.

New outputs are written under `dgp-2/results/`, relative to the current working
directory. Each run saves `<experiment>_<timestamp>_summary.csv`,
`_config.json`, and `_full.npz`. Reference outputs are stored separately and
will not be overwritten. Failed methods can produce NaNs; check `n_valid` and
the console output before interpreting a result.

The historical source is not fully seed-deterministic: treatment assignment
calls `default_rng(seed=None)` even after setting the legacy NumPy seed,
and the drivers do not fix PyTorch seeds. Accordingly, fresh runs need not
match archived estimates bit for bit. The package preserves this behavior
rather than silently altering the original experiment. The archived numeric
arrays provide the original draws for inspecting the reported point summaries.

## Archived results and paper mapping

| Paper result | Reference run prefix | What is available |
| --- | --- | --- |
| Main synthetic study, `tab:synthetic_full` | `out1_dgp2_parallel_20260128_054224` | Ten HT, Hajek, linear, GAT, and PNA method summaries |
| Main synthetic linear-raw row | `out1_dgp2_linear_no_cal_20260128_194808` | Separate `linear-no-cal` ablation |
| Medium-density study, `tab::simu_out2_results` | `out2_dgp2_parallel_20260128_222447` | All eleven implemented method summaries |

The following mapping is based on the archived numerical values. It records
the paper labels without changing the implementations:

| Source method key | Paper row label |
| --- | --- |
| `ht` | HT |
| `hajek` | Hajek |
| `linear-no-cal` | linear-raw |
| `linear` | linear-linear |
| `lasso` | linear-pred |
| `gat-pred` | GAT-raw |
| `gat-linear` | GAT-linear |
| `gat-no-harm` | GAT-pred |
| `pna-pred` | PNA-raw |
| `pna-linear` | PNA-linear |
| `pna-no-harm` | PNA-pred |

The summary CSVs contain bias, sample SD (`ddof=1`),
`sqrt(bias**2 + sd**2)` as the source's RMSE convention, valid-replication
counts, coverage, and power. Coverage is the fraction of available 95% CIs
containing the generated effect; power is the fraction excluding zero.
NPZ keys are `results_<method>`, `coverage_<method>`, and `power_<method>`.
The JSONs record available run metadata, not a complete software/hardware or
random-seed manifest.

The drivers and archived CSVs do not export the Monte Carlo uncertainty
half-widths displayed after `+/-` in the paper's tables. The exact original
table-formatting/error-bar procedure must be supplied separately; this
package does not substitute a guessed uncertainty calculation.

## Author completion needed before full-paper reproduction

1. **External baselines:** add the adapted NetEstimator, GCN-Deconfounder,
   and RRNet implementations, configurations, and outputs for the synthetic
   and semi-synthetic comparisons. They were not found in this checkout.
2. **Student-network study:** add the ICPSR 37070 acquisition/access procedure,
   permitted data or derived inputs, preprocessing, DGP parameters, and the
   200-randomization driver. The manuscript's dataset identifier is not an
   executable data pipeline.
3. **Cash-transfer study:** add the permitted dataset/access instructions,
   village-network and exposure construction, outcome preprocessing, analysis
   code, and selected configurations, including the stated wage-model seed
   50039. `Interference-CashTransfer/` is empty in this checkout, despite a
   recorded Git submodule entry; its code was not available to collect.
4. **Exact environment and reporting:** supply the historical dependency
   versions and the Monte Carlo error-bar/table-generation procedure. Review
   the method-label mapping and driver settings against final paper edits.
5. **Release review:** confirm anonymization and redistribution permissions
   for all materials added to the supplement. No private or third-party
   datasets were included by default.

The archived synthetic summaries agree with the corresponding displayed
point estimates at the paper's rounding precision. This is a static source
and output comparison, not a successful rerun or a certification that every
paper result can currently be reproduced.
