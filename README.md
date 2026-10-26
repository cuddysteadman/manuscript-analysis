# High density extracellular recordings from brain organoids enable automated high throughput functional assay


## Repository layout

```
.
├── run_training.ipynb        # stage 2: UNSORTED, SORTED, and PRUNED training
├── run_ablation.ipynb        # stage 3: matched feature ablation
├── null_distribution.ipynb   # stage 4: UNSORTED null distribution
├── prep_test_data.ipynb      # stage 5: score the drug-test and insertion recordings
├── config/
│   └── training_config.json  # input features, block settings, LSTM architecture, training settings
├── src/manuscript_analysis/  # installable package behind the notebooks
├── input/                    # data from Zenodo plus the MATLAB inputs (see Input data)
├── results/                  # trained runs, exported analyses, null distribution, paper-era model
├── output/                   # figures, tables, and scored test data
├── pyproject.toml            # package metadata and dependencies
└── requirements.txt          # pinned dependency versions
```

### `config/training_config.json`

The training, ablation, and null workflows all read this file. It has three blocks:

- `DATA_CFG`: the four input features, the two classes (`CTRL-CO`, `AD-CO`), `n`
  (spikes per block, 500), `min_chunks` and `max_chunks` (fewest and most blocks kept
  per recording, 10 and 250), `cutoff_time` (spikes at or after this sample index are
  dropped), and `exclude_sessions`.
- `UNSORTED_ARCH_LSTM_CFG`: LSTM layer widths, dense layer size, dropout, L2
  regularization, and layer normalization.
- `TRAIN_CFG`: epochs, batch size, early-stopping patience, learning rate, and the
  plateau learning-rate schedule.

Runs that already have complete artifacts are skipped (see [Workflows](#workflows)),
so use an empty output folder if you change these settings. 

### `src/manuscript_analysis/`

| Path | Role |
|---|---|
| `contract.py` | Fixed design constants and `find_project_root` |
| `data.py` | Prepares recordings, builds spike blocks, selects the cohort per pipeline, creates split manifests, scales features |
| `modeling.py` | LSTM model, training of one pipeline and seed, block- and session-level metrics |
| `artifacts.py` | Writes, reads, and validates per-run artifacts; consolidates run exports |
| `io.py` | Atomic JSON and CSV writers, zip helper |
| `workflows/` | `canonical.py`, `ablation.py`, and `null.py` hold the logic behind `run_training.ipynb`, `run_ablation.ipynb`, and `null_distribution.ipynb` |
| `figures/` | Figure notebooks, shared plotting modules (`style.py`, `export.py`, `accuracy.py`, `ablation_forest.py`, `feature_medians.py`, `score_hist.py`), and the MATLAB scripts in `matlab/` |
| `tables/` | Notebooks for Tables 1 and 2 |

### `results/`

- `matched_three_pipeline_run/` and `matched_ablation_run/`: `runs.zip` (models,
  histories, scalers, per-run metrics) and `analysis_exports/` (consolidated CSVs
  that the figure notebooks read). Unzipped `runs/` folders are gitignored.
- `null-distribution/`: the null distribution CSVs and JSON summaries.
- `old_code/saved_data/`: the original pre-cross validation model created on seed 42
  (`Model_CTRL-CO-DAY150-AD-CO-DAY150.keras`), with
  the arrays it was built from (`X_all.npy` in Git LFS, `session_ids.npy`,
  `session_labels.npy`, `split_labels.npy`). Only Figure 6C uses this model as it would be impossible to
  display such data with all cross-validation models, so the original model for the September 2025 preprint was used 
  instead.

### `output/`

`output/figures/<fig>/` and `output/tables/<id>/` hold each figure's or table's
source data as `.csv` plus `.pdf`, `.png`, and `.svg` exports with and without
labels. The committed copies (Figures 6B, 6C, 8A, S5; Tables 1, 2, 6A) are the
manuscript versions, so you can compare your regenerated files against them.
`output/test_data/` holds the stage 5 scores for the drug-test and insertion
recordings.

## Install

Requires Python 3.11 or newer. From the repository root, create a virtual environment, activate
it, and install the package.

macOS, Linux, or WSL2:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[notebook]"
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[notebook]"
```

### GPU support

TensorFlow 2.11 and later cannot use a GPU on native Windows, so training there runs
on the CPU and is slower. To train on an
NVIDIA GPU, use WSL2 or Linux (see the
[TensorFlow install guide](https://www.tensorflow.org/install/pip)).
### Input data
The input pickles can be found on Zenodo at DOI: `10.5281/zenodo.23665779`. After downloading, they must be reorganized
as follows:
```
input/
├── batch_units_depth_struct.mat
├── EXP43_CTRL-CO-DAY150-7_2ND-REC_NOV24.spikes.cellinfo.mat
├── fig4_combined_raster_features.csv
├── s6_firing_rate.csv
├── drug_test/
│   ├── AD-CO-DAY150-MKAD2-77-10NM-30DAYS.pkl
│   ├── AD-CO-DAY150-MKAD2-77-100NM-30DAYS.pkl
│   └── OL-AD-CO-DAY150.pkl
├── insertion/
│   └── insertion_dfs_Nov24.pkl
├── pruned/
│   ├── AD-CO-DAY150.pkl
│   └── CTRL-CO-DAY150.pkl
├── sorted/
│   ├── AD-CO-DAY150.pkl
│   └── CTRL-CO-DAY150.pkl
└── unsorted/
    ├── AD-CO-DAY150.pkl
    └── CTRL-CO-DAY150.pkl
```
Note especially that the files on Zenodo entitled `PRUNED_{class}.pkl`, `SORTED_{class}.pkl`, `UNSORTED_{class}.pkl`
must be renamed to all just be `{class}.pkl` once placed in the correct folder. (They were renamed as such because Zenodo
does not allow duplicate file names / directory organization.)

#### Git LFS 
The two `.mat` files and two `.csv` files directly under `input/` are
committed to this repository as ordinary files, so a plain clone includes them. The `.pkl` files
are not in git at all (`input/*` is gitignored apart from `.mat` and `.csv`), so they must come from Zenodo.
Git LFS is used for one file only, the paper-era array `X_all.npy` under `results/old_code/saved_data/`
(needed for Figure 6C). Install [Git LFS](https://git-lfs.com) and run `git lfs install` before cloning,
or run `git lfs pull` afterwards; without it that file is a small text pointer and the Figure 6C notebook
fails when loading it.

### Extract the archived runs

The trained models, histories, scalers, and per-run metrics are committed as zip
archives. The extracted `runs/` folders are gitignored, so a fresh clone has only
the archives. Extract them in place (the archives already contain a top-level
`runs/` folder), with the virtual environment active:

```bash
python -m zipfile -e results/matched_three_pipeline_run/runs.zip results/matched_three_pipeline_run/
python -m zipfile -e results/matched_ablation_run/runs.zip results/matched_ablation_run/
```

This produces:

```
results/
├── matched_three_pipeline_run/runs/{unsorted,sorted,pruned}/seed0 ... seed24/
└── matched_ablation_run/runs/{all,drop_*,only_*}/seed0 ... seed24/
```

`results/null-distribution/null_distribution.zip` only bundles CSV/JSON files
that already sit unzipped in the same folder, so it needs no extraction.

Skip this step if you intend to retrain from scratch (see the note at the end of
[Workflows](#workflows)).

## Workflows

| # | Stage                                                                                                                     | Needs                                                                                           | Writes                                                               |
|---|---------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| 0 | Install, download `input/` from Zenodo, `git lfs pull`                                                                    |                                                                                                 |                                                                      |
| 1 | Extract the archived runs (above), or skip to retrain                                                                     |                                                                                                 | `results/*/runs/`                                                    |
| 2 | `run_training.ipynb`: UNSORTED, SORTED, and PRUNED training                                                               | `input/`                                                                                        | `results/matched_three_pipeline_run/` (`runs/`, `analysis_exports/`) |
| 3 | `run_ablation.ipynb`: matched feature ablation                                                                            | `input/unsorted/`; stage 2 `runs/unsorted/` and `analysis_exports/pipeline_splits_25_seeds.csv` | `results/matched_ablation_run/`                                      |
| 4 | `null_distribution.ipynb`: UNSORTED null distribution (set `PROFILE = "FULL"`; the default `"SMOKE"` is a 2 x 2 test run) | `input/`                                                                                        | `results/null-distribution/`                                         |
| 5 | `prep_test_data.ipynb`: score the drug-test and insertion recordings                                                      | `input/drug_test/`, `input/insertion/`; stage 2 `runs/unsorted/`                                | `output/test_data/`                                                  |
| 6 | Figure and table notebooks (below)                                                                                        | see below                                                                                       | `output/figures/`, `output/tables/`                                  |

Stage 6 notebooks live in `src/manuscript_analysis/figures/` and
`src/manuscript_analysis/tables/`:

| Output | Notebook | Reads |
|---|---|---|
| Figures 5A-5D | `figures/5_feature_session_medians.ipynb` | `input/unsorted/`, `input/drug_test/` |
| Table 1 | `tables/1_feature_statistics_per_class.ipynb` | `input/` |
| Table 2 | `tables/2_spike_yield.ipynb` | stage 2 `analysis_exports/` |
| Figure 6A | `figures/6A_session_table.ipynb` | `input/unsorted/` |
| Figure 6B | `figures/6B_block_auc_boxplot.ipynb` | stage 2 `analysis_exports/` |
| Figure 6C | `figures/6C_test_sessions_overview.ipynb` | original model created pre-cross validation and arrays in `results/old_code/saved_data/` (Git LFS) |
| Figure 7A | `figures/7A_session_auc_boxplot.ipynb` | stage 2 `analysis_exports/` |
| Figure 7B | `figures/7B_test_classification_hist.ipynb` | stage 3 `analysis_exports/` |
| Figure 7C | `figures/7C_drug_sessions_predictions_hist.ipynb` | stage 5 `output/test_data/` |
| Figure 8A | `figures/8A_insertion_trajectory.ipynb` | `input/insertion/`; stage 2 `runs/unsorted/` |
| Figure 8C | `figures/8C_test_sessions_predictions_hist.ipynb` | stage 5 `output/test_data/` |
| Figure S5 | `figures/S5_feature_ablation.ipynb` | stage 3 `analysis_exports/`; stage 4 `null_distribution.csv` |
| Classification table for 7B, 7C, 8C | `figures/sup_7B_7C_8C.ipynb` | stage 3 `analysis_exports/`; stage 5 `output/test_data/` |

Figures 3, 4, and S6 are MATLAB scripts in `src/manuscript_analysis/figures/matlab/`
that read `input/batch_units_depth_struct.mat`,
`input/EXP43_CTRL-CO-DAY150-7_2ND-REC_NOV24.spikes.cellinfo.mat`,
`input/fig4_combined_raster_features.csv`, and `input/s6_firing_rate.csv`. They
are independent of stages 1 to 5.

The `analysis_exports/` folders and `output/test_data/` are committed, so the
figure notebooks that read only those (6B, 7A, 7B, 7C, 8C, S5, Table 2) run
without extraction or retraining. Stage 3, stage 5, and Figure 8A are the ones
that need the extracted `runs/` folders.

Runs with complete artifacts are skipped, so extracting the archives first
makes the training notebooks re-consolidate the saved results instead of
retraining. Resuming a partly finished run on different hardware mixes results
from both stacks. Use an empty output folder for a single-stack run.

## Reproducibility

Training uses fixed seeds and `tf.config.experimental.enable_op_determinism()`,
so rerunning on the same hardware with the same package versions reproduces results. A different hardware or
software stack (e.g. CPU vs. GPU, CUDA/cuDNN version, TensorFlow build, operating
system) changes floating-point rounding. Because training stops early on
validation AUC, that can change the stopping epoch and final model, so
individual seed results shift slightly and summary statistics will not match the
submitted values exactly. 

All manuscript values come from the original submission runs. The models, training history, scalers, and metrics for 
each run of each seed in each permutation are available for each pipeline as a .zip file.
