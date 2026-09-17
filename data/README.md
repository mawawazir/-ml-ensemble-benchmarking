# Data Directory

This directory holds datasets used by the benchmarking pipeline. Raw data
files are **not committed to version control** (see `.gitignore`) — only
the directory structure is preserved via `.gitkeep` files.

## Structure

```
data/
├── raw/          # Original, immutable data dumps (CSV)
└── processed/    # Cleaned, feature-engineered data ready for modeling
```

## Getting Data

The pipeline supports two modes:

1. **Bring your own data**: Place a CSV file in `data/raw/` and point
   `configs/default.yaml`'s `data.raw_path` at it. The file must include
   the column named in `data.target_column` (default: `target`).

2. **Synthetic demonstration data**: If `data.raw_path` is left as `null`
   (the default), `src/data/load_data.py` generates a reproducible,
   moderately imbalanced binary-classification dataset with
   `random_state=42` via `sklearn.datasets.make_classification`. This
   lets the entire pipeline — feature engineering, tuning, benchmarking,
   reporting — run end-to-end without a proprietary dataset, while still
   producing consistent, reproducible results.

## Dataset Characteristics (Synthetic Mode)

| Property | Value |
|---|---|
| Rows | 50,000 (configurable via `data.n_samples`) |
| Numeric features | 20 (configurable via `data.n_features`) |
| Categorical features | 2 (`category_region`, `category_channel`) |
| Target | Binary, ~15% positive class (imbalanced) |
| Missing values | ~3% of `feature_0`, injected intentionally |
| Random seed | 42 |

## Processed Data

Files written to `data/processed/` by the pipeline (e.g. post feature
engineering) follow the naming convention:

```
data/processed/<stage>_<split>.csv
# e.g. data/processed/engineered_train.csv
```

These are also excluded from version control to keep the repository size
small; they are fully reproducible by re-running the pipeline against the
same raw data and configuration.
