# Architecture

This document describes how the pieces of the ML Ensemble Benchmarking
Framework fit together, from raw data to final benchmark report.

## Pipeline Overview

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   configs/  │────▶│  src/data/  │────▶│ src/features/│────▶│ src/tuning/ │
│  (YAML)     │     │ load +      │     │ engineer +   │     │ GridSearchCV│
│             │     │ preprocess  │     │ select +     │     │             │
│             │     │             │     │ imbalance    │     │             │
└─────────────┘     └─────────────┘     └──────────────┘     └──────┬──────┘
                                                                     │
                                                                     ▼
┌──────────────┐     ┌──────────────┐     ┌───────────────────────────┐
│  reports/    │◀────│ src/         │◀────│      src/models/          │
│  figures +   │     │ visualization│     │      Benchmark            │
│  benchmark_  │     │              │     │  (orchestrates fit/eval   │
│  results     │     │              │     │   across all models)      │
└──────────────┘     └──────────────┘     └───────────────────────────┘
```

## Component Responsibilities

### `configs/`
Each model has its own YAML file (`random_forest.yaml`, `xgboost.yaml`,
`svm.yaml`) that is deep-merged on top of `default.yaml`. This keeps
shared settings (data paths, CV folds, output directories) in one place
while letting each model override only what it needs — chiefly its
hyperparameter grid.

### `src/data/`
- `load_data.py` reads a raw CSV (or synthesizes a reproducible demo
  dataset when none is configured) and performs the stratified
  train/test split.
- `preprocess.py` builds a `ColumnTransformer` that imputes missing
  values, one-hot encodes categoricals, and scales numeric columns. It
  is fit only on training data and applied to test data to avoid
  leakage.

### `src/features/`
- `build_features.py` derives interaction, polynomial, ratio, and
  row-aggregate features from the numeric columns.
- `selection.py` reduces dimensionality via variance filtering plus
  either Random Forest importance or ANOVA F-value (`SelectKBest`).
- `imbalance.py` rebalances the **training set only**, via SMOTE
  oversampling or balanced class weights.

### `src/models/`
- `base_model.py` defines the abstract `BaseModel` interface
  (`build_estimator`, `param_grid`, `fit`, `predict`, `predict_proba`,
  `save`/`load`) that every model implements.
- `random_forest.py`, `xgboost_model.py`, `svm_model.py` are concrete
  implementations.
- `benchmark.py`'s `Benchmark` class orchestrates the full loop: tune
  each model with `GridSearchCV`, refit with the best parameters,
  evaluate on the held-out test set, and collect results into a single
  comparison table.

### `src/tuning/`
`grid_search.py` wraps `sklearn.model_selection.GridSearchCV` with a
`StratifiedKFold` cross-validation strategy, returning a structured
`GridSearchResult` (best estimator, best params, best score, full
`cv_results_`, and timing).

### `src/evaluation/`
- `metrics.py` computes accuracy, precision, recall, F1, ROC-AUC, and
  a confusion matrix via one shared function, so every model is scored
  identically.
- `cross_validation.py` provides a standalone stratified CV scoring
  utility, independent of hyperparameter tuning, for post-hoc model
  validation.

### `src/visualization/`
`plots.py` generates the four report figures — model comparison bars,
overlaid ROC curves, side-by-side confusion matrices, and feature
importance bar charts — and saves them to `reports/figures/`.

## Data Flow Contract

1. **No leakage**: any transformer that learns from data (imputers,
   scalers, encoders, feature selectors, SMOTE) is fit exclusively on
   the training split.
2. **Reproducibility**: every source of randomness — the synthetic
   data generator, train/test split, cross-validation folds, SMOTE, and
   every estimator — is seeded with `random_state=42`.
3. **Single source of truth for metrics**: all models are scored via
   `src/evaluation/metrics.py::compute_classification_metrics`, so
   comparisons in `reports/benchmark_report.md` are apples-to-apples.

## Extending the Architecture

Adding a new model, config, or metric touches exactly one file each:

| To add... | Touch... |
|---|---|
| A new model | `src/models/<name>.py` + `configs/<name>.yaml` |
| A new feature transform | `src/features/build_features.py` |
| A new evaluation metric | `src/evaluation/metrics.py` |
| A new report figure | `src/visualization/plots.py` |

See [`docs/methodology.md`](methodology.md) for the reasoning behind each
stage's default settings, and [`CONTRIBUTING.md`](../CONTRIBUTING.md) for
the step-by-step process of adding a new model.
