# Benchmarking Guide

A practical, step-by-step guide to running and extending benchmarks with
this framework.

## Prerequisites

```bash
git clone https://github.com/mawawazir/ml-ensemble-benchmarking.git
cd ml-ensemble-benchmarking
make install
```

## Running the Default Benchmark

The simplest invocation runs all three models (Random Forest, XGBoost,
SVM) against the configured (or synthetic) dataset:

```bash
make benchmark
```

This is equivalent to:

```python
from src.data.load_data import load_raw_data, train_test_split_data
from src.data.preprocess import preprocess_pipeline
from src.features.build_features import engineer_features
from src.features.selection import select_features
from src.features.imbalance import apply_smote
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.models.svm_model import SVMModel
from src.models.benchmark import Benchmark

# 1. Load and split
df = load_raw_data()  # uses configs/default.yaml settings
X_train, X_test, y_train, y_test = train_test_split_data(df)

# 2. Preprocess (impute, encode, scale)
X_train, X_test, _ = preprocess_pipeline(X_train, X_test)

# 3. Engineer features
X_train = engineer_features(X_train)
X_test = engineer_features(X_test)

# 4. Select features (fit on train, apply same columns to test)
X_train, selected_cols = select_features(X_train, y_train, method="importance", k=20)
X_test = X_test[selected_cols]

# 5. Handle class imbalance (training data only)
X_train, y_train = apply_smote(X_train, y_train)

# 6. Benchmark
benchmark = Benchmark(
    models=[RandomForestModel(), XGBoostModel(), SVMModel()],
    cv_folds=5,
    scoring="f1",
)
results = benchmark.run(X_train, y_train, X_test, y_test)

# 7. Inspect results
print(benchmark.summary_dataframe())
benchmark.save_report("reports/benchmark_results.json")
```

## Running a Single Model

To tune and evaluate just one model:

```python
from src.tuning.grid_search import tune_model
from src.models.xgboost_model import XGBoostModel

model = XGBoostModel()
result = tune_model(model, X_train, y_train, cv_folds=5, scoring="f1")

print(result.best_params)
print(result.best_score)
```

## Using a Custom Dataset

1. Place your CSV in `data/raw/your_data.csv`. It must contain the
   target column (default name: `target`).
2. Update `configs/default.yaml`:
   ```yaml
   data:
     raw_path: "data/raw/your_data.csv"
     target_column: "target"
   ```
3. Re-run `make benchmark`.

## Customizing Hyperparameter Grids

Edit the relevant model's YAML config, e.g. `configs/random_forest.yaml`:

```yaml
model:
  param_grid:
    n_estimators: [100, 300, 500]
    max_depth: [10, 20, null]
```

Or override at call time without touching the config file:

```python
model = RandomForestModel(param_grid_override={"n_estimators": [500]})
```

## Choosing a Feature Selection Strategy

```python
# Random Forest importance (default) — captures non-linear effects
X_train, cols = select_features(X_train, y_train, method="importance", k=20)

# ANOVA F-value — faster, model-agnostic baseline
X_train, cols = select_features(X_train, y_train, method="kbest", k=20)
```

## Choosing an Imbalance Strategy

```python
from src.features.imbalance import apply_smote, apply_class_weights

# Option A: oversample the training set
X_train, y_train = apply_smote(X_train, y_train)

# Option B: compute weights and pass to a model that supports class_weight
weights = apply_class_weights(y_train)
model = RandomForestModel(params={"class_weight": weights})
```

Only apply one strategy at a time — combining SMOTE with class weights
tends to over-correct for imbalance and can hurt precision.

## Generating Report Figures

```python
from src.visualization.plots import (
    plot_model_comparison,
    plot_roc_curves,
    plot_confusion_matrices,
    plot_feature_importance,
)

summary = benchmark.summary_dataframe()
plot_model_comparison(summary, metrics=["accuracy", "f1", "roc_auc"])

fitted = {name: r.fitted_model.estimator for name, r in benchmark.results.items()}
plot_roc_curves(fitted, X_test, y_test)
plot_confusion_matrices(fitted, X_test, y_test)

rf_estimator = benchmark.results["random_forest"].fitted_model.estimator
plot_feature_importance(X_train.columns.tolist(), rf_estimator.feature_importances_)
```

All figures are saved to `reports/figures/` by default.

## Interpreting `reports/benchmark_results.json`

```json
{
  "random_forest": {
    "metrics": {"accuracy": 0.95, "precision": 0.83, "recall": 0.85, "f1": 0.84, "roc_auc": 0.97, "confusion_matrix": [[...]]},
    "best_params": {"n_estimators": 200, "max_depth": 20, "...": "..."},
    "cv_score": 0.96,
    "tuning_time_seconds": 36.75,
    "training_time_seconds": 0.0
  }
}
```

- `metrics` — held-out test-set performance.
- `best_params` — the winning hyperparameter combination from `GridSearchCV`.
- `cv_score` — the best cross-validated score seen during tuning (on
  training data, not the held-out test set).
- `tuning_time_seconds` — total wall-clock time spent searching the grid.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `TuningError: GridSearchCV failed` | Check the param grid for invalid combinations (e.g. `kernel="linear"` with a `gamma` value — `gamma` is ignored but shouldn't error; check estimator-specific constraints). |
| `ImbalanceHandlingError` from SMOTE | Minority class too small for the requested `k_neighbors`; the function auto-reduces `k_neighbors`, but an extremely small minority class (<2 samples) will still fail. |
| Low SVM performance vs. tree models | Expected on non-linear, high-dimensional data — try `kernel="rbf"` (default) and confirm scaling was applied in preprocessing. |
| Slow grid search | Reduce the grid size, lower `cv_folds`, or set `n_jobs=-1` (default) to use all cores. |
