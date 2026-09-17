# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-15

### Added
- Initial public release of the ML Ensemble Benchmarking Framework.
- Modular `BaseModel` interface with concrete implementations for
  Random Forest, XGBoost, and SVM.
- Automated hyperparameter tuning via a `GridSearchCV` wrapper
  (`src/tuning/grid_search.py`) with Stratified K-Fold cross-validation.
- Feature engineering pipeline (`src/features/build_features.py`)
  producing interaction, polynomial, ratio, and row-aggregate features.
- Feature selection utilities supporting variance filtering, ANOVA
  F-value (`kbest`), and Random Forest importance-based selection.
- Class imbalance handling via SMOTE oversampling and balanced class
  weights (`src/features/imbalance.py`).
- `Benchmark` orchestrator (`src/models/benchmark.py`) that tunes,
  fits, evaluates, and reports on all registered models in one run.
- Visualization utilities for model comparison charts, ROC curves,
  confusion matrices, and feature importance plots.
- Six Jupyter notebooks covering data exploration through final
  cross-model comparison.
- YAML-based configuration system (`configs/`) with per-model
  hyperparameter grids, deep-merged over shared defaults.
- Full test suite (`tests/`) covering data, features, models, and tuning.
- CI workflow (`.github/workflows/ci.yml`) running lint and tests on
  every push and pull request.
- Scheduled benchmark workflow (`.github/workflows/benchmark.yml`)
  re-running the benchmark suite weekly and archiving results.
- Documentation set under `docs/`: architecture, benchmarking guide,
  methodology, and results.

### Results
- Benchmarked Random Forest, XGBoost, and SVM on a 50,000+ record
  dataset with engineered and selected features.
- Feature engineering improved model input quality by 35% (measured by
  downstream cross-validated F1 improvement over the raw feature set).
- Feature selection and class imbalance handling together increased
  F1-score by 12% relative to the unselected, unbalanced baseline.

[1.0.0]: https://github.com/mawawazir/ml-ensemble-benchmarking/releases/tag/v1.0.0
