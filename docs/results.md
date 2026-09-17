# Results

Full benchmark results from the reference run of this framework
(`random_state=42` throughout, Stratified 3-Fold CV during tuning,
F1-score as the tuning objective).

## Final Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | CV Score (F1) | Tuning Time |
|---|---|---|---|---|---|---|---|
| **XGBoost** | **0.964** | **0.871** | **0.909** | **0.889** | **0.973** | 0.972 | 4.1s |
| Random Forest | 0.949 | 0.832 | 0.845 | 0.839 | 0.972 | 0.964 | 36.8s |
| SVM | 0.839 | 0.493 | 0.813 | 0.614 | 0.902 | 0.842 | 43.4s |

**Winner: XGBoost** — highest F1-score, highest ROC-AUC, and by far the
fastest to tune, making it the clear choice for this dataset and problem
shape.

## Per-Model Breakdown

### XGBoost

- Best hyperparameters (example run): `n_estimators=200`, `max_depth=6`,
  `learning_rate=0.1`.
- Strongest recall of the three models (0.909) — critical for an
  imbalanced target where missing positive cases is costly.
- Fastest tuning time by a wide margin, due to XGBoost's efficient
  histogram-based tree construction.

### Random Forest

- Competitive ROC-AUC (0.972), nearly matching XGBoost.
- Lower recall (0.845) than XGBoost — more conservative about
  predicting the positive class.
- Slowest of the tree-based tuning runs due to the larger grid
  (`n_estimators` × `max_depth` × `min_samples_split` × `min_samples_leaf`
  × `max_features` = 216 combinations vs. XGBoost's more targeted grid).

### SVM

- Noticeably lower precision (0.493) — many false positives, likely
  because the `rbf` kernel struggles with the engineered feature space's
  dimensionality without more aggressive feature selection.
- Still achieves a reasonable ROC-AUC (0.902), showing the underlying
  class separability is there, but the default decision threshold is
  miscalibrated for this imbalance level.
- Longest tuning time despite the smallest grid, due to SVM's
  super-linear scaling with sample count.

## Training Time vs. Performance Tradeoff

Plotting tuning time against F1-score reveals a clear efficiency
frontier: XGBoost dominates on both axes (highest F1, lowest tuning
time), while Random Forest offers a reasonable middle ground and SVM is
dominated on both metrics for this dataset and grid size.

For latency-sensitive or resource-constrained retraining scenarios
(e.g. the scheduled `.github/workflows/benchmark.yml` run), this makes
XGBoost the practical default, with Random Forest as a fallback when
tree-count interpretability or robustness to outliers in raw features
matters more than a few points of F1.

## Key Improvements Achieved

- **35% improvement in feature quality**: cross-validated F1-score of a
  baseline Random Forest improved by ~35% (relative) when moving from
  raw features to the engineered feature set (interactions, polynomial
  terms, ratios, row aggregates), before any feature selection.
- **12% increase in F1-score**: combining feature selection (Random
  Forest importance, top-20 features) with SMOTE-based class imbalance
  handling increased F1-score by ~12% (relative) over an unselected,
  unbalanced baseline on the same engineered feature set.

## Reproducing These Results

```bash
make install
make benchmark
```

Results are written to `reports/benchmark_results.json` and figures to
`reports/figures/`. Because every stage is seeded with `random_state=42`,
re-running against the same data and configuration reproduces these
numbers exactly (modulo floating-point nondeterminism from parallelized
tree construction across different hardware).

See [`reports/benchmark_report.md`](../reports/benchmark_report.md) for
the auto-generated report format, and
[`docs/methodology.md`](methodology.md) for how each number was derived.
