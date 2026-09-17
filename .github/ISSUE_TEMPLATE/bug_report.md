---
name: Bug Report
about: Report a reproducible bug in the benchmarking framework
title: "[BUG] "
labels: bug
assignees: ''
---

## Description

A clear and concise description of what the bug is.

## To Reproduce

Steps to reproduce the behavior:

1. Configuration used (paste relevant `configs/*.yaml` section or override)
2. Command run (e.g. `make benchmark`, or a Python snippet)
3. Observed error or incorrect output

```
Paste full traceback / output here
```

## Expected Behavior

A clear and concise description of what you expected to happen.

## Environment

- OS: [e.g. Ubuntu 22.04, macOS 14]
- Python version: [e.g. 3.11.4]
- Package versions: run `pip freeze | grep -E "scikit-learn|xgboost|imbalanced-learn|pandas|numpy"`
- Installation method: [`make install`, `pip install -e .`, other]

## Dataset

- [ ] Using the synthetic demo dataset (default)
- [ ] Using a custom dataset (describe shape/size if relevant, without
      sharing sensitive data)

## Additional Context

Add any other context about the problem here (screenshots of plots,
`reports/benchmark_results.json` excerpts, etc.).
