# Contributing to ML Ensemble Benchmarking Framework

Thank you for your interest in contributing! This document outlines the
process for contributing code, reporting bugs, and proposing new features.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Adding a New Model](#adding-a-new-model)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By
participating, you are expected to uphold it.

## Getting Started

1. Fork the repository and clone your fork:
   ```bash
   git clone https://github.com/<your-username>/ml-ensemble-benchmarking.git
   cd ml-ensemble-benchmarking
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   make install
   ```
3. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

- Keep changes focused — one logical change per pull request.
- Write or update tests for any behavior you add or change.
- Run the full test suite locally before opening a PR: `make test`.
- Run linting before committing: `make lint` (or `make format` to auto-fix).

## Coding Standards

- Follow [PEP 8](https://peps.python.org/pep-0008/); formatting is enforced
  with `black` and linted with `flake8` (see `Makefile`).
- All public functions and classes require docstrings (Google style, as
  used throughout `src/`).
- Use type hints on all new function signatures.
- Prefer raising specific, module-level exceptions (e.g. `ModelError`,
  `FeatureSelectionError`) over bare `Exception`.
- Set `random_state=42` (or use the module-level `RANDOM_STATE` constant)
  anywhere randomness is involved, to keep results reproducible.

## Adding a New Model

The framework is designed for extensibility via the `BaseModel` abstract
class. To add a new model:

1. Run `make add-model NAME=your_model` to scaffold the files, or manually:
   - Create `src/models/your_model.py`, subclassing `BaseModel` and
     implementing `build_estimator()` and `param_grid()`.
   - Create `configs/your_model.yaml`, following the pattern in
     `configs/random_forest.yaml`.
2. Register the model in `src/models/__init__.py`.
3. Add unit tests in `tests/test_models.py` covering fit/predict and the
   parameter grid.
4. Update `README.md`'s results table and `docs/methodology.md` if the
   model changes the benchmarking methodology.

See [`docs/architecture.md`](docs/architecture.md) for how models plug into
the broader pipeline.

## Testing Requirements

- All new code must include corresponding tests under `tests/`.
- Tests must pass with `pytest tests/ -v`.
- Aim to test both the "happy path" and expected failure modes (e.g. a
  `pytest.raises(...)` block for invalid input).

## Pull Request Process

1. Ensure `make test` and `make lint` both pass.
2. Update documentation (`README.md`, `docs/`) if your change affects usage.
3. Add an entry to `CHANGELOG.md` under an "Unreleased" heading.
4. Fill out the pull request template completely.
5. A maintainer will review your PR; please address review feedback
   promptly. PRs may be squash-merged once approved.

Questions? Open a [discussion](https://github.com/mawawazir/ml-ensemble-benchmarking/issues)
or reach out via the contact details in `README.md`.
