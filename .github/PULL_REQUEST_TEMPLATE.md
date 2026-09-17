## Description

Briefly describe what this PR changes and why.

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would change existing behavior)
- [ ] Documentation update
- [ ] New model addition
- [ ] Refactor / code quality improvement

## Checklist

- [ ] I have read [`CONTRIBUTING.md`](../CONTRIBUTING.md)
- [ ] My code follows the project's style guidelines (`make lint` passes)
- [ ] I have added tests that prove my fix is effective or my feature works
- [ ] New and existing unit tests pass locally (`make test`)
- [ ] I have updated relevant documentation (`README.md`, `docs/`, docstrings)
- [ ] I have added an entry to `CHANGELOG.md` under "Unreleased"
- [ ] I have set `random_state=42` (or reused the module constant) for any
      new source of randomness

## If This Adds a New Model

- [ ] Implements `BaseModel.build_estimator()` and `BaseModel.param_grid()`
- [ ] Has a corresponding `configs/<model>.yaml`
- [ ] Registered in `src/models/__init__.py`
- [ ] Covered by tests in `tests/test_models.py`

## Related Issues

Closes #

## Screenshots / Output (if applicable)

Paste relevant benchmark output, plots, or terminal output here.
