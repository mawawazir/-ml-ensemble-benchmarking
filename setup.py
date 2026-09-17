"""Package setup for the ML Ensemble Benchmarking Framework."""

from pathlib import Path

from setuptools import find_packages, setup

THIS_DIR = Path(__file__).resolve().parent
README = (THIS_DIR / "README.md").read_text(encoding="utf-8") if (THIS_DIR / "README.md").exists() else ""


def _read_requirements() -> list:
    req_path = THIS_DIR / "requirements.txt"
    if not req_path.exists():
        return []
    lines = req_path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


setup(
    name="ml-ensemble-benchmarking",
    version="1.0.0",
    description=(
        "A systematic benchmarking framework for comparing ensemble machine "
        "learning methods (Random Forest, XGBoost, SVM) with automated "
        "hyperparameter tuning, feature engineering, and class imbalance handling."
    ),
    long_description=README,
    long_description_content_type="text/markdown",
    author="Mawa Wazir",
    license="MIT",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.9",
    install_requires=_read_requirements(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "ml-benchmark=src.models.benchmark:Benchmark",
        ],
    },
    include_package_data=True,
)
