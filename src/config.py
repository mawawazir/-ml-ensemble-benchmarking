"""
Configuration loading and validation utilities.

Provides a thin, typed wrapper around YAML configuration files so the
rest of the codebase can rely on a consistent, validated configuration
object instead of raw dictionaries.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


class ConfigError(Exception):
    """Raised when a configuration file is missing, malformed, or invalid."""


@dataclass
class Config:
    """
    Container for a fully-resolved configuration.

    Attributes:
        raw: The underlying dictionary parsed from YAML.
        path: The filesystem path the configuration was loaded from, if any.
    """

    raw: Dict[str, Any] = field(default_factory=dict)
    path: Optional[Path] = None

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """
        Retrieve a nested value using dot notation.

        Args:
            dotted_key: Key path such as ``"model.random_forest.n_estimators"``.
            default: Value returned if the key path does not exist.

        Returns:
            The resolved value, or ``default`` if any segment is missing.

        Example:
            >>> cfg = Config({"data": {"test_size": 0.2}})
            >>> cfg.get("data.test_size")
            0.2
        """
        node: Any = self.raw
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted_key: str, value: Any) -> None:
        """
        Set a nested value using dot notation, creating intermediate dicts.

        Args:
            dotted_key: Key path such as ``"tuning.cv_folds"``.
            value: Value to assign.
        """
        parts = dotted_key.split(".")
        node = self.raw
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def merge(self, other: Dict[str, Any]) -> "Config":
        """
        Return a new Config with ``other`` deep-merged on top of this one.

        Args:
            other: Dictionary whose keys take precedence on conflict.

        Returns:
            A new, merged Config instance (this instance is left untouched).
        """
        merged = copy.deepcopy(self.raw)
        _deep_merge(merged, other)
        return Config(raw=merged, path=self.path)

    def to_dict(self) -> Dict[str, Any]:
        """Return a deep copy of the underlying configuration dictionary."""
        return copy.deepcopy(self.raw)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def __contains__(self, key: str) -> bool:
        return key in self.raw


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    """Recursively merge ``override`` into ``base`` in place."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def load_config(path: Optional[str | Path] = None) -> Config:
    """
    Load a YAML configuration file into a :class:`Config` object.

    Args:
        path: Path to a YAML file. If ``None``, loads ``configs/default.yaml``.

    Returns:
        A populated :class:`Config` instance.

    Raises:
        ConfigError: If the file does not exist or cannot be parsed as YAML.
    """
    resolved = Path(path) if path is not None else DEFAULT_CONFIG_PATH

    if not resolved.exists():
        raise ConfigError(f"Configuration file not found: {resolved}")

    try:
        with open(resolved, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML config at {resolved}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"Configuration at {resolved} must be a mapping at the top level.")

    return Config(raw=data, path=resolved)


def load_model_config(model_name: str, config_dir: Optional[str | Path] = None) -> Config:
    """
    Load and merge a model-specific config on top of the default config.

    Args:
        model_name: One of ``"random_forest"``, ``"xgboost"``, ``"svm"``.
        config_dir: Directory containing the YAML files. Defaults to ``configs/``.

    Returns:
        Merged :class:`Config` (default config overridden by the model config).

    Raises:
        ConfigError: If either config file is missing or malformed.
    """
    base_dir = Path(config_dir) if config_dir is not None else DEFAULT_CONFIG_PATH.parent
    default_cfg = load_config(base_dir / "default.yaml")
    model_cfg_path = base_dir / f"{model_name}.yaml"

    if not model_cfg_path.exists():
        raise ConfigError(f"No configuration found for model '{model_name}' at {model_cfg_path}")

    with open(model_cfg_path, "r", encoding="utf-8") as handle:
        model_data = yaml.safe_load(handle) or {}

    return default_cfg.merge(model_data)
