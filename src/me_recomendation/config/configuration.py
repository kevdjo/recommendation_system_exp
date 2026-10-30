"""
Configuration module for Recommendation System.
Loaded from config.yaml.
"""

from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def _load_yaml_config() -> dict[str, Any]:
    if not _CONFIG_PATH.exists():
        return {}
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


CONFIG_MAP: dict[str, Any] = _load_yaml_config()


def getConfig(key1: str, key2: str, sc=None) -> Any:
    """
    Retrieve typed configuration value given section (key1) and item name (key2).
    Drop-in compatibility with legacy tasks.
    """
    if key1 in CONFIG_MAP and isinstance(CONFIG_MAP[key1], dict):
        if key2 in CONFIG_MAP[key1]:
            return CONFIG_MAP[key1][key2]
    raise KeyError(f"Configuration key ({key1}, {key2}) not found in config.yaml.")


def setConfig(key1: str, key2: str, value: Any = None, backup: bool = False) -> None:
    """
    Update configuration value in-memory and write back to config.yaml if needed.
    """
    if key1 not in CONFIG_MAP or not isinstance(CONFIG_MAP[key1], dict):
        CONFIG_MAP[key1] = {}
    CONFIG_MAP[key1][key2] = value


if __name__ == "__main__":
    print(f"Loaded config sections: {list(CONFIG_MAP.keys())}")
    for section, val in CONFIG_MAP.items():
        if isinstance(val, dict):
            print(f"  [{section}] ({len(val)} items)")
