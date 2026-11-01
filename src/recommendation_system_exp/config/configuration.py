"""
Configuration module for Recommendation System.

Values are loaded from config.yaml (shipped in the wheel). Runtime changes made
with setConfig() are persisted to a JSON state file under ARTIFACT_ROOT so that
later tasks (separate processes / separate Databricks job tasks) see them.
This restores the legacy CDSW behaviour, where setConfig() wrote back to
configuration.csv.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def _load_yaml_config() -> dict[str, Any]:
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {_CONFIG_PATH}; check that it is packaged in the wheel."
        )
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


CONFIG_MAP: dict[str, Any] = _load_yaml_config()

# Root for all persisted artifacts (Spark ML pipelines/models, state, dictionaries).
# Precedence: env var (set by runner from --artifact_root) > config.yaml > legacy relative path.
ARTIFACT_ROOT: str = (
    os.environ.get("ME_REC_ARTIFACT_ROOT")
    or CONFIG_MAP.get("core", {}).get("artifact_root")
    or "./recommendation_system"
).rstrip("/")

# Schemas the legacy two-level table names resolve to (mldb.*, dm_analytics.*).
# Precedence: env var (set by runner from --output_db / --source_db) > config.yaml > legacy name.
OUTPUT_DB: str = (
    os.environ.get("ME_REC_OUTPUT_DB")
    or CONFIG_MAP.get("core", {}).get("output_db")
    or "mldb"
)
SOURCE_DB: str = (
    os.environ.get("ME_REC_SOURCE_DB")
    or CONFIG_MAP.get("core", {}).get("source_db")
    or "dm_analytics"
)


def _expand_artifact_root(node: Any) -> Any:
    """Resolve "${artifact_root}" in config.yaml string values (reference/hard-rule CSV paths)."""
    if isinstance(node, dict):
        return {k: _expand_artifact_root(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand_artifact_root(v) for v in node]
    if isinstance(node, str):
        return node.replace("${artifact_root}", ARTIFACT_ROOT)
    return node


CONFIG_MAP = _expand_artifact_root(CONFIG_MAP)

STATE_PATH: str = (
    os.environ.get("ME_REC_STATE_PATH") or f"{ARTIFACT_ROOT}/state/config_state.json"
)


def _read_state() -> dict[str, dict[str, Any]]:
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except FileNotFoundError:
        return {}


def _write_state(state: dict[str, dict[str, Any]]) -> None:
    try:
        os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
        tmp = STATE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(tmp, STATE_PATH)
    except OSError as e:  # e.g. local run without the Volume mounted
        log.warning("Could not persist config state to %s: %s", STATE_PATH, e)


# Overlay persisted state on top of the YAML defaults.
for _section, _items in _read_state().items():
    CONFIG_MAP.setdefault(_section, {})
    if isinstance(CONFIG_MAP[_section], dict) and isinstance(_items, dict):
        CONFIG_MAP[_section].update(_items)


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
    Update a configuration value in memory and persist it for later tasks.

    Legacy semantics preserved:
      - value=None is stored as the string "None" (support_function.getConfig
        treats "None" as "use the latest artifact on disk").
      - a comma-separated string assigned to a key whose default is a list is
        stored as a list (configuration.csv had a 'list' type column).
    """
    if value is None:
        value = "None"
    current = (
        CONFIG_MAP.get(key1, {}).get(key2)
        if isinstance(CONFIG_MAP.get(key1), dict)
        else None
    )
    if isinstance(current, list) and isinstance(value, str):
        value = [v for v in value.split(",") if v != ""]

    if key1 not in CONFIG_MAP or not isinstance(CONFIG_MAP[key1], dict):
        CONFIG_MAP[key1] = {}
    CONFIG_MAP[key1][key2] = value

    state = _read_state()
    state.setdefault(key1, {})[key2] = value
    _write_state(state)


if __name__ == "__main__":
    print(f"Loaded config sections: {list(CONFIG_MAP.keys())}")
    print(f"ARTIFACT_ROOT={ARTIFACT_ROOT}  STATE_PATH={STATE_PATH}")
    for section, val in CONFIG_MAP.items():
        if isinstance(val, dict):
            print(f"  [{section}] ({len(val)} items)")
