"""
Helper utilities for Databricks migration runtime compatibility.
"""

from typing import Any


def init_params(
    argv: list[str] | None = None, defaults: list[Any] | None = None
) -> None:
    """Compatibility no-op for legacy ws4conv scripts."""
