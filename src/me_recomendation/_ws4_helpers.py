"""
Helper utilities for Databricks migration runtime compatibility.
"""

from typing import Any, List, Optional


def init_params(argv: Optional[List[str]] = None, defaults: Optional[List[Any]] = None) -> None:
    """Compatibility no-op for legacy ws4conv scripts."""
    pass
