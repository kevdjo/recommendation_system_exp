"""Logging setup, called once by the runner (tasks only use logging.getLogger)."""

import logging
import os
import sys

FORMAT = "%(asctime)s %(levelname)s [%(task)s] %(name)s %(message)s"


class _TaskFilter(logging.Filter):
    """Adds the %(task)s field, so lines from `__main__` task scripts stay identifiable."""

    def __init__(self, task: str) -> None:
        super().__init__()
        self.task = task

    def filter(self, record: logging.LogRecord) -> bool:
        record.task = self.task
        return True


def setup_logging(level: str | None = None, task: str = "-") -> None:
    """Configure the root logger. Safe to call more than once (replaces our handler).

    level: defaults to the ME_REC_LOG_LEVEL env var, then INFO.
    task: label shown in every line, e.g. the task script's file name.
    """
    level = (level or os.environ.get("ME_REC_LOG_LEVEL") or "INFO").upper()
    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, "_me_rec", False):
            root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler._me_rec = True  # type: ignore[attr-defined]
    handler.setFormatter(logging.Formatter(FORMAT))
    handler.addFilter(_TaskFilter(task))
    root.addHandler(handler)
    root.setLevel(level)
