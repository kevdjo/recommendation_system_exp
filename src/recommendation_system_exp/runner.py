"""
Universal task execution harness for recommendation_system_exp.
Executes migrated training and inference tasks without requiring file renames.

Each Databricks job task is a separate process running this entry point once per
task script, so there is no shared Python state between steps. Two things below
exist only to bridge that gap:
  - CLI args are written to env vars (not passed as function args) because the
    task scripts import config.configuration at module level; the env vars have
    to be set *before* that import happens, not handed to it directly.
  - The script is executed with runpy instead of a normal import because some
    task scripts guard their body with `if __name__ == "__main__"` while
    others run top-level unconditionally; runpy.run_path(run_name="__main__")
    runs both kinds correctly, whereas a plain import would skip the guarded
    ones. (Standardizing every script to always run its body on import would
    let this move to a plain importlib.import_module call instead.)

Usage (Databricks python_wheel_task, entry point `me_rec_runner`):
    --task <path relative to the package, e.g. training/preprocess/t01_customer_preprocessing.py>
    [--catalog <UC catalog>]        sets the session default catalog, so the legacy
                                    two-level names (mldb.*, dm_analytics.*) resolve in it
    [--artifact_root <path>]        overrides core.artifact_root (models, pipelines, state)
    [--output_db <schema>]          schema the legacy mldb.* names resolve to
    [--source_db <schema>]          schema the legacy dm_analytics.* names resolve to
    [--today <YYYY-MM-DD>]          run date for task 01 (backfill); empty = today
    [--log_level <LEVEL>]           DEBUG/INFO/WARNING/ERROR (default INFO, or ME_REC_LOG_LEVEL)
Any other arguments are forwarded to the task script in sys.argv.
"""

import argparse
import logging
import os
import runpy
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
log = logging.getLogger(__name__)

# Ensure the package root is in sys.path so tasks can import support_function
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def resolve_task(task_input: str) -> Path:
    """--task is always given relative to the package root, e.g.
    'training/preprocess/t01_customer_preprocessing.py' (see resources/jobs.yml)."""
    candidate = Path(task_input)
    path = candidate if candidate.is_absolute() else PACKAGE_ROOT / candidate
    if not path.exists():
        raise FileNotFoundError(f"Task script not found: {path}")
    return path


def run() -> None:
    parser = argparse.ArgumentParser(
        description="Task runner for recommendation_system_exp"
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Path of the task script relative to the package "
        "(e.g. 'training/preprocess/t01_customer_preprocessing.py')",
    )
    parser.add_argument("--catalog", default=None)
    parser.add_argument("--artifact_root", default=None)
    parser.add_argument("--output_db", default=None)
    parser.add_argument("--source_db", default=None)
    parser.add_argument("--today", default=None)
    parser.add_argument("--log_level", default=None)
    args, unknown = parser.parse_known_args()

    from recommendation_system_exp.common.log import setup_logging

    candidate = resolve_task(args.task)
    setup_logging(args.log_level, task=candidate.stem)

    # Must be set before the task imports the config module (read at import time).
    for arg, env in (
        ("artifact_root", "ME_REC_ARTIFACT_ROOT"),
        ("output_db", "ME_REC_OUTPUT_DB"),
        ("source_db", "ME_REC_SOURCE_DB"),
        ("today", "ME_REC_TODAY"),
    ):
        if getattr(args, arg):
            os.environ[env] = getattr(args, arg)

    # Serverless needs a UC Volume path for Spark ML's DataFrame caching during .fit().
    if "SPARKML_TEMP_DFS_PATH" not in os.environ:
        from recommendation_system_exp.config.configuration import ARTIFACT_ROOT

        os.environ["SPARKML_TEMP_DFS_PATH"] = f"{ARTIFACT_ROOT}/tmp"

    from recommendation_system_exp.common.spark import get_spark

    spark = get_spark()
    # The legacy code relies on non-ANSI casts (e.g. numeric == "null"); Serverless
    # defaults to ANSI on, which raises instead of returning NULL.
    spark.conf.set("spark.sql.ansi.enabled", "false")

    if args.catalog:
        spark.sql(f"USE CATALOG `{args.catalog}`")
        log.info(
            "Default catalog: %s", spark.sql("SELECT current_catalog()").first()[0]
        )

    log.info("Executing task script: %s", candidate)
    sys.argv = [str(candidate)] + unknown
    runpy.run_path(str(candidate), run_name="__main__")
    log.info("Finished task script: %s", candidate.name)


if __name__ == "__main__":
    run()
