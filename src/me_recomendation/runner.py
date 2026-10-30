"""
Universal task execution harness for recommendation_system_exp.
Executes migrated training and inference tasks without requiring file renames.
"""

import argparse
import runpy
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent

# Ensure the package root is in sys.path so tasks can import support_function and _ws4_helpers
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def run() -> None:
    parser = argparse.ArgumentParser(
        description="Task runner for recommendation_system_exp"
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Path or relative name of the task script to execute (e.g., 'training/1. customer data preprocessing.py')",
    )
    args, unknown = parser.parse_known_args()

    task_input = args.task
    candidate = Path(task_input)

    if not candidate.is_absolute():
        if (PACKAGE_ROOT / candidate).exists():
            candidate = PACKAGE_ROOT / candidate
        elif (PACKAGE_ROOT / "training" / candidate).exists():
            candidate = PACKAGE_ROOT / "training" / candidate
        elif (PACKAGE_ROOT / "inference" / candidate).exists():
            candidate = PACKAGE_ROOT / "inference" / candidate
        else:
            raise FileNotFoundError(
                f"Task script not found: {task_input} (checked in {PACKAGE_ROOT})"
            )

    print(f"--> Executing task script: {candidate}")
    sys.argv = [str(candidate)] + unknown
    runpy.run_path(str(candidate), run_name="__main__")
    print(f"<-- Finished task script: {candidate.name}")


if __name__ == "__main__":
    run()
