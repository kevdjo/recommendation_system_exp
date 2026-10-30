"""Databricks job task-values helpers."""

from typing import Any

from pyspark.sql import SparkSession


def set_task_value(spark: SparkSession, key: str, value: Any) -> None:
    """Set a task value on the current task."""
    from pyspark.dbutils import DBUtils

    DBUtils(spark).jobs.taskValues.set(key=key, value=value)


def get_task_value(
    spark: SparkSession,
    task_key: str,
    key: str,
    default: Any = None,
) -> Any:
    """Read a task value produced by a prior task in the same job run."""
    from pyspark.dbutils import DBUtils

    return DBUtils(spark).jobs.taskValues.get(
        taskKey=task_key, key=key, default=default, debugValue=default
    )
