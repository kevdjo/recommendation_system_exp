"""Task 2: transform data passed from extract."""

import argparse

from pyspark.sql import functions as F

from me_recomendation.common.spark import get_spark
from me_recomendation.common.task_values import get_task_value
from me_recomendation.config import load_config


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--raw_catalog", default=None)
    parser.add_argument("--raw_schema", default=None)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--experiment_id", default=None)
    args, _ = parser.parse_known_args(argv)
    return args


def run() -> None:
    args = _parse()
    config = load_config()
    spark = get_spark()

    input_table = get_task_value(spark, task_key="extract", key="output_table")
    expected_rows = get_task_value(spark, task_key="extract", key="row_count")
    if input_table is None:
        raise SystemExit("transform: no `extract` task value found; was extract run?")

    output_table = f"{args.catalog}.{args.schema}.transform_sample"

    df = spark.read.table(input_table)
    actual_rows = df.count()
    if actual_rows != expected_rows:
        raise SystemExit(
            f"transform: row count mismatch -- extract reported "
            f"{expected_rows}, got {actual_rows} rows from {input_table}"
        )

    (
        df.withColumn("value_doubled", F.col("value") * 2)
        .write.mode("overwrite")
        .saveAsTable(output_table)
    )

    print(
        f"transform OK: wrote {actual_rows} rows to {output_table} "
        f"(project={config.project_name})"
    )


if __name__ == "__main__":
    run()
