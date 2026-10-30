"""Task 1: extract sample data and write to UC."""

import argparse

from pyspark.sql import functions as F

from me_recomendation.common.spark import get_spark
from me_recomendation.common.task_values import set_task_value
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

    output_table = f"{args.catalog}.{args.schema}.extract_sample"

    df = spark.range(100).toDF("id").withColumn("value", F.rand(seed=42))
    df.write.mode("overwrite").saveAsTable(output_table)

    row_count = spark.read.table(output_table).count()

    set_task_value(spark, "output_table", output_table)
    set_task_value(spark, "row_count", row_count)

    print(
        f"extract OK: wrote {row_count} rows to {output_table} "
        f"(project={config.project_name})"
    )


if __name__ == "__main__":
    run()
