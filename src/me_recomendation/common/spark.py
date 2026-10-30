"""SparkSession factory."""

from pyspark.sql import SparkSession


def get_spark() -> SparkSession:
    return SparkSession.builder.getOrCreate()
