"""Smoke test for `make connect-test`.

Confirms Databricks Connect can reach the dev workspace and run a distributed query.

Serverless path:
  export DATABRICKS_SERVERLESS_COMPUTE_ID=auto
  make connect-test

Classic cluster path:
  export DATABRICKS_CLUSTER_ID=<cluster-id>
  make connect-test
"""

from databricks.connect import DatabricksSession


def main() -> None:
    spark = DatabricksSession.builder.getOrCreate()
    df = spark.range(5).toDF("n")
    rows = df.collect()
    print(f"Connected OK. Spark version: {spark.version}")
    print(f"Sample rows: {[r['n'] for r in rows]}")


if __name__ == "__main__":
    main()
