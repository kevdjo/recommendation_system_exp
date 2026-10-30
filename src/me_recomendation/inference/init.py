"""Entry task for score_job."""

import argparse

from me_recomendation.common.spark import get_spark
from me_recomendation.config import load_config


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--raw_catalog", default=None)
    parser.add_argument("--raw_schema", default=None)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--model_alias", default=None)
    args, _ = parser.parse_known_args(argv)
    return args


def run() -> None:
    args = _parse()
    config = load_config()
    spark = get_spark()
    print(
        f"score_init OK: project={config.project_name} "
        f"target={args.catalog}.{args.schema} "
        f"spark={spark.version}"
    )


if __name__ == "__main__":
    run()
