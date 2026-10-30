"""Verify each shipped task's `run` is importable + shipped config parses."""


def test_extract_run_is_callable() -> None:
    from recommendation_system_exp.training.extract import run

    assert callable(run)


def test_transform_run_is_callable() -> None:
    from recommendation_system_exp.training.transform import run

    assert callable(run)


def test_score_init_run_is_callable() -> None:
    from recommendation_system_exp.inference.init import run

    assert callable(run)


def test_me_rec_runner_is_callable() -> None:
    from recommendation_system_exp.runner import run

    assert callable(run)


def test_model_uri_shape() -> None:
    from recommendation_system_exp.common.mlflow import model_uri

    assert model_uri("cat", "sch", "m", "champion") == "models:/cat.sch.m@champion"


def test_entry_points_tolerate_extra_job_params() -> None:
    from recommendation_system_exp.training.extract import _parse as parse_extract
    from recommendation_system_exp.training.transform import _parse as parse_transform

    from recommendation_system_exp.inference.init import _parse as parse_score_init

    argv = [
        "--catalog",
        "c",
        "--schema",
        "s",
        "--raw_catalog",
        "rc",
        "--raw_schema",
        "rs",
        "--model_name",
        "m",
        "--model_alias",
        "a",
        "--future_param",
        "x",
    ]
    for parse in (parse_extract, parse_transform, parse_score_init):
        args = parse(argv)
        assert args.catalog == "c"
        assert args.schema == "s"


def test_shipped_config_parses() -> None:
    from recommendation_system_exp.config import load_config

    cfg = load_config()
    assert cfg.project_name == "me_recomendation"


def test_yaml_config_and_get_config() -> None:
    from recommendation_system_exp.config.configuration import getConfig

    assert getConfig("core", "day") == 2
    assert isinstance(getConfig("core", "list_table"), list)
    assert len(getConfig("core", "list_table")) > 0


def test_support_function_importable() -> None:
    from recommendation_system_exp.support_function import getConfig as sf_getConfig

    assert sf_getConfig("core", "day") == 2
