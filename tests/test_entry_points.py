"""Smoke tests for packaging, runner path resolution and config state handling.

These run without Spark/Java. Spark behaviour is validated on Databricks DEV.
"""

import importlib
import json
import sys

import pytest

TRAINING_TASKS = [
    "training/preprocess/t01_customer_preprocessing.py",
    "training/preprocess/t02_rfm_analysis.py",
    "training/preprocess/t03_customer_segmentation.py",
    "training/preprocess/t04_product_preprocessing.py",
    "training/preprocess/t05_mobi_probability.py",
    "training/preprocess/t06_product_transformation.py",
    "training/train/t07_model_selection.py",
    "training/train/t08_model_optimization.py",
    "training/train/t09_product_recommendation.py",
    "training/postprocess/t10_post_processing.py",
]


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """Fresh configuration module whose artifact root / state file live in tmp_path."""
    monkeypatch.setenv("ME_REC_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.delenv("ME_REC_STATE_PATH", raising=False)
    from recommendation_system_exp.config import configuration

    return importlib.reload(configuration)


def test_me_rec_runner_is_callable() -> None:
    from recommendation_system_exp.runner import run

    assert callable(run)


@pytest.mark.parametrize("task", TRAINING_TASKS)
def test_runner_resolves_every_job_task(task) -> None:
    from recommendation_system_exp.runner import resolve_task

    assert resolve_task(task).is_file()


def test_runner_rejects_unknown_task() -> None:
    from recommendation_system_exp.runner import resolve_task

    with pytest.raises(FileNotFoundError):
        resolve_task("training/1. customer data preprocessing.py")


def test_jobs_yml_tasks_exist() -> None:
    from pathlib import Path

    import yaml

    jobs = yaml.safe_load(Path("resources/jobs.yml").read_text())["resources"]["jobs"]
    paths = [
        t["python_wheel_task"]["parameters"][1]
        for job in jobs.values()
        for t in job["tasks"]
    ]
    assert sorted(paths) == sorted(TRAINING_TASKS)


def test_shipped_config_parses() -> None:
    from recommendation_system_exp.config import load_config

    assert load_config().project_name == "recommendation_system_exp"


def test_yaml_config_and_get_config(cfg) -> None:
    assert cfg.getConfig("core", "day") == 2
    assert cfg.getConfig("product_data_preprocessing", "cport_used_cols")
    assert cfg.getConfig("post_processing", "tnc_fpath").endswith(".csv")


def test_set_config_persists_across_processes(cfg, tmp_path) -> None:
    cfg.setConfig("core", "today", "2026-10-30")
    cfg.setConfig("prediction", "fe_fpath")  # legacy: None -> "None"
    cfg.setConfig("customer_data_preprocessing", "marital_status", "widow,divorced")

    state = json.loads((tmp_path / "state" / "config_state.json").read_text())
    assert state["core"]["today"] == "2026-10-30"

    reloaded = importlib.reload(cfg)  # simulates the next job task
    assert reloaded.getConfig("core", "today") == "2026-10-30"
    assert reloaded.getConfig("prediction", "fe_fpath") == "None"
    assert reloaded.getConfig("customer_data_preprocessing", "marital_status") == [
        "widow",
        "divorced",
    ]


def test_support_function_latest_artifact_fallback(cfg, tmp_path, monkeypatch) -> None:
    for d in ("age_bucketizer_20260101", "age_bucketizer_20260301"):
        (tmp_path / "pipeline" / "bucketizer" / d).mkdir(parents=True)
    sys.modules.pop("support_function", None)
    import support_function

    import recommendation_system_exp.runner  # noqa: F401  (puts package dir on sys.path)

    assert support_function.getConfig("prediction", "age_bucketizer_fpath") == (
        f"{tmp_path}/pipeline/bucketizer/age_bucketizer_20260301"
    )
