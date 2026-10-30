"""Project-level config."""

from importlib.resources import files

import yaml
from pydantic import BaseModel


class ProjectConfig(BaseModel):
    """Minimum viable config."""

    project_name: str


def load_config() -> ProjectConfig:
    raw_yaml = files("me_recomendation.config").joinpath("config.yaml").read_text()
    return ProjectConfig.model_validate(yaml.safe_load(raw_yaml))
