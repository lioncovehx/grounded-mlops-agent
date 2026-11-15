"""Project config: which ML project the agent manages (mlops-agent.yaml).

Missing values fall back to generic defaults; secrets and service URLs stay in env vars.
"""

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_FILE = Path(__file__).resolve().parents[1] / "mlops-agent.yaml"


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DvcConfig(Section):
    repo_dir: Path = Path(".")
    version_tag_pattern: str = Field("data-*", description="Git tags that mark dataset versions")
    default_dataset: str | None = Field(None, description="Used when a tool isn't given a path")


class ProvenanceTags(Section):
    """MLflow run tag names that link a run to its data and code."""

    data_hash: str = "data.md5"
    data_version: str = "data.tag"
    data_commit: str = "data.commit"
    data_path: str = "data.path"
    code_commit: str = "git.commit"
    quality_gate: str = "quality_gate"


class Aliases(Section):
    candidate: str = "candidate"
    deployed: str = "champion"


class MlflowConfig(Section):
    experiment: str | None = None
    registered_model: str | None = None
    primary_metric: str | None = None
    aliases: Aliases = Aliases()
    model_max_age_days: int = 30
    tags: ProvenanceTags = ProvenanceTags()
    feature_stats_artifact: str = "feature_stats.json"


class AirflowConfig(Section):
    default_dag: str | None = None
    allowed_dags: list[str] = Field(default_factory=list, description="DAGs the agent may trigger")


class KubernetesConfig(Section):
    namespace: str = "default"
    default_deployment: str | None = None
    pod_selector: str | None = Field(None, description="Label selector for the inference Pods")
    model_version_label: str = "mlops/model-version"


class MonitoringConfig(Section):
    prediction_event: str = "prediction"
    drift_threshold_std: float = 0.5
    min_predictions: int = 30


class Config(Section):
    project: str = "ml-project"
    dvc: DvcConfig = DvcConfig()
    mlflow: MlflowConfig = MlflowConfig()
    airflow: AirflowConfig = AirflowConfig()
    kubernetes: KubernetesConfig = KubernetesConfig()
    monitoring: MonitoringConfig = MonitoringConfig()


def load(path: Path) -> Config:
    """Read and validate a config file. Relative repo paths are resolved from the file's folder."""
    config = Config.model_validate(yaml.safe_load(path.read_text()) or {})
    config.dvc.repo_dir = (path.parent / config.dvc.repo_dir).resolve()
    return config


@lru_cache(maxsize=1)
def settings() -> Config:
    path = Path(os.environ.get("MLOPS_AGENT_CONFIG") or DEFAULT_FILE)
    if path.is_file():
        return load(path)
    config = Config()
    config.dvc.repo_dir = Path.cwd()
    return config


def required(value: str | None, what: str) -> str:
    """A value a tool needs: from its arguments or the config, else a clear error."""
    if not value:
        raise ValueError(f"no {what} given and none configured in mlops-agent.yaml")
    return value
