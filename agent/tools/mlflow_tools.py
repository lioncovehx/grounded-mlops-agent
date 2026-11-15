"""MLflow tools: experiments, runs, the model registry and run-to-data lineage, read-only.

Experiment, model, metric, alias and provenance tag names come from the project config, and
every one can also be passed as an argument, so the tools work on any MLflow server.
"""

import os
from datetime import UTC, datetime

from pydantic import Field

from agent.config import required, settings
from agent.registry import Args, NoArgs, tool
from agent.tools import dvc_tools

# Fail fast when MLflow is down instead of retrying for minutes.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")


def client():
    from mlflow import MlflowClient  # imported lazily: mlflow is slow to import

    return MlflowClient()


def iso(ms: int | None) -> str | None:
    return datetime.fromtimestamp(ms / 1000, UTC).isoformat() if ms else None


def provenance(tags: dict) -> dict:
    """The run's provenance tags, keyed by role (data_hash, data_commit, ...)."""
    names = settings().mlflow.tags.model_dump()
    return {role: tags[name] for role, name in names.items() if name in tags}


def summarize(run) -> dict:
    """A small, plain view of a run: what the agent needs, nothing more."""
    return {
        "run_id": run.info.run_id,
        "experiment_id": run.info.experiment_id,
        "status": run.info.status,
        "start_time": iso(run.info.start_time),
        "metrics": run.data.metrics,
        "params": run.data.params,
        "provenance": provenance(run.data.tags),
    }


def experiment_id(name_or_id: str | None) -> str:
    name_or_id = required(name_or_id or settings().mlflow.experiment, "experiment")
    c = client()
    exp = c.get_experiment_by_name(name_or_id)
    if exp is None and name_or_id.isdigit():
        exp = c.get_experiment(name_or_id)
    if exp is None:
        raise ValueError(f"no experiment named or numbered {name_or_id!r}")
    return exp.experiment_id


def model_name(model: str | None) -> str:
    return required(model or settings().mlflow.registered_model, "registered model")


def deployed_version(model: str):
    """The registry version behind the 'deployed' alias, or None if the alias isn't set."""
    try:
        return client().get_model_version_by_alias(model, settings().mlflow.aliases.deployed)
    except Exception as exc:
        if "not found" in str(exc).lower() or "RESOURCE_DOES_NOT_EXIST" in str(exc):
            return None
        raise


def source_run_id(version) -> str:
    return version.tags.get("source_run") or version.run_id


Experiment = Field(None, description="Experiment name or numeric ID; omit for the project default")
Model = Field(None, description="Registered model name; omit for the project default")


class ExperimentArgs(Args):
    experiment: str | None = Experiment
    limit: int = Field(10, ge=1, le=50, description="Most recent runs to return")


class RunArgs(Args):
    run_id: str = Field(pattern=r"^[0-9a-f]{32}$", description="32-character MLflow run ID")


class BestRunArgs(Args):
    metric: str | None = Field(None, description="Metric to rank by; omit for the primary metric")
    experiment: str | None = Experiment


class CompareRunsArgs(Args):
    run_ids: list[str] = Field(min_length=2, max_length=10, description="Run IDs to compare")


class ModelArgs(Args):
    model: str | None = Model


@tool(NoArgs)
def list_experiments(_: NoArgs) -> dict:
    """List MLflow experiments with their IDs, and the project's default experiment and model."""
    cfg = settings().mlflow
    return {
        "experiments": [
            {"experiment_id": e.experiment_id, "name": e.name}
            for e in client().search_experiments()
        ],
        "default_experiment": cfg.experiment,
        "default_registered_model": cfg.registered_model,
        "primary_metric": cfg.primary_metric,
    }


@tool(ExperimentArgs)
def list_runs(args: ExperimentArgs) -> dict:
    """List the most recent runs of an experiment, newest first, with metrics, parameters,
    dataset provenance and quality gate result."""
    runs = client().search_runs(
        [experiment_id(args.experiment)],
        order_by=["attributes.start_time DESC"],
        max_results=args.limit,
    )
    return {"runs": [summarize(r) for r in runs]}


@tool(RunArgs)
def get_run_details(args: RunArgs) -> dict:
    """Full details of one run: metrics, parameters, provenance tags and logged artifacts."""
    c = client()
    details = summarize(c.get_run(args.run_id))
    details["artifacts"] = [a.path for a in c.list_artifacts(args.run_id)]
    return details


@tool(BestRunArgs)
def get_best_run(args: BestRunArgs) -> dict:
    """Find the run with the highest value of a metric. Runs without the metric are excluded,
    never ranked. The best metric alone does not mean production-ready: check the quality gate
    and the dataset version."""
    metric = required(args.metric or settings().mlflow.primary_metric, "metric")
    runs = client().search_runs([experiment_id(args.experiment)], max_results=500)
    ranked = [r for r in runs if metric in r.data.metrics]
    if not ranked:
        logged = sorted({m for r in runs for m in r.data.metrics})
        raise ValueError(f"no run has logged metric {metric!r}; metrics logged: {logged}")
    ranked.sort(key=lambda r: r.data.metrics[metric], reverse=True)
    return {
        "metric": metric,
        "best": summarize(ranked[0]),
        "runner_up": summarize(ranked[1]) if len(ranked) > 1 else None,
        "runs_ranked": len(ranked),
        "runs_missing_metric": len(runs) - len(ranked),
    }


@tool(CompareRunsArgs)
def compare_runs(args: CompareRunsArgs) -> dict:
    """Compare several runs side by side: metrics, parameters, dataset version and gate result."""
    c = client()
    return {"runs": [summarize(c.get_run(run_id)) for run_id in args.run_ids]}


@tool(ModelArgs)
def get_registry_info(args: ModelArgs) -> dict:
    """List versions of a registered model with their aliases, source run, primary metric and
    dataset version."""
    c = client()
    name = model_name(args.model)
    metric = settings().mlflow.primary_metric
    model = c.get_registered_model(name)
    aliases_by_version: dict[str, list[str]] = {}
    for alias, version in model.aliases.items():
        aliases_by_version.setdefault(str(version), []).append(alias)

    versions = []
    for v in c.search_model_versions(f"name='{name}'"):
        run_id = source_run_id(v)
        run = c.get_run(run_id) if run_id else None
        versions.append(
            {
                "version": v.version,
                "aliases": aliases_by_version.get(str(v.version), []),
                "source_run": run_id,
                "created": iso(v.creation_timestamp),
                "metrics": {metric: run.data.metrics.get(metric)} if run and metric else None,
                "provenance": provenance(run.data.tags) if run else None,
            }
        )
    versions.sort(key=lambda v: int(v["version"]), reverse=True)
    return {"model": name, "aliases": dict(model.aliases), "versions": versions}


@tool(RunArgs)
def get_dataset_for_run(args: RunArgs) -> dict:
    """Identify the dataset version a run was trained on, verified against Git: compares the run's
    recorded data hash with the DVC pointer at the recorded data commit."""
    recorded = provenance(client().get_run(args.run_id).data.tags)
    missing = [role for role in ("data_hash", "data_commit") if role not in recorded]
    if missing:
        names = settings().mlflow.tags.model_dump()
        return {
            "run_id": args.run_id,
            "recorded": recorded,
            "verified": False,
            "finding": f"run is missing provenance tags {[names[m] for m in missing]}; "
            "its dataset cannot be verified",
        }
    path = dvc_tools.resolve_dataset(recorded.get("data_path"))
    pointer = dvc_tools.pointer_at(recorded["data_commit"], path)
    return {
        "run_id": args.run_id,
        "dataset": path,
        "recorded": recorded,
        "pointer_md5_at_data_commit": pointer["md5"],
        "verified": pointer["md5"] == recorded["data_hash"],
        "version_tag_for_hash": dvc_tools.tag_for_md5(recorded["data_hash"], path),
    }


@tool(ModelArgs)
def get_retraining_signals(args: ModelArgs) -> dict:
    """Facts for deciding whether to retrain the deployed model (the 'deployed' alias): newer data
    available, model age, and its quality gate result. Reports evidence; does not decide."""
    cfg = settings().mlflow
    name = model_name(args.model)
    version = deployed_version(name)
    if version is None:
        return {"model": name, "deployed": None, "finding": f"no {cfg.aliases.deployed!r} alias"}

    run = client().get_run(source_run_id(version))
    recorded = provenance(run.data.tags)
    path = dvc_tools.resolve_dataset(recorded.get("data_path"))
    current = dvc_tools.pointer_at("HEAD", path)
    age_days = (datetime.now(UTC).timestamp() * 1000 - run.info.start_time) / 86_400_000
    return {
        "model": name,
        "deployed_alias": cfg.aliases.deployed,
        "deployed_version": version.version,
        "source_run": run.info.run_id,
        "dataset": path,
        "trained_on": {
            "md5": recorded.get("data_hash"),
            "version_tag": recorded.get("data_version"),
        },
        "current_data": {
            "md5": current["md5"],
            "version_tag": dvc_tools.tag_for_md5(current["md5"], path),
        },
        "newer_data_available": recorded.get("data_hash") != current["md5"],
        "model_age_days": round(age_days, 1),
        "max_age_days": cfg.model_max_age_days,
        "too_old": age_days > cfg.model_max_age_days,
        "quality_gate": recorded.get("quality_gate"),
        "metrics": run.data.metrics,
        "degradation_evidence": "not checked here: call check_prediction_drift for input drift; "
        "live accuracy needs true labels, which the system does not have",
    }


class PromoteArgs(Args):
    version: str = Field(pattern=r"^\d+$", description="Registry version to promote, e.g. 7")
    model: str | None = Model


@tool(PromoteArgs, mutates=True)
def promote_model(args: PromoteArgs) -> dict:
    """Move the deployed alias (champion) to a registry version. Only versions whose run passed
    the quality gate and has data provenance. Does not redeploy the running service."""
    cfg = settings().mlflow
    name = model_name(args.model)
    c = client()
    version = c.get_model_version(name, args.version)
    recorded = provenance(c.get_run(source_run_id(version)).data.tags)
    if recorded.get("quality_gate") != "passed":
        raise PermissionError(f"version {args.version} did not pass the quality gate")
    if "data_hash" not in recorded:
        raise PermissionError(f"version {args.version} has no data provenance")
    previous = deployed_version(name)
    c.set_registered_model_alias(name, cfg.aliases.deployed, args.version)
    return {
        "model": name,
        "alias": cfg.aliases.deployed,
        "version": args.version,
        "previous_version": previous.version if previous else None,
        "data_version": recorded.get("data_version"),
        "note": "registry updated only. The service keeps its current model: a restart does not change "
        "it; serving this version needs a new image built and deployed (manual step)",
    }
