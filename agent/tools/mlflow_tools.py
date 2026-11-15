"""MLflow tools: experiments, runs, the model registry and run-to-data lineage, read-only."""

import os
from datetime import UTC, datetime

from pydantic import Field

from agent.registry import Args, NoArgs, tool
from agent.tools import dvc_tools

# Fail fast when MLflow is down instead of retrying for minutes.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")

DEFAULT_EXPERIMENT = "wine-quality"
DEFAULT_MODEL = "wine-quality"
KNOWN_METRICS = {"f1"}
PROVENANCE_TAGS = ("data.md5", "data.tag", "data.commit", "git.commit", "quality_gate")


def client():
    from mlflow import MlflowClient  # imported lazily: mlflow is slow to import

    return MlflowClient()


def iso(ms: int | None) -> str | None:
    return datetime.fromtimestamp(ms / 1000, UTC).isoformat() if ms else None


def summarize(run) -> dict:
    """A small, plain view of a run: what the agent needs, nothing more."""
    tags = run.data.tags
    return {
        "run_id": run.info.run_id,
        "experiment_id": run.info.experiment_id,
        "status": run.info.status,
        "start_time": iso(run.info.start_time),
        "metrics": run.data.metrics,
        "params": run.data.params,
        "tags": {k: tags[k] for k in PROVENANCE_TAGS if k in tags},
    }


def experiment_id(name_or_id: str) -> str:
    c = client()
    exp = c.get_experiment_by_name(name_or_id)
    if exp is None and name_or_id.isdigit():
        exp = c.get_experiment(name_or_id)
    if exp is None:
        raise ValueError(f"no experiment named or numbered {name_or_id!r}")
    return exp.experiment_id


class ExperimentArgs(Args):
    experiment: str = Field(DEFAULT_EXPERIMENT, description="Experiment name or numeric ID")
    limit: int = Field(10, ge=1, le=50, description="Most recent runs to return")


class RunArgs(Args):
    run_id: str = Field(pattern=r"^[0-9a-f]{32}$", description="32-character MLflow run ID")


class BestRunArgs(Args):
    metric: str = Field("f1", description="Metric to rank by. Known metrics: f1")
    experiment: str = Field(DEFAULT_EXPERIMENT, description="Experiment name or numeric ID")


class CompareRunsArgs(Args):
    run_ids: list[str] = Field(min_length=2, max_length=10, description="Run IDs to compare")


class ModelArgs(Args):
    model: str = Field(DEFAULT_MODEL, description="Registered model name")


@tool(NoArgs)
def list_experiments(_: NoArgs) -> dict:
    """List MLflow experiments with their IDs."""
    return {
        "experiments": [
            {"experiment_id": e.experiment_id, "name": e.name}
            for e in client().search_experiments()
        ]
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
    return {"experiment": args.experiment, "runs": [summarize(r) for r in runs]}


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
    never ranked. The best metric alone does not mean production-ready: check quality_gate and
    the dataset version."""
    if args.metric not in KNOWN_METRICS:
        raise ValueError(f"unknown metric {args.metric!r}; known: {sorted(KNOWN_METRICS)}")
    exp = experiment_id(args.experiment)
    runs = client().search_runs([exp], max_results=500)
    ranked = [r for r in runs if args.metric in r.data.metrics]
    if not ranked:
        raise ValueError(f"no run in {args.experiment!r} has logged {args.metric!r}")
    ranked.sort(key=lambda r: r.data.metrics[args.metric], reverse=True)
    return {
        "metric": args.metric,
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
    """List versions of a registered model with their aliases (candidate, champion), source run,
    F1 and dataset version."""
    c = client()
    model = c.get_registered_model(args.model)
    aliases_by_version: dict[str, list[str]] = {}
    for alias, version in model.aliases.items():
        aliases_by_version.setdefault(str(version), []).append(alias)

    versions = []
    for v in c.search_model_versions(f"name='{args.model}'"):
        run = c.get_run(v.run_id) if v.run_id else None
        versions.append(
            {
                "version": v.version,
                "aliases": aliases_by_version.get(str(v.version), []),
                "source_run": v.tags.get("source_run", v.run_id),
                "created": iso(v.creation_timestamp),
                "f1": run.data.metrics.get("f1") if run else None,
                "data_tag": run.data.tags.get("data.tag") if run else None,
            }
        )
    versions.sort(key=lambda v: int(v["version"]), reverse=True)
    return {"model": args.model, "aliases": dict(model.aliases), "versions": versions}


@tool(RunArgs)
def get_dataset_for_run(args: RunArgs) -> dict:
    """Identify the dataset version a run was trained on, verified against Git: compares the run's
    recorded data hash with the DVC pointer at the recorded data commit."""
    tags = client().get_run(args.run_id).data.tags
    recorded = {k: tags.get(k) for k in ("data.md5", "data.tag", "data.commit", "git.commit")}
    missing = [k for k in ("data.md5", "data.commit") if not recorded[k]]
    if missing:
        return {
            "run_id": args.run_id,
            "recorded": recorded,
            "verified": False,
            "finding": f"run is missing provenance tags {missing}; its dataset cannot be verified",
        }
    pointer = dvc_tools.pointer_at(recorded["data.commit"])
    return {
        "run_id": args.run_id,
        "recorded": recorded,
        "pointer_md5_at_data_commit": pointer["md5"],
        "verified": pointer["md5"] == recorded["data.md5"],
        "data_tag_for_hash": dvc_tools.tag_for_md5(recorded["data.md5"]),
    }


@tool(ModelArgs)
def get_retraining_signals(args: ModelArgs) -> dict:
    """Facts for deciding whether to retrain the deployed (champion) model: newer data available,
    model age, and its quality gate result. Reports evidence; does not decide."""
    c = client()
    max_age_days = int(os.environ.get("MODEL_MAX_AGE_DAYS", "30"))
    try:
        version = c.get_model_version_by_alias(args.model, "champion")
    except Exception as exc:
        if "not found" not in str(exc).lower() and "RESOURCE_DOES_NOT_EXIST" not in str(exc):
            raise
        return {"model": args.model, "champion": None, "finding": "no champion alias is set"}

    run = c.get_run(version.tags.get("source_run", version.run_id))
    trained_md5 = run.data.tags.get("data.md5")
    current = dvc_tools.pointer_at("HEAD")
    age_days = (datetime.now(UTC).timestamp() * 1000 - run.info.start_time) / 86_400_000
    return {
        "model": args.model,
        "champion_version": version.version,
        "source_run": run.info.run_id,
        "trained_on": {"md5": trained_md5, "data_tag": run.data.tags.get("data.tag")},
        "current_data": {"md5": current["md5"], "data_tag": dvc_tools.tag_for_md5(current["md5"])},
        "newer_data_available": trained_md5 != current["md5"],
        "model_age_days": round(age_days, 1),
        "max_age_days": max_age_days,
        "too_old": age_days > max_age_days,
        "quality_gate": run.data.tags.get("quality_gate"),
        "f1": run.data.metrics.get("f1"),
        "degradation_evidence": "none supplied; only the user can provide production metrics",
    }
