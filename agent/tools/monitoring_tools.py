"""Drift monitoring: live prediction inputs vs. the deployed model's training data.

Needs prediction log lines and a feature_stats.json artifact; formats in the README.
"""

import json
import tempfile
from pathlib import Path

from pydantic import Field

from agent.config import required, settings
from agent.registry import tool
from agent.tools import kubernetes_tools, mlflow_tools


class DriftArgs(kubernetes_tools.NamespaceArgs):
    max_log_lines: int = Field(
        1000, ge=50, le=5000, description="Recent log lines to read from each Pod"
    )


def prediction_records(log_text: str) -> list[dict]:
    """Prediction events from a Pod log; other lines are skipped."""
    event_name = settings().monitoring.prediction_event
    records = []
    for line in log_text.splitlines():
        start = line.find("{")
        if start < 0:
            continue
        try:
            event = json.loads(line[start:])
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("event") == event_name and "features" in event:
            records.append(event)
    return records


def baseline() -> dict:
    """The deployed model's feature_stats.json, with the version and run it came from."""
    cfg = settings().mlflow
    model = mlflow_tools.model_name(None)
    version = mlflow_tools.deployed_version(model)
    if version is None:
        raise ValueError(f"model {model!r} has no {cfg.aliases.deployed!r} alias")
    run_id = mlflow_tools.source_run_id(version)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            path = mlflow_tools.client().download_artifacts(run_id, cfg.feature_stats_artifact, tmp)
        except Exception as exc:
            if "not found" in str(exc).lower() or "No such file" in str(exc):
                raise ValueError(
                    f"run {run_id} has no {cfg.feature_stats_artifact}; log it at training time"
                ) from exc
            raise
        stats = json.loads(Path(path).read_text())
    return {"version": version.version, "run_id": run_id, **stats}


def compare(records: list[dict], stats: dict, threshold: float) -> dict:
    """Mean shift of each feature, in training standard deviations."""
    features = {}
    for name, base in stats["features"].items():
        values = [r["features"][name] for r in records if name in r["features"]]
        if not values:
            continue
        live_mean = sum(values) / len(values)
        shift = (live_mean - base["mean"]) / (base["std"] or 1.0)
        features[name] = {
            "training_mean": round(base["mean"], 4),
            "live_mean": round(live_mean, 4),
            "shift_in_std": round(shift, 2),
            "drifted": abs(shift) > threshold,
        }
    return features


@tool(DriftArgs)
def check_prediction_drift(args: DriftArgs) -> dict:
    """Check whether live prediction inputs have drifted from the deployed model's training data:
    per-feature mean shift in training standard deviations, and the live positive-prediction rate
    against the training label rate. Input drift only: live accuracy needs true labels."""
    ns = args.checked()
    mon = settings().monitoring
    selector = required(settings().kubernetes.pod_selector, "inference Pod selector")
    core, _ = kubernetes_tools.clients()
    pods = core.list_namespaced_pod(
        ns, label_selector=selector, _request_timeout=kubernetes_tools.REQUEST_TIMEOUT
    ).items
    from kubernetes.client.exceptions import ApiException

    records, read, skipped = [], [], {}
    for pod in pods:
        try:
            log_text = core.read_namespaced_pod_log(
                pod.metadata.name,
                ns,
                tail_lines=args.max_log_lines,
                _request_timeout=kubernetes_tools.REQUEST_TIMEOUT,
            )
        except ApiException as exc:
            # A Pod that never started has no logs; skip it rather than fail the whole check.
            skipped[pod.metadata.name] = f"logs unavailable (HTTP {exc.status})"
            continue
        read.append(pod.metadata.name)
        records.extend(prediction_records(log_text))

    result = {
        "pods_read": read,
        "pods_skipped": skipped,
        "predictions_found": len(records),
        "window": "since each Pod started, up to max_log_lines per Pod",
        "model_versions_seen": sorted({str(r.get("model_version")) for r in records}),
    }
    if len(records) < mon.min_predictions:
        return {
            **result,
            "drift_detected": None,
            "finding": f"only {len(records)} predictions logged; need at least "
            f"{mon.min_predictions} to judge drift",
        }

    stats = baseline()
    features = compare(records, stats, mon.drift_threshold_std)
    drifted = [name for name, f in features.items() if f["drifted"]]
    predictions = [r.get("prediction") for r in records]
    binary = all(p in (0, 1, True, False) for p in predictions)
    return {
        **result,
        "deployed_version": stats["version"],
        "baseline_run": stats["run_id"],
        "threshold_std": mon.drift_threshold_std,
        "drift_detected": bool(drifted),
        "drifted_features": drifted,
        "features": features,
        "positive_rate": {
            "live_predictions": round(sum(map(bool, predictions)) / len(predictions), 3),
            "training_labels": stats.get("label_positive_rate"),
        }
        if binary
        else None,
        "limitation": "input drift only; live accuracy needs true labels",
    }
