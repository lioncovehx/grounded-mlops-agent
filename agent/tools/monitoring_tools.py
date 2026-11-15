"""Production monitoring: compare live prediction inputs with the deployed model's training data.

Live side: the JSON prediction lines the inference service writes to its Pod logs.
Baseline: the exact dataset the champion was trained on, found through its provenance tags
(champion -> source run -> data.commit) and read from the DVC cache.
"""

import io
import json
from functools import lru_cache

import pandas as pd
from pydantic import Field

from agent.registry import tool
from agent.tools import dvc_tools, kubernetes_tools, mlflow_tools

MODEL = "wine-quality"
APP_LABEL = "app=wine-api"
DRIFT_THRESHOLD_STD = 0.5
MIN_PREDICTIONS = 30


class DriftArgs(kubernetes_tools.NamespaceArgs):
    max_log_lines: int = Field(
        1000, ge=50, le=5000, description="Recent log lines to read from each Pod"
    )


def prediction_records(log_text: str) -> list[dict]:
    """Prediction events from a Pod log. Other lines (access logs, startup) are skipped."""
    records = []
    for line in log_text.splitlines():
        start = line.find("{")
        if start < 0:
            continue
        try:
            event = json.loads(line[start:])
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("event") == "prediction":
            records.append(event)
    return records


@lru_cache(maxsize=4)
def training_data(data_commit: str) -> pd.DataFrame:
    """The training data at a data commit, preprocessed the same way as training/train.py."""
    import dvc.api  # imported lazily: slow, and not needed by other tools

    text = dvc.api.read(dvc_tools.DATASET, repo=str(dvc_tools.REPO), rev=data_commit)
    df = pd.read_csv(io.StringIO(text)).drop_duplicates()
    df["good"] = (df["quality"] >= 7).astype(int)
    df = df.drop(columns="quality")
    df.columns = [c.replace(" ", "_") for c in df.columns]
    return df


def champion_baseline() -> tuple[str, dict, pd.DataFrame]:
    c = mlflow_tools.client()
    version = c.get_model_version_by_alias(MODEL, "champion")
    run = c.get_run(version.tags.get("source_run", version.run_id))
    tags = run.data.tags
    if not tags.get("data.commit"):
        raise ValueError("champion run has no data.commit tag; its training data is unknown")
    return version.version, tags, training_data(tags["data.commit"])


def compare(live: pd.DataFrame, base: pd.DataFrame) -> dict:
    features = {}
    for col in [c for c in base.columns if c != "good" and c in live.columns]:
        std = base[col].std() or 1.0
        shift = (live[col].mean() - base[col].mean()) / std
        features[col] = {
            "training_mean": round(float(base[col].mean()), 4),
            "live_mean": round(float(live[col].mean()), 4),
            "shift_in_std": round(float(shift), 2),
            "drifted": bool(abs(shift) > DRIFT_THRESHOLD_STD),
        }
    return features


@tool(DriftArgs)
def check_prediction_drift(args: DriftArgs) -> dict:
    """Check whether live prediction inputs have drifted from the deployed (champion) model's
    training data: per-feature mean shift in training standard deviations, and the live share of
    'good' predictions against the training rate. Measures input drift only; live accuracy needs
    true quality labels, which this system does not have."""
    ns = args.checked()
    core, _ = kubernetes_tools.clients()
    pods = core.list_namespaced_pod(
        ns, label_selector=APP_LABEL, _request_timeout=kubernetes_tools.REQUEST_TIMEOUT
    ).items
    records = []
    for pod in pods:
        log_text = core.read_namespaced_pod_log(
            pod.metadata.name,
            ns,
            tail_lines=args.max_log_lines,
            _request_timeout=kubernetes_tools.REQUEST_TIMEOUT,
        )
        records.extend(prediction_records(log_text))

    result = {
        "pods_read": [p.metadata.name for p in pods],
        "predictions_found": len(records),
        "window": "since each Pod started, up to max_log_lines per Pod",
        "model_versions_seen": sorted({str(r.get("model_version")) for r in records}),
    }
    if len(records) < MIN_PREDICTIONS:
        return {
            **result,
            "drift_detected": None,
            "finding": f"only {len(records)} predictions logged; need at least {MIN_PREDICTIONS} "
            "to judge drift",
        }

    champion_version, tags, base = champion_baseline()
    live = pd.DataFrame([r["features"] for r in records])
    features = compare(live, base)
    drifted = [name for name, f in features.items() if f["drifted"]]
    live_rate = sum(bool(r.get("good_quality")) for r in records) / len(records)
    return {
        **result,
        "champion_version": champion_version,
        "baseline": {
            "data_tag": tags.get("data.tag"),
            "data_commit": tags["data.commit"],
            "rows": len(base),
        },
        "threshold_std": DRIFT_THRESHOLD_STD,
        "drift_detected": bool(drifted),
        "drifted_features": drifted,
        "features": features,
        "good_prediction_rate": {
            "live": round(live_rate, 3),
            "training_label_rate": round(float(base["good"].mean()), 3),
        },
        "limitation": "input drift only; live accuracy needs true quality labels",
    }
