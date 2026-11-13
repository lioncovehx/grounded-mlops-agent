"""Training pipeline stages for the wine quality model.

Run one stage at a time: python training/train.py <stage> [options]
Log messages go to stderr; the last line on stdout is the stage's result.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/wine.csv"
POINTER = ROOT / "data/raw/wine.csv.dvc"
PROCESSED = ROOT / "data/processed/wine.csv"
OUTPUTS = ROOT / "outputs"

EXPERIMENT = "wine-quality"
REGISTERED_MODEL = "wine-quality"
TARGET = "good"
RAW_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality",
]

log = logging.getLogger("train")


def run(*cmd: str) -> str:
    return subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def pointer_md5(text: str) -> str:
    return yaml.safe_load(text)["outs"][0]["md5"]


def provenance() -> dict[str, str]:
    """Identify the data version by matching the current pointer's hash to a data-* tag."""
    md5 = pointer_md5(POINTER.read_text())
    for tag in run("git", "tag", "-l", "data-*").splitlines():
        if pointer_md5(run("git", "show", f"{tag}:data/raw/wine.csv.dvc")) == md5:
            return {
                "data.md5": md5,
                "data.tag": tag,
                "data.commit": run("git", "rev-list", "-n", "1", tag),
                "git.commit": run("git", "rev-parse", "HEAD"),
            }
    raise SystemExit(f"data md5 {md5} matches no data-* tag; version the data before training")


def split(df: pd.DataFrame):
    X, y = df.drop(columns=TARGET), df[TARGET]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def validate(_args) -> None:
    df = pd.read_csv(RAW)
    if list(df.columns) != RAW_COLUMNS:
        raise SystemExit(f"unexpected columns: {list(df.columns)}")
    if df.isna().any().any():
        raise SystemExit("data contains null values")
    if not df["quality"].between(0, 10).all():
        raise SystemExit("quality outside 0-10")
    status = json.loads(run(sys.executable, "-m", "dvc", "status", str(POINTER), "--json") or "{}")
    if status:
        raise SystemExit(f"raw data differs from its DVC pointer: {status}")
    log.info("validated %d rows", len(df))


def preprocess(_args) -> None:
    df = pd.read_csv(RAW).drop_duplicates()
    df[TARGET] = (df["quality"] >= 7).astype(int)
    df = df.drop(columns="quality")
    df.columns = [c.replace(" ", "_") for c in df.columns]
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED, index=False)
    log.info("wrote %d rows, %.1f%% positive", len(df), 100 * df[TARGET].mean())
    print(PROCESSED)


def train(args) -> None:
    X_train, _, y_train, _ = split(pd.read_csv(PROCESSED))
    if args.model == "rf":
        model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=42, n_jobs=-1)
        params = {"model": "rf", "n_estimators": args.n_estimators}
    else:
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        params = {"model": "logreg"}

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run() as active:
        run_id = active.info.run_id
        mlflow.set_tags(provenance())
        mlflow.log_params({**params, "train_rows": len(X_train)})
        model.fit(X_train, y_train)

        path = OUTPUTS / run_id / "model.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        mlflow.log_artifact(str(path))
        # We built this model, so its tree storage is safe for MLflow's skops serializer.
        info = mlflow.sklearn.log_model(
            model, name="model", skops_trusted_types=["sklearn.tree._tree.Tree"]
        )
        mlflow.set_tag("model.uri", info.model_uri)
    log.info("trained %s", params)
    print(run_id)


def evaluate(args) -> None:
    _, X_test, _, y_test = split(pd.read_csv(PROCESSED))
    model = joblib.load(OUTPUTS / args.run_id / "model.joblib")
    f1 = f1_score(y_test, model.predict(X_test))
    with mlflow.start_run(run_id=args.run_id):
        mlflow.log_metric("f1", f1)
    log.info("f1=%.3f", f1)
    print(args.run_id)


def gate(args) -> None:
    client = MlflowClient()
    run_data = client.get_run(args.run_id).data
    threshold = float(os.environ.get("QUALITY_GATE_F1", "0.45"))
    f1 = run_data.metrics.get("f1")
    passed = f1 is not None and f1 >= threshold
    client.set_tag(args.run_id, "quality_gate", "passed" if passed else "failed")
    if not passed:
        raise SystemExit(f"quality gate failed: f1={f1} threshold={threshold}")

    version = mlflow.register_model(run_data.tags["model.uri"], REGISTERED_MODEL).version
    client.set_model_version_tag(REGISTERED_MODEL, version, "source_run", args.run_id)
    client.set_registered_model_alias(REGISTERED_MODEL, "candidate", version)
    log.info("f1=%.3f passed; registered %s v%s as candidate", f1, REGISTERED_MODEL, version)
    print(version)


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(message)s")
    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
    # Keep stdout to the stage result only; MLflow otherwise prints a UI link after each run.
    os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true")

    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    stages.add_parser("validate").set_defaults(fn=validate)
    stages.add_parser("preprocess").set_defaults(fn=preprocess)
    p = stages.add_parser("train")
    p.add_argument("--model", choices=["rf", "logreg"], required=True)
    p.add_argument("--n-estimators", type=int, default=100)
    p.set_defaults(fn=train)
    for name, fn in [("evaluate", evaluate), ("gate", gate)]:
        p = stages.add_parser(name)
        p.add_argument("--run-id", required=True)
        p.set_defaults(fn=fn)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
