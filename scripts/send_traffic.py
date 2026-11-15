"""Send sample prediction requests to the inference service, to exercise drift monitoring.

normal: real rows from the processed training data.
skewed: the same rows with residual sugar tripled (capped at the API's limit), a simulated shift.
"""

import argparse
from pathlib import Path

import httpx
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data/processed/wine.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["normal", "skewed"])
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()

    rows = pd.read_csv(DATA).drop(columns="good").sample(args.n, random_state=7)
    if args.mode == "skewed":
        rows["residual_sugar"] = (rows["residual_sugar"] * 3).clip(upper=100)

    good = 0
    with httpx.Client(base_url=args.url, timeout=10) as client:
        for row in rows.to_dict(orient="records"):
            resp = client.post("/predict", json=row)
            resp.raise_for_status()
            good += resp.json()["good_quality"]
    print(f"sent {args.n} {args.mode} requests; {good} predicted good")


if __name__ == "__main__":
    main()
