"""Build a version of the Wine Quality dataset from the UCI source files.

v1: red wine only. v2: red followed by white. Output is a standard CSV.
"""

import argparse
from pathlib import Path

import pandas as pd

EXPECTED_ROWS = {"v1": 1599, "v2": 6497}


def build(version: str, source: Path) -> pd.DataFrame:
    frames = [pd.read_csv(source / "winequality-red.csv", sep=";")]
    if version == "v2":
        frames.append(pd.read_csv(source / "winequality-white.csv", sep=";"))
    df = pd.concat(frames, ignore_index=True)
    if len(df) != EXPECTED_ROWS[version]:
        raise SystemExit(f"{version}: expected {EXPECTED_ROWS[version]} rows, got {len(df)}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", choices=EXPECTED_ROWS)
    parser.add_argument("--source", type=Path, required=True, help="folder with the UCI CSV files")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.version, args.source).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
