# Grounded MLOps Agent

Conversational AI agent for MLOps: traces dataset lineage (DVC), compares experiments (MLflow), runs training pipelines (Airflow) and diagnoses Kubernetes deployments, backing every answer with real tool evidence.

**Status:** in development.

## Data

Uses the Wine Quality dataset by P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis,
from the UCI Machine Learning Repository, licensed under CC BY 4.0:
https://archive.ics.uci.edu/dataset/186/wine+quality

Two versions are tracked with DVC and tagged in Git:

| Tag | Contents | Rows |
|---|---|---|
| `data-v1` | Red wine | 1,599 |
| `data-v2` | Red and white wine | 6,497 |

The DVC remote is local to the author's machine, so `dvc pull` won't work on a fresh clone.
Rebuild the data from the UCI files instead:

```bash
python data/build_dataset.py v2 --source <folder with the UCI CSV files> --out data/raw/wine.csv
```
