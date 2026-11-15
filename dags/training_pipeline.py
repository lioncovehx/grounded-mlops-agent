"""Training pipeline: validate, preprocess, train, evaluate and gate the wine quality model.

The train task's last output line (the run ID) is its XCom value, passed on as --run-id.
"""

from datetime import timedelta

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

STAGE = "cd /opt/project && /opt/project-venv/bin/python training/train.py"
RUN_ID = "{{ ti.xcom_pull(task_ids='train') }}"

with DAG(
    dag_id="training_pipeline",
    schedule=None,
    catchup=False,
    default_args={"retries": 2, "retry_delay": timedelta(seconds=30)},
    tags=["training"],
    doc_md=__doc__,
):
    # Bad data and a failed gate won't fix themselves, so those two never retry.
    validate = BashOperator(task_id="validate_dataset", bash_command=f"{STAGE} validate", retries=0)
    preprocess = BashOperator(task_id="preprocess", bash_command=f"{STAGE} preprocess")
    train = BashOperator(
        task_id="train", bash_command=f"{STAGE} train --model rf --n-estimators 100"
    )
    evaluate = BashOperator(
        task_id="evaluate_and_log", bash_command=f"{STAGE} evaluate --run-id {RUN_ID}"
    )
    gate = BashOperator(
        task_id="quality_gate", bash_command=f"{STAGE} gate --run-id {RUN_ID}", retries=0
    )

    validate >> preprocess >> train >> evaluate >> gate
