"""Tool layer: validation, permissions, timeouts, failures and logging."""

import json
import logging
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent import registry
from agent.registry import execute_tool
from agent.tools import dvc_tools, kubernetes_tools, mlflow_tools


def make_run(run_id, f1=None, data_tag="data-v2", gate="passed"):
    metrics = {} if f1 is None else {"f1": f1}
    return SimpleNamespace(
        info=SimpleNamespace(
            run_id=run_id, experiment_id="1", status="FINISHED", start_time=1_700_000_000_000
        ),
        data=SimpleNamespace(
            metrics=metrics,
            params={"model": "rf"},
            tags={"data.tag": data_tag, "data.md5": "abc", "quality_gate": gate},
        ),
    )


@pytest.fixture
def fake_mlflow(monkeypatch):
    fake = MagicMock()
    fake.get_experiment_by_name.return_value = SimpleNamespace(experiment_id="1")
    monkeypatch.setattr(mlflow_tools, "client", lambda: fake)
    return fake


def test_best_run_is_highest_f1_and_skips_runs_without_the_metric(fake_mlflow):
    fake_mlflow.search_runs.return_value = [
        make_run("a" * 32, f1=0.41, gate="failed"),
        make_run("b" * 32, f1=0.49),
        make_run("c" * 32),  # crashed before evaluation
    ]
    result = execute_tool("get_best_run", {"metric": "f1"}, "t1")
    assert result["status"] == "ok"
    assert result["data"]["best"]["run_id"] == "b" * 32
    assert result["data"]["best"]["tags"]["quality_gate"] == "passed"
    assert result["data"]["runner_up"]["run_id"] == "a" * 32
    assert result["data"]["runs_missing_metric"] == 1


def test_unknown_metric_is_rejected(fake_mlflow):
    result = execute_tool("get_best_run", {"metric": "accuracy"}, "t2")
    assert result["status"] == "error"
    assert "unknown metric" in result["error"]
    fake_mlflow.search_runs.assert_not_called()


def test_mlflow_down_is_reported_as_unavailable(monkeypatch):
    # The real MLflow client, pointed at a closed port by conftest.
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    result = execute_tool("list_experiments", {}, "t3")
    assert result["status"] == "unavailable"
    assert result["data"] is None


@pytest.mark.parametrize(
    "raw_args",
    [{"limit": 999}, {"limit": "many"}, {"experiment": "wine-quality", "surprise": 1}],
)
def test_invalid_arguments_give_an_error_envelope(raw_args):
    result = execute_tool("list_runs", raw_args, "t4")
    assert result["status"] == "error"
    assert result["error"].startswith("invalid arguments")


def test_slow_tool_times_out_without_crashing(monkeypatch):
    monkeypatch.setattr(registry, "TOOL_TIMEOUT_S", 0.2)
    monkeypatch.setattr(dvc_tools, "run", lambda *a: time.sleep(2))
    result = execute_tool("detect_data_changes", {}, "t5")
    assert result["status"] == "timeout"
    assert result["duration_ms"] < 1500


@pytest.fixture
def fake_k8s(monkeypatch):
    core, apps = MagicMock(), MagicMock()
    monkeypatch.setattr(kubernetes_tools, "clients", lambda: (core, apps))
    return core, apps


@pytest.mark.parametrize("namespace", ["default", "kube-system"])
def test_restart_outside_approved_namespace_never_reaches_kubernetes(fake_k8s, namespace):
    _, apps = fake_k8s
    result = execute_tool(
        "restart_deployment", {"name": "wine-api", "namespace": namespace}, "t6", confirmed=True
    )
    assert result["status"] == "error"
    assert "outside the approved namespace" in result["error"]
    apps.patch_namespaced_deployment.assert_not_called()


def test_restart_without_confirmation_does_nothing(fake_k8s):
    _, apps = fake_k8s
    result = execute_tool("restart_deployment", {"name": "wine-api"}, "t7")
    assert result["status"] == "confirmation_required"
    apps.patch_namespaced_deployment.assert_not_called()


def test_confirmed_restart_patches_the_deployment(fake_k8s):
    _, apps = fake_k8s
    result = execute_tool("restart_deployment", {"name": "wine-api"}, "t8", confirmed=True)
    assert result["status"] == "ok"
    name, namespace, patch = apps.patch_namespaced_deployment.call_args.args
    assert (name, namespace) == ("wine-api", "mlops-app")
    assert (
        "kubectl.kubernetes.io/restartedAt" in patch["spec"]["template"]["metadata"]["annotations"]
    )


def test_trigger_dag_refuses_dags_outside_the_allowlist(monkeypatch):
    from agent.tools import airflow_tools

    api = MagicMock()
    monkeypatch.setattr(airflow_tools, "api", api)
    result = execute_tool("trigger_dag", {"dag_id": "drop_everything"}, "t9", confirmed=True)
    assert result["status"] == "error"
    assert "not in the approved list" in result["error"]
    api.assert_not_called()


def test_dvc_refuses_unapproved_commands_and_flag_like_revisions(monkeypatch):
    with pytest.raises(PermissionError):
        dvc_tools.run("git", "push")
    with pytest.raises(PermissionError):
        dvc_tools.run("dvc", "gc")

    ran = MagicMock()
    monkeypatch.setattr(dvc_tools.subprocess, "run", ran)
    for rev in ["--output=/tmp/x", "-h", "HEAD; rm -rf /", "a b"]:
        result = execute_tool("get_dataset_version", {"rev": rev}, "t10")
        assert result["status"] == "error", rev
    ran.assert_not_called()


def test_dataset_for_run_reports_missing_provenance_instead_of_guessing(fake_mlflow):
    fake_mlflow.get_run.return_value = SimpleNamespace(data=SimpleNamespace(tags={}))
    result = execute_tool("get_dataset_for_run", {"run_id": "d" * 32}, "t11")
    assert result["status"] == "ok"
    assert result["data"]["verified"] is False
    assert "missing provenance" in result["data"]["finding"]


def test_log_record_has_fixed_fields_and_no_arguments(caplog, fake_mlflow):
    secret = "hunter2-secret-token"
    with caplog.at_level(logging.INFO, logger="agent.tools"):
        execute_tool("list_runs", {"experiment": secret}, "req-42")
    records = [json.loads(r.getMessage()) for r in caplog.records if r.name == "agent.tools"]
    assert records == [
        {
            "event": "tool_call",
            "tool": "list_runs",
            "status": "ok",
            "duration_ms": records[0]["duration_ms"],
            "request_id": "req-42",
        }
    ]
    assert secret not in caplog.text


def test_stuck_rollout_is_not_reported_complete(fake_k8s):
    # Seen live: a bad image tag left the old Pods available while the new one could not pull.
    _, apps = fake_k8s
    apps.read_namespaced_deployment.return_value = SimpleNamespace(
        metadata=SimpleNamespace(name="wine-api"),
        spec=SimpleNamespace(
            replicas=2,
            template=SimpleNamespace(
                metadata=SimpleNamespace(labels={}),
                spec=SimpleNamespace(containers=[SimpleNamespace(image="wine-api:0.2.0")]),
            ),
        ),
        status=SimpleNamespace(
            ready_replicas=2, available_replicas=2, updated_replicas=1, unavailable_replicas=1
        ),
    )
    result = execute_tool("get_deployed_version", {}, "t12")
    assert result["data"]["rollout_complete"] is False
