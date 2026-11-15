"""Agent loop and HTTP API, with a scripted LLM and mocks at the service edges."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from conftest import FakeLLM, answer, calls, tool_call
from fastapi.testclient import TestClient

from agent import main, orchestrator
from agent.llm import LLMError
from agent.tools import airflow_tools, dvc_tools, kubernetes_tools, mlflow_tools

GOOD_RUN = "b" * 32


@pytest.fixture
def services(monkeypatch):
    """Mock every external system at its client boundary; tool code itself runs for real."""
    run = SimpleNamespace(
        info=SimpleNamespace(
            run_id=GOOD_RUN, experiment_id="1", status="FINISHED", start_time=1_700_000_000_000
        ),
        data=SimpleNamespace(
            metrics={"f1": 0.49},
            params={"model": "rf"},
            tags={
                "data.md5": "m2",
                "data.tag": "data-v2",
                "data.commit": "c2",
                "quality_gate": "passed",
            },
        ),
    )
    ml = MagicMock()
    ml.get_experiment_by_name.return_value = SimpleNamespace(experiment_id="1")
    ml.search_runs.return_value = [run]
    ml.get_run.return_value = run
    monkeypatch.setattr(mlflow_tools, "client", lambda: ml)

    monkeypatch.setattr(dvc_tools, "pointer_at", lambda rev, path: {"md5": "m2", "size_bytes": 1})
    monkeypatch.setattr(
        dvc_tools, "data_tags", lambda path: {"data-v2": {"md5": "m2", "commit": "c2"}}
    )
    monkeypatch.setattr(dvc_tools, "tracked_datasets", lambda: ["data/raw/wine.csv"])

    af = MagicMock(
        side_effect=lambda method, path, **kw: {
            "dag_runs": [{"dag_run_id": "manual__1"}],
            "state": "success",
            "task_instances": [{"task_id": "quality_gate", "state": "success", "try_number": 1}],
        }
    )
    monkeypatch.setattr(airflow_tools, "api", af)

    deployment = SimpleNamespace(
        metadata=SimpleNamespace(name="wine-api"),
        spec=SimpleNamespace(
            replicas=2,
            template=SimpleNamespace(
                metadata=SimpleNamespace(labels={"mlops/model-version": "4"}),
                spec=SimpleNamespace(containers=[SimpleNamespace(image="wine-api:0.1.0")]),
            ),
        ),
        status=SimpleNamespace(
            ready_replicas=2, available_replicas=2, updated_replicas=2, unavailable_replicas=0
        ),
    )
    core, apps = MagicMock(), MagicMock()
    apps.read_namespaced_deployment.return_value = deployment
    monkeypatch.setattr(kubernetes_tools, "clients", lambda: (core, apps))
    return SimpleNamespace(mlflow=ml, airflow=af, core=core, apps=apps)


def test_selected_tool_runs_and_its_result_reaches_the_evidence(services):
    llm = FakeLLM(calls(tool_call("get_best_run", '{"metric": "f1"}')), answer("Best is bbbb."))
    result = orchestrator.run_agent("best model?", "r1", llm=llm)
    assert [e["tool"] for e in result["evidence"]] == ["get_best_run"]
    assert result["evidence"][0]["data"]["best"]["run_id"] == GOOD_RUN
    # The LLM saw the real tool result, after its own tool-call message.
    tool_msg = llm.calls[1][-1]
    assert tool_msg["role"] == "tool" and GOOD_RUN in tool_msg["content"]
    assert llm.calls[1][-2]["tool_calls"]


def test_mandatory_scenario_collects_evidence_from_all_four_systems(services):
    llm = FakeLLM(
        calls(tool_call("get_best_run", '{"metric": "f1"}')),
        calls(
            tool_call("get_dataset_for_run", json.dumps({"run_id": GOOD_RUN})),
            tool_call("get_deployed_version"),
            tool_call("get_dag_run_status"),
        ),
        answer("**Known facts** ... **Uncertainties** ... **Recommended actions** ..."),
    )
    result = orchestrator.run_agent("Find the best recent model ...", "r2", llm=llm)
    assert all(e["status"] == "ok" for e in result["evidence"])
    modules = {e["tool"] for e in result["evidence"]}
    assert modules == {
        "get_best_run",
        "get_dataset_for_run",
        "get_deployed_version",
        "get_dag_run_status",
    }
    assert result["evidence"][1]["data"]["verified"] is True
    # Answering must not start a pipeline run.
    assert not any(c.args[0] == "POST" for c in services.airflow.call_args_list)


def test_failed_tool_is_named_by_code_and_no_data_is_invented(monkeypatch):
    # MLflow is unreachable (closed port from conftest); the LLM tries to answer anyway.
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    llm = FakeLLM(calls(tool_call("get_best_run")), answer("The best model has F1 0.99."))
    result = orchestrator.run_agent("best model?", "r3", llm=llm)
    evidence = result["evidence"][0]
    assert evidence["status"] == "unavailable"
    assert evidence["data"] is None
    assert "get_best_run (unavailable)" in result["answer"]


def test_invalid_tool_arguments_from_the_llm_become_an_error_result(services):
    llm = FakeLLM(calls(tool_call("get_best_run", "{not json")), answer("Could not check."))
    result = orchestrator.run_agent("best model?", "r4", llm=llm)
    assert result["evidence"][0]["status"] == "error"
    services.mlflow.search_runs.assert_not_called()


def test_llm_outage_gives_a_clear_answer():
    def down(*_):
        raise LLMError("LLM unavailable at http://127.0.0.1:9/v1: ConnectError")

    result = orchestrator.run_agent("hello", "r5", llm=down)
    assert "language model is unavailable" in result["answer"]


def test_restart_waits_for_confirmation_through_the_api_and_runs_once(services, monkeypatch):
    llm = FakeLLM(
        calls(tool_call("restart_deployment", '{"name": "wine-api"}')), answer("Please confirm.")
    )
    monkeypatch.setattr(
        main, "run_agent", lambda msg, rid: orchestrator.run_agent(msg, rid, llm=llm)
    )
    api = TestClient(main.app)

    chat = api.post("/chat", json={"message": "Restart the wine-api deployment"}).json()
    assert chat["evidence"][0]["status"] == "confirmation_required"
    action_id = chat["pending_action_id"]
    services.apps.patch_namespaced_deployment.assert_not_called()

    confirmed = api.post(f"/actions/{action_id}/confirm").json()
    assert confirmed["evidence"][0]["status"] == "ok"
    services.apps.patch_namespaced_deployment.assert_called_once()
    assert api.post(f"/actions/{action_id}/confirm").status_code == 404


def test_cancelled_action_can_no_longer_be_confirmed(services, monkeypatch):
    llm = FakeLLM(
        calls(tool_call("trigger_dag", '{"dag_id": "training_pipeline"}')), answer("Confirm?")
    )
    monkeypatch.setattr(
        main, "run_agent", lambda msg, rid: orchestrator.run_agent(msg, rid, llm=llm)
    )
    api = TestClient(main.app)
    action_id = api.post("/chat", json={"message": "retrain now"}).json()["pending_action_id"]
    assert api.post(f"/actions/{action_id}/cancel").status_code == 200
    assert api.post(f"/actions/{action_id}/confirm").status_code == 404
    assert not any(c.args[0] == "POST" for c in services.airflow.call_args_list)


@pytest.mark.parametrize("body", [{"message": ""}, {"message": "x" * 4001}, {}])
def test_chat_rejects_invalid_requests(body):
    assert TestClient(main.app).post("/chat", json=body).status_code == 422
