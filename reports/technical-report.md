# Technical Report: Grounded MLOps Agent

## 1. System architecture

The system has four operational layers and an agent on top:

| Layer | Technology | Role |
|---|---|---|
| Data versioning | DVC 3.67 with Git tags `data-v1`, `data-v2` | Versions the Wine Quality CSV. Each version is a `.dvc` pointer (md5 and size) in a tagged commit |
| Experiment tracking | MLflow 3.18, sqlite store, proxied artifacts | Records params, F1, model artifacts and four provenance tags per run. Registry model `wine-quality` with aliases `candidate` and `champion` |
| Orchestration | Airflow 3.4, CeleryExecutor, Docker Compose | `training_pipeline`: validate → preprocess → train → evaluate → quality gate |
| Serving | FastAPI in Kubernetes (minikube) | `wine-api`: 2 replicas, readiness and liveness probes, requests and limits, ConfigMap, ClusterIP Service |
| Agent | FastAPI + OpenAI-compatible LLM | 24 tools behind one runner; answers from tool evidence; changes need human confirmation |

Diagrams of the components, the request flow, data lineage and the permission layers are in [`../architecture.md`](../architecture.md).

**Model.** Binary target `good = quality ≥ 7`. Duplicates are removed before an 80/20 stratified split with `random_state=42`, because with duplicates left in, identical rows fall on both sides of the split and F1 rises to about 0.75. The F1 scores on the test split:

| Model | `data-v1` | `data-v2` |
|---|---|---|
| Logistic regression | 0.346 | 0.410 |
| Random forest, 100 trees | 0.473 | 0.492 |
| Random forest, 300 trees | – | 0.495 |

The quality gate threshold is `F1 ≥ 0.45`, so logistic regression always fails it and the random forests pass.

**LLM.** The local setup ran Ornith 1.5 9B (Q4_K_M) on llama.cpp's `llama-server` with `--jinja`, using 7.3 GB of 8 GB VRAM. Training ran on the CPU. Any OpenAI-compatible endpoint can replace the LLM by changing `LLM_BASE_URL`, `LLM_MODEL` and `LLM_API_KEY`.

## 2. Tool definitions and permissions

Every tool runs through `execute_tool(name, args, request_id, confirmed=False)`, which:
1. validates arguments with a Pydantic model that rejects unknown fields
2. refuses a tool that changes the system unless `confirmed` is true
3. runs the tool with a 30-second timeout
4. returns an envelope `{tool, status, data, error, duration_ms, request_id}`, with status `ok`, `error`, `timeout`, `unavailable` or `confirmation_required`
5. logs one JSON line with only `tool`, `status`, `duration_ms` and `request_id`

| Module | Tool | Arguments | Access | Limits |
|---|---|---|---|---|
| DVC | `list_tracked_datasets` | – | read | `dvc list`; tags with their hashes |
| DVC | `get_dataset_version` | `rev` | read | Revision pattern `^[\w./-]+$`, no leading `-` |
| DVC | `compare_dataset_versions` | `rev_a`, `rev_b` | read | `dvc diff --json`, validated revisions |
| DVC | `detect_data_changes` | – | read | `dvc status`, `git status` of pointers |
| DVC | `get_reproduction_steps` | `rev` | read | Checks that the revision exists; returns commands, runs nothing |
| MLflow | `list_experiments` | – | read | |
| MLflow | `list_runs` | `experiment`, `limit` (1–50) | read | |
| MLflow | `get_run_details` | `run_id` (32 hex characters) | read | |
| MLflow | `get_best_run` | `metric` (known metrics only), `experiment` | read | Runs without the metric are excluded and counted |
| MLflow | `compare_runs` | `run_ids` (2–10) | read | |
| MLflow | `get_registry_info` | `model` | read | |
| MLflow | `get_dataset_for_run` | `run_id` | read | Verifies the recorded hash against Git; reports missing tags as a finding |
| MLflow | `get_retraining_signals` | `model` | read | Follows `champion`; reports evidence, doesn't decide |
| Airflow | `list_dags` | – | read | |
| Airflow | `get_dag_run_status` | `dag_id`, `dag_run_id` (default: latest) | read | |
| Airflow | `get_failed_tasks` | `dag_id`, `dag_run_id` | read | |
| Airflow | `get_task_diagnostics` | `dag_id`, `dag_run_id`, `task_id`, `try_number` | read | Last 40 log lines |
| Airflow | `trigger_dag` | `dag_id` | **write** | Confirmation required; `AGENT_ALLOWED_DAGS` only |
| Kubernetes | `list_workloads` | `namespace` | read | `AGENT_ALLOWED_NAMESPACE` only |
| Kubernetes | `get_pod_status_and_logs` | `namespace`, `pod`, `tail_lines` (≤100), `previous` | read | Same namespace limit |
| Kubernetes | `inspect_services` | `namespace` | read | Counts ready Pods behind each Service |
| Kubernetes | `find_unhealthy_pods` | `namespace` | read | Status reasons, restarts, last 15 warning Events |
| Kubernetes | `get_deployed_version` | `namespace`, `name` | read | Image, `model-version` label, rollout state |
| Kubernetes | `restart_deployment` | `namespace`, `name` | **write** | Confirmation required; namespace limit; patches `restartedAt` |

**Credentials.** The agent uses its own ServiceAccount token, not the cluster-admin kubeconfig. `kubectl auth can-i` output, from [`evidence/system-kubernetes.txt`](evidence/system-kubernetes.txt):

| Check, as the agent | Result |
|---|---|
| list pods, get pods/log, patch deployments in `mlops-app` | yes |
| delete pods, list secrets in `mlops-app` | no |
| list pods in `kube-system` | no |

The namespace limit is enforced twice. The tool refuses before any API call, and Kubernetes also returns 403 to the agent's token.

## 3. Evidence from each system

All files are in [`evidence/`](evidence/).

| System | File | Shows |
|---|---|---|
| DVC | `system-dvc.txt` | `dvc diff data-v1 data-v2` (modified `data/raw/wine.csv`), clean `dvc status`, both tags |
| MLflow | `system-mlflow.txt` | 8 runs with F1, `data.tag`, `quality_gate` and `git.commit`; versions 1–5; `candidate=5`, `champion=4` |
| Airflow | `system-airflow.txt` | Four pipeline runs: success, failed gate, failed validation, and the agent-triggered success |
| Kubernetes | `system-kubernetes.txt` | Deployment, Pods, Service and ConfigMap; rollout history; RBAC checks |
| Agent logs | `structured-logs.txt` | JSON log lines with tool, status, duration and request ID |
| CI | Pull request #1 | `lint-and-test` passing on GitHub Actions: ruff lint, format check, and 26 tests |

**The Airflow scenarios**, each run for real:

| Run | Setup | Result |
|---|---|---|
| `16:03:39` | Normal | All five tasks succeeded. `candidate` moved from version 3 to 4 |
| `16:05:00` | `QUALITY_GATE_F1=0.60` | `quality_gate` failed (F1 0.492). Run tagged `quality_gate=failed`; `candidate` stayed at 4 |
| `16:06:22` | Row `bad,row` appended to the raw CSV | `validate_dataset` failed with "data contains null values"; four tasks `upstream_failed` |
| `16:42:13` | Triggered by the agent after confirmation | All five succeeded; `candidate` moved to 5, `champion` stayed at 4 |

## 4. Agent conversations

Each JSON file holds the question, the answer, and every tool envelope.

| # | File | Question | Tools called | Outcome |
|---|---|---|---|---|
| 1 | `01-dataset-provenance.json` | Which dataset did the latest run use, and how is it reproduced? | `list_runs`, `get_dataset_for_run`, `get_reproduction_steps` | `data-v2`, hash verified against Git, exact restore commands |
| 2 | `02-model-selection.json` | Compare the last five runs and recommend a candidate | `list_runs` | Ranked table; recommended `dd037273` (F1 0.495, gate passed); noted that a run with equal F1 failed its gate |
| 3 | `03-pipeline-execution.json`, `03-confirmation.json`, `03-pipeline-result.json` | Trigger the training pipeline and report the result | `trigger_dag` → confirm → `get_dag_run_status`, `get_registry_info` | Pending action until confirmed; the run count stayed at 3 before confirmation. Run succeeded; version 5 became `candidate` and was correctly reported as not promoted |
| 4 | `04-deployment-diagnosis.json` | Is the prediction API healthy? | `list_workloads`, `inspect_services`, `find_unhealthy_pods`, `get_deployed_version` | Healthy, 2/2 ready; noted the earlier events belonged to a Pod that no longer exists |
| 5 | `05-multi-tool.json` | Find the best recent model, its dataset, the deployment status, and whether retraining is needed | `get_best_run`, `get_deployed_version`, `detect_data_changes`, `get_dag_run_status`, `get_retraining_signals` | Evidence from all four systems; answered in Known facts, Uncertainties and Recommended actions; **no pipeline run started** (count 3 before and after) |

Additional recorded exchanges:
- `extra-dataset-comparison.json`: the v1-to-v2 diff, and which runs used each version.
- `extra-restart-request.json` and `extra-restart-confirmed.json`: a restart held until confirmed. It then rolled both Pods, and a second confirmation returned 404.

**Iterating on the mandatory scenario.** The first version of the system prompt produced correct facts, but the evidence didn't include Airflow and the answer didn't use the three required sections. The second version fixed the sections, but the answer mentioned a `data-v3` that doesn't exist. The final prompt makes `get_dag_run_status` and `detect_data_changes` required for retraining questions, and forbids naming versions that no tool returned. All three outputs were checked against the live systems.

## 5. Failure investigations

### Failure 1: Deployment rolled to a missing image

- **Injected.** `kubectl -n mlops-app set image deployment/wine-api wine-api=wine-api:0.2.0`, a tag that was never built.
- **Observed.** A new Pod in `ErrImagePull`, while the two old Pods kept serving.
- **Asked.** "Users say the wine prediction API might be broken after a recent change. Is the deployment healthy? Diagnose any problem." Recorded in `failure-1-bad-image.json`.
- **The agent checked** `list_workloads`, `find_unhealthy_pods`, `inspect_services` and `get_deployed_version`.
- **Evidence it cited:**
  - Pod `wine-api-646bf5746f-v68lv` waiting with `ErrImagePull` on image `wine-api:0.2.0`
  - the warning Event "pull access denied, repository does not exist"
  - the Service matching 3 Pods, of which only 2 were ready
  - the two Pods still running `wine-api:0.1.0`
- **Root cause it stated.** The Deployment was changed to an image tag that doesn't exist in any reachable registry. The service was degraded but still up, because the old ReplicaSet was still serving.
- **Its recommendation.** Verify or build the image, or revert to `0.1.0`. It took no action without confirmation.
- **Resolution.** `kubectl -n mlops-app rollout undo deployment/wine-api`. Afterwards there were 2/2 Pods ready, and `04-deployment-diagnosis.json` confirms the healthy state.
- **Bug it exposed.** The agent reported a contradiction: `get_deployed_version` said `rollout_complete: true` during the broken rollout, because the tool compared only available replicas with desired ones. The tool now also requires `updated == desired` and `unavailable == 0`, and `test_stuck_rollout_is_not_reported_complete` covers this case. This is the intended effect of grounding: the inconsistency appeared in the evidence and was named, rather than hidden behind a confident summary.

### Failure 2: MLflow tracking server down

- **Injected.** `docker compose stop mlflow`.
- **Asked.** "Which model has the best F1 score, and what dataset was it trained on?" Recorded in `failure-2-mlflow-down.json`.
- **The agent checked** `get_best_run`, `list_runs` and `get_registry_info`. Each returned `status: "unavailable"` within about 0.5 s, because MLflow's HTTP retries are disabled for the agent so that a dead server fails fast.
- **Result.**
  - The answer stated that no facts were available and named all three failed tools.
  - It invented no F1 or run IDs. Every envelope has `data: null`.
  - It recommended checking MLflow's reachability.
  - The code-generated note listed `get_best_run (unavailable), list_runs (unavailable), get_registry_info (unavailable)`.
- **Resolution.** `docker compose start mlflow`. Later conversations show MLflow tools returning `ok`.

### Additional: bad training data

`failure-3-bad-data.json`. After the malformed row was injected, the agent called `get_dag_run_status`, `get_failed_tasks` and `get_task_diagnostics`. It quoted the task log line `data contains null values` and the `upstream_failed` tasks, and recommended fixing the data before re-running. The data was restored with `dvc checkout -f data/raw/wine.csv.dvc`.

### Agent errors observed

These appeared in the recorded answers. None of them was a fabricated fact, but each is a real weakness:
- In `05-multi-tool.json`, it inferred that the earlier validation failure "is not a dirty-data issue" because the data is clean *now*. That used the current state to explain a past event, and the data had in fact been restored after the failure.
- In `03-pipeline-result.json`, it offered to "trigger a deployment rollout" to promote version 5. No tool promotes a model, so this overstated what the agent can do.
- In `02-model-selection.json`, the table caption said "newest first" while the rows were ranked by F1.

## 6. Limitations and improvements

| Limitation | Improvement |
|---|---|
| Local single-node cluster and Compose stack with default Airflow credentials | A managed cluster, secrets from a vault, TLS |
| Local DVC remote; a fresh clone rebuilds the data | Object storage remote (S3, GCS) |
| One train/test split, small dataset, untuned models | Cross-validation and a hyperparameter search logged to MLflow |
| Pending actions in memory, single process, 5-minute expiry | A shared store such as Redis, with an audit trail |
| Short-lived Kubernetes token, recreated by hand | A token refreshed by a controller, or workload identity |
| No production metrics, so degradation can't be judged | Export prediction logs and live F1, and add a tool that reads them |
| No tool to promote a model to `champion` or deploy it | A confirmed promotion tool that updates the alias, rebuilds and rolls out |
| Tool choice and inferences vary between LLM runs | Plans enforced in code for known question types, and evaluation sets of questions scored automatically |
| Model and image versions linked by a label set by hand | Image tags built from the registry version in CI |
