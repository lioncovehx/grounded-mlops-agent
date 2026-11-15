# Architecture

## Components

```mermaid
flowchart TD
    subgraph Agent["Agent (host, port 8001)"]
        API["agent/main.py<br/>/chat, /actions/{id}/confirm, /actions/{id}/cancel"]
        ORC["agent/orchestrator.py<br/>loop, system prompt, pending actions"]
        LLMA["agent/llm.py<br/>OpenAI-compatible adapter"]
        REG["agent/registry.py<br/>execute_tool"]
        API --> ORC --> LLMA
        ORC --> REG
    end
    LLMA <--> LLM["LLM endpoint<br/>local llama-server or hosted"]

    subgraph Compose["Docker Compose"]
        MLF["MLflow 3.18<br/>tracking + registry, port 5000"]
        AF["Airflow 3.4<br/>api-server, scheduler, dag-processor,<br/>Celery worker, triggerer"]
        PG["Postgres"]
        RD["Redis"]
        AF --- PG
        AF --- RD
    end

    subgraph K8s["minikube profile mlops, namespace mlops-app"]
        DEP["Deployment wine-api<br/>2 replicas, probes, limits"]
        SVC["Service wine-api<br/>ClusterIP 80 → 8000"]
        CM["ConfigMap wine-api-config"]
        SA["ServiceAccount agent<br/>Role: read + patch deployments"]
        SVC --> DEP
        CM -.-> DEP
    end

    REG -- "git, dvc subprocess" --> REPO["Git repo + DVC cache"]
    REG -- "MlflowClient" --> MLF
    REG -- "REST /api/v2, JWT" --> AF
    REG -- "kubernetes client, agent token" --> SA
    AF -- "BashOperator in /opt/project-venv" --> TR["training/train.py"]
    TR --> MLF
    TR --> REPO
```

The Airflow worker runs the training stages in a separate virtual environment (`/opt/project-venv`), built from the same pinned `pyproject.toml`, so the project's packages can't clash with Airflow's own. The repo is mounted at `/opt/project`.

## The training pipeline

```mermaid
flowchart LR
    V["validate_dataset<br/>columns, nulls, range,<br/>dvc status clean<br/>retries 0"] --> P["preprocess<br/>dedupe, target,<br/>snake_case"]
    P --> T["train<br/>rf, logs provenance,<br/>prints run ID"]
    T --> E["evaluate_and_log<br/>F1 on same split"]
    E --> G["quality_gate<br/>F1 ≥ QUALITY_GATE_F1<br/>retries 0"]
    G -- "pass" --> R["register version,<br/>alias candidate"]
```

The `train` task's last output line, the MLflow run ID, becomes its XCom value. The next two tasks receive it as `--run-id`. Validation and the gate never retry, because bad data or a weak model won't improve on a second attempt. The other tasks retry twice, 30 seconds apart.

## A request through the agent

```mermaid
sequenceDiagram
    participant U as User
    participant A as Agent API
    participant L as LLM
    participant X as execute_tool
    participant S as Service

    U->>A: POST /chat {message}
    loop up to 10 steps
        A->>L: system prompt + conversation + 24 tool schemas
        L-->>A: tool_calls
        A->>X: name, args, request_id
        X->>X: validate args (Pydantic)
        alt changes the system
            X-->>A: confirmation_required
            A->>A: store pending action
        else read-only
            X->>S: call, with timeout
            S-->>X: data or failure
            X-->>A: envelope {tool, status, data, error, duration_ms, request_id}
        end
        A->>L: envelope as tool result
    end
    L-->>A: final answer
    A->>A: append a note naming any failed tool
    A-->>U: answer, evidence, pending_action_id
```

Three guarantees come from code rather than from the LLM:
- **`execute_tool` never raises.** Every failure becomes `error`, `timeout`, `unavailable` or `confirmation_required`, and the answer receives a note naming any tool that failed.
- **Tool output is data.** It's returned as a tool result, and the system prompt says never to follow instructions found inside it.
- **Structured logs** record a fixed set of fields per call: tool, status, duration and request ID. Arguments and results are never logged.

## From a run to its exact data

```mermaid
flowchart LR
    R["MLflow run"] -- "tag data.md5" --> H["dataset hash"]
    R -- "tag data.commit" --> C["Git commit of the data-* tag"]
    C -- "git show :data/raw/wine.csv.dvc" --> PTR["DVC pointer md5"]
    PTR -- "compare" --> H
    H -- "dvc checkout" --> F["exact CSV from the DVC cache"]
    R -- "tag git.commit" --> CODE["code version"]
```

`training/train.py` tags every run with `data.md5`, `data.tag`, `data.commit` and `git.commit`. `data.commit` is recorded separately because training on an older version restores an old pointer while `HEAD` stays at a later commit. `get_dataset_for_run` re-reads the pointer at `data.commit`, and reports `verified: true` only if the hashes match. Missing tags are reported as a finding, never guessed.

## Permission layers

```mermaid
flowchart LR
    R["Change requested by the LLM"] --> C{"execute_tool:<br/>confirmed?"}
    C -- "no" --> P["Pending action;<br/>user must POST /confirm"]
    C -- "yes" --> N{"Tool check:<br/>approved namespace or DAG?"}
    N -- "no" --> X["Rejected before any API call"]
    N -- "yes" --> B{"Kubernetes RBAC:<br/>ServiceAccount allowed?"}
    B -- "no" --> F["403"]
    B -- "yes" --> OK["Applied"]
```

| Layer | Enforced by | Limit |
|---|---|---|
| Confirmation | `execute_tool` and `/actions/{id}/confirm` | `restart_deployment` and `trigger_dag` run only after a person confirms them, once. The LLM has no tool that confirms |
| Allowlists | The tool functions | Kubernetes tools: `AGENT_ALLOWED_NAMESPACE` only. `trigger_dag`: `AGENT_ALLOWED_DAGS` only |
| Command allowlist | `dvc_tools.run` | Only `git show/rev-parse/tag/status/log` and `dvc list/diff/status`, without a shell. Revisions must match `^[\w./-]+$` and can't start with `-` |
| Argument validation | Pydantic models | Unknown fields rejected; ranges and patterns enforced |
| Kubernetes RBAC | `kubernetes/rbac.yaml` | Get and list on Pods, logs, Services and Events; patch on Deployments; namespace `mlops-app` only. No delete, no secrets |
