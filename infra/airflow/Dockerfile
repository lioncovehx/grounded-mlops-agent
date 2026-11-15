# Airflow with the project's dependencies in a separate virtual environment,
# so they can't clash with Airflow's own packages. Tasks run /opt/project-venv/bin/python.
FROM apache/airflow:3.4.0-python3.12

USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*
# The repo is mounted from the host, so Git sees a different owner.
RUN git config --system --add safe.directory /opt/project

COPY pyproject.toml README.md /opt/build/
RUN python -m venv /opt/project-venv \
 && /opt/project-venv/bin/pip install --no-cache-dir /opt/build \
 && rm -rf /opt/build
USER airflow
