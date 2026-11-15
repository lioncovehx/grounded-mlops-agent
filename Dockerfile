# Inference service: serves the promoted model on port 8000.
FROM python:3.12-slim
WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY api/ api/
COPY model/ model/

RUN useradd --create-home appuser
USER appuser

ARG MODEL_VERSION=unknown
ENV MODEL_VERSION=$MODEL_VERSION
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
