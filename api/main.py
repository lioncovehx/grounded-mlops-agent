"""Inference service for the wine quality model."""

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "model/model.joblib"))
MODEL_VERSION = os.environ.get("MODEL_VERSION", "unknown")

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger("wine-api")
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # A missing or broken model leaves the service live but not ready, so
    # Kubernetes keeps traffic away and the reason shows in the logs.
    try:
        state["model"] = joblib.load(MODEL_PATH)
        log.info("loaded model version %s from %s", MODEL_VERSION, MODEL_PATH)
    except Exception:
        log.exception("could not load model from %s", MODEL_PATH)
    yield
    state.clear()


app = FastAPI(title="Wine quality inference", lifespan=lifespan)


class Wine(BaseModel):
    """The 11 physicochemical features, in training column order.

    Bounds are generous physical limits around the observed data, to reject nonsense.
    """

    model_config = ConfigDict(extra="forbid")

    fixed_acidity: float = Field(ge=0, le=25)
    volatile_acidity: float = Field(ge=0, le=3)
    citric_acid: float = Field(ge=0, le=3)
    residual_sugar: float = Field(ge=0, le=100)
    chlorides: float = Field(ge=0, le=1)
    free_sulfur_dioxide: float = Field(ge=0, le=500)
    total_sulfur_dioxide: float = Field(ge=0, le=600)
    density: float = Field(ge=0.9, le=1.1)
    pH: float = Field(ge=2, le=5)
    sulphates: float = Field(ge=0, le=3)
    alcohol: float = Field(ge=5, le=20)


@app.post("/predict")
def predict(wine: Wine) -> dict:
    if "model" not in state:
        raise HTTPException(503, "model not loaded")
    features = wine.model_dump()
    good = bool(state["model"].predict(pd.DataFrame([features]))[0])
    # One JSON line per prediction: the input record used for drift monitoring.
    log.info(
        json.dumps(
            {
                "event": "prediction",
                "model_version": MODEL_VERSION,
                "features": features,
                "prediction": good,
            }
        )
    )
    return {"good_quality": good, "model_version": MODEL_VERSION}


@app.get("/health/live")
def live() -> dict:
    return {"status": "ok"}


@app.get("/health/ready")
def ready() -> dict:
    if "model" not in state:
        raise HTTPException(503, "model not loaded")
    return {"status": "ready", "model_version": MODEL_VERSION}


@app.get("/version")
def version() -> dict:
    return {"model_version": MODEL_VERSION}
