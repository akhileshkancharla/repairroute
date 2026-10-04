"""HTTP API for RepairRoute's trained APS inspection decision engine."""

from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd

from repairroute.decision import capacity_curve, outcome_at_capacity, random_expected_cost

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
MAX_BATCH_SIZE = 1000

app = FastAPI(
    title="RepairRoute API",
    description="Cost-aware inspection prioritisation for recorded Scania APS faults.",
    version="0.1.0",
)
allowed_origins = [item.strip() for item in os.getenv("REPAIRROUTE_CORS_ORIGINS", "").split(",") if item.strip()]
if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


class ScoreRequest(BaseModel):
    records: list[dict[str, float | None]] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


@app.get("/", include_in_schema=False)
def website():
    return FileResponse(FRONTEND / "index.html")


@lru_cache(maxsize=1)
def historical_data():
    try:
        scores = pd.read_csv(ARTIFACTS / "historical_scores.csv")
        metrics = json.loads((ARTIFACTS / "metrics.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not load historical artifacts: {exc}") from exc
    expected = {"record_id", "priority_score", "is_aps"}
    if not expected.issubset(scores.columns):
        raise RuntimeError("Historical score artifact has missing columns")
    if scores["record_id"].duplicated().any():
        raise RuntimeError("Historical score artifact has duplicate record IDs")
    curve = capacity_curve(scores["priority_score"].to_numpy(), scores["is_aps"].to_numpy().astype(bool))
    ranked = scores.iloc[curve["order"]].reset_index(drop=True)
    return ranked, metrics, curve


@lru_cache(maxsize=1)
def model_artifact():
    try:
        payload = joblib.load(ARTIFACTS / "model.joblib")
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Could not load trained model: {exc}") from exc
    if "model" not in payload or "feature_columns" not in payload:
        raise RuntimeError("Trained model artifact is incomplete")
    return payload


@app.get("/health")
def health():
    historical_data()
    model_artifact()
    return {"status": "ok"}


@app.get("/v1/metadata")
def metadata():
    ranked, metrics, _ = historical_data()
    return {
        "historical_records": len(ranked),
        "historical_aps_faults": metrics["test_aps_faults"],
        "feature_count": len(model_artifact()["feature_columns"]),
        "model": metrics["model"],
        "score_label": "APS priority score",
        "fixed_test_policy": metrics["fixed_test_policy"],
        "test_accuracy": metrics["test_accuracy"],
        "test_balanced_accuracy": metrics["test_balanced_accuracy"],
        "test_precision": metrics["test_precision"],
        "test_recall": metrics["test_recall"],
        "test_f1": metrics["test_f1"],
        "test_specificity": metrics["test_specificity"],
        "test_confusion_matrix": metrics["test_confusion_matrix"],
        "test_average_precision": metrics["test_average_precision"],
        "test_roc_auc": metrics["test_roc_auc"],
        "costs": {"unnecessary_inspection": 10, "missed_aps_fault": 500, "unit": "benchmark units"},
        "source": metrics["data_source"],
        "note": "Fixed test policy was selected on validation data. Capacity exploration is retrospective.",
    }


@app.get("/v1/historical/decision")
def historical_decision(capacity: int = Query(ge=0)):
    ranked, metrics, curve = historical_data()
    if capacity > len(ranked):
        raise HTTPException(status_code=422, detail=f"Capacity must be at most {len(ranked)}")
    outcome = outcome_at_capacity(curve, capacity)
    return {
        "capacity": capacity,
        "batch_size": len(ranked),
        "outcome": outcome,
        "comparisons": {
            "inspect_all": {"inspections": len(ranked), "cost": metrics["test_inspect_all_cost"]},
            "inspect_none": {"inspections": 0, "cost": metrics["test_inspect_none_cost"]},
            "random_same_capacity_expected_cost": round(
                random_expected_cost(len(ranked), metrics["test_aps_faults"], capacity), 2
            ),
        },
        "next_record_id": str(ranked.iloc[capacity]["record_id"]) if capacity < len(ranked) else None,
        "simulation": "retrospective on held-out historical records",
    }


@app.get("/v1/historical/queue")
def historical_queue(
    capacity: int = Query(ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    ranked, _, _ = historical_data()
    if capacity > len(ranked):
        raise HTTPException(status_code=422, detail=f"Capacity must be at most {len(ranked)}")
    selected = ranked.iloc[offset : min(offset + limit, capacity)] if offset < capacity else ranked.iloc[0:0]
    records = [
        {"rank": offset + i + 1, "record_id": str(row.record_id), "priority_score": float(row.priority_score)}
        for i, row in enumerate(selected.itertuples(index=False))
    ]
    return {"capacity": capacity, "total_selected": capacity, "offset": offset, "records": records}


@app.get("/v1/historical/curve")
def historical_curve(
    points: int = Query(default=201, ge=2, le=1001),
    max_capacity: int | None = Query(default=None, ge=1),
):
    ranked, _, curve = historical_data()
    end = len(ranked) if max_capacity is None else min(max_capacity, len(ranked))
    capacities = np.unique(np.linspace(0, end, num=points, dtype=int))
    return {
        "batch_size": len(ranked),
        "max_capacity": end,
        "points": [
            {"capacity": int(k), "cost": int(curve["cost"][k]), "caught": int(curve["caught"][k]),
             "missed": int(curve["missed"][k])}
            for k in capacities
        ],
        "simulation": "retrospective on held-out historical records",
    }


@app.get("/v1/sample-batch", include_in_schema=False)
def sample_batch():
    return FileResponse(
        ARTIFACTS / "sample_batch.csv", media_type="text/csv",
        filename="repairroute-example-batch.csv",
    )


@app.post("/v1/score")
def score_batch(request: ScoreRequest):
    payload = model_artifact()
    columns = payload["feature_columns"]
    expected = set(columns)
    for index, record in enumerate(request.records):
        if set(record) != expected:
            missing = sorted(expected - set(record))
            extra = sorted(set(record) - expected)
            raise HTTPException(
                status_code=422,
                detail={"record_index": index, "missing_columns": missing, "extra_columns": extra},
            )
    features = pd.DataFrame(request.records, columns=columns, dtype="float32")
    if np.isinf(features.to_numpy()).any():
        raise HTTPException(status_code=422, detail="Feature values must be finite numbers or null")
    scores = payload["model"].predict_proba(features)[:, 1]
    order = np.argsort(-scores, kind="stable")
    return {
        "count": len(scores),
        "records": [
            {"input_index": int(index), "rank": rank + 1, "priority_score": float(scores[index])}
            for rank, index in enumerate(order)
        ],
        "note": "Priority scores rank existing faults for APS inspection; they are not calibrated probabilities.",
    }
