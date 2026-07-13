"""
FastAPI backend for AgriAI dual-output prediction service.
Endpoints:
  POST /predict       — risk classification + yield regression
  GET  /districts     — list districts with current risk summary
  GET  /trends        — monthly aggregate trends
  GET  /health        — model metrics + version
"""

import json
import pathlib
import pickle

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = pathlib.Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
DATA_PATH = ROOT / "data" / "agriculture_climate_market_signals.csv"

app = FastAPI(
    title="AgriAI — Agriculture Climate & Market Signal API",
    version="1.0.0",
    description="Dual-output ML: risk classification + yield regression for Zimbabwe districts.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------- #
# Load artifacts at startup
# --------------------------------------------------------------------------- #
_clf_bundle = None
_reg_bundle = None
_metrics = None
_df = None


def _load():
    global _clf_bundle, _reg_bundle, _metrics, _df
    with open(MODELS_DIR / "classifier.pkl", "rb") as f:
        _clf_bundle = pickle.load(f)
    with open(MODELS_DIR / "regressor.pkl", "rb") as f:
        _reg_bundle = pickle.load(f)
    with open(MODELS_DIR / "metrics.json") as f:
        _metrics = json.load(f)
    _df = pd.read_csv(DATA_PATH)
    _df["month_num"] = pd.to_datetime(_df["month"]).dt.month


@app.on_event("startup")
def startup():
    _load()


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class PredictRequest(BaseModel):
    rainfall_mm: float = Field(..., ge=0, description="Rainfall in mm")
    ndvi_proxy_0_1: float = Field(..., ge=0, le=1, description="NDVI vegetation index 0-1")
    pest_incidents_reported: int = Field(..., ge=0)
    irrigation_coverage_pct: float = Field(..., ge=0, le=100)
    input_availability_score_0_100: float = Field(..., ge=0, le=100)
    avg_farmgate_price_usd_per_tonne: float = Field(..., gt=0)
    climate_crop_risk_score_0_100: float = Field(..., ge=0, le=100)
    month: int = Field(..., ge=1, le=12, description="Month number 1-12")
    crop: str = Field(..., description="e.g. Maize, Tomatoes, Groundnuts")
    province: str = Field(..., description="e.g. Harare, Manicaland")


class PredictResponse(BaseModel):
    risk_level: str
    risk_probabilities: dict
    estimated_yield_t_per_ha: float
    action_recommendation: str


class DistrictRisk(BaseModel):
    province: str
    district: str
    avg_risk_score: float
    dominant_risk_level: str
    high_risk_crops: list[str]
    avg_yield_t_per_ha: float


class TrendPoint(BaseModel):
    month: str
    avg_rainfall_mm: float
    avg_ndvi: float
    avg_risk_score: float
    avg_yield_t_per_ha: float
    high_risk_count: int


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
CLF_FEATURES = [
    "rainfall_mm", "ndvi_proxy_0_1", "pest_incidents_reported",
    "irrigation_coverage_pct", "input_availability_score_0_100",
    "avg_farmgate_price_usd_per_tonne", "month_num", "crop", "province",
]
REG_FEATURES = CLF_FEATURES + ["climate_crop_risk_score_0_100"]


def _build_action(risk_level: str, row: PredictRequest) -> str:
    if risk_level == "High":
        parts = ["URGENT: escalate to extension officer."]
        if row.pest_incidents_reported > 15:
            parts.append("Deploy pest control within 48 h.")
        if row.rainfall_mm < 20:
            parts.append("Activate emergency irrigation.")
        if row.input_availability_score_0_100 < 50:
            parts.append("Request emergency input resupply.")
        return " ".join(parts)
    if risk_level == "Medium":
        parts = ["MONITOR: check field conditions in 7 days."]
        if row.ndvi_proxy_0_1 < 0.35:
            parts.append("NDVI below threshold — scout for stress.")
        return " ".join(parts)
    return "LOW RISK: continue standard agronomic practices."


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    base = {
        "rainfall_mm": req.rainfall_mm,
        "ndvi_proxy_0_1": req.ndvi_proxy_0_1,
        "pest_incidents_reported": req.pest_incidents_reported,
        "irrigation_coverage_pct": req.irrigation_coverage_pct,
        "input_availability_score_0_100": req.input_availability_score_0_100,
        "avg_farmgate_price_usd_per_tonne": req.avg_farmgate_price_usd_per_tonne,
        "climate_crop_risk_score_0_100": req.climate_crop_risk_score_0_100,
        "month_num": req.month,
        "crop": req.crop,
        "province": req.province,
    }
    row_clf = pd.DataFrame([{k: base[k] for k in CLF_FEATURES}])
    row_reg = pd.DataFrame([{k: base[k] for k in REG_FEATURES}])

    clf_pipe = _clf_bundle["pipeline"]
    le = _clf_bundle["label_encoder"]
    reg_pipe = _reg_bundle["pipeline"]

    risk_enc = clf_pipe.predict(row_clf)[0]
    risk_proba = clf_pipe.predict_proba(row_clf)[0]
    risk_level = le.inverse_transform([risk_enc])[0]
    yield_pred = float(reg_pipe.predict(row_reg)[0])

    proba_dict = {cls: round(float(p), 3) for cls, p in zip(le.classes_, risk_proba)}

    return PredictResponse(
        risk_level=risk_level,
        risk_probabilities=proba_dict,
        estimated_yield_t_per_ha=round(yield_pred, 2),
        action_recommendation=_build_action(risk_level, req),
    )


@app.get("/districts", response_model=list[DistrictRisk])
def districts(province: str | None = None):
    df = _df.copy()
    if province:
        df = df[df["province"].str.lower() == province.lower()]
        if df.empty:
            raise HTTPException(404, f"Province '{province}' not found")

    results = []
    for (prov, dist), g in df.groupby(["province", "district"]):
        high_crops = g[g["risk_level"] == "High"]["crop"].unique().tolist()
        results.append(DistrictRisk(
            province=prov,
            district=dist,
            avg_risk_score=round(g["climate_crop_risk_score_0_100"].mean(), 1),
            dominant_risk_level=g["risk_level"].mode()[0],
            high_risk_crops=high_crops,
            avg_yield_t_per_ha=round(g["estimated_yield_t_per_ha"].mean(), 2),
        ))

    results.sort(key=lambda x: x.avg_risk_score, reverse=True)
    return results


@app.get("/trends", response_model=list[TrendPoint])
def trends(province: str | None = None, crop: str | None = None):
    df = _df.copy()
    if province:
        df = df[df["province"].str.lower() == province.lower()]
    if crop:
        df = df[df["crop"].str.lower() == crop.lower()]
    if df.empty:
        raise HTTPException(404, "No data for the specified filters")

    out = []
    for month, g in df.groupby("month"):
        out.append(TrendPoint(
            month=month,
            avg_rainfall_mm=round(g["rainfall_mm"].mean(), 1),
            avg_ndvi=round(g["ndvi_proxy_0_1"].mean(), 3),
            avg_risk_score=round(g["climate_crop_risk_score_0_100"].mean(), 1),
            avg_yield_t_per_ha=round(g["estimated_yield_t_per_ha"].mean(), 2),
            high_risk_count=int((g["risk_level"] == "High").sum()),
        ))
    return sorted(out, key=lambda x: x.month)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {
            "classifier_accuracy": _metrics["classifier_accuracy"],
            "regressor_mae_t_per_ha": _metrics["regressor_mae"],
            "regressor_r2": _metrics["regressor_r2"],
        },
        "dataset_statement": (
            "This API uses synthetic aggregate sample data provided for the AI4I challenge. "
            "It does NOT represent official national statistics. A full pilot would require "
            "real AGRITEX field survey data, ZIMSTAT seasonal assessments, and live IoT sensor feeds."
        ),
    }
