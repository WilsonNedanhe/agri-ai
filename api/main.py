"""
FastAPI backend for the AgriRisk classification service.
Endpoints:
  POST /predict    — risk classification from raw field inputs
  GET  /districts  — district risk summary (for internal review, not USSD)
  GET  /health     — model metrics + dataset disclosure
"""

import json
import pathlib
import pickle

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = pathlib.Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
DATA_PATH = ROOT / "data" / "agriculture_climate_market_signals.csv"

app = FastAPI(
    title="AgriRisk — Crop Risk Classification API",
    version="1.0.0",
    description="Binary risk classification (Low/Elevated) for USSD-delivered farmer alerts.",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_bundle = None
_df = None


def _load():
    global _bundle, _df
    with open(MODELS_DIR / "classifier.pkl", "rb") as f:
        _bundle = pickle.load(f)
    _df = pd.read_csv(DATA_PATH)
    _df["month_num"] = pd.to_datetime(_df["month"]).dt.month


@app.on_event("startup")
def startup():
    _load()


class PredictRequest(BaseModel):
    # Every field here is something a farmer/extension officer can supply
    # via a USSD menu or an ESP32 sensor relay — no derived/composite
    # scores required, unlike the earlier draft.
    rainfall_mm: float = Field(..., ge=0)
    ndvi_proxy_0_1: float = Field(..., ge=0, le=1)
    pest_incidents_reported: int = Field(..., ge=0)
    irrigation_coverage_pct: float = Field(..., ge=0, le=100)
    input_availability_score_0_100: float = Field(..., ge=0, le=100)
    avg_farmgate_price_usd_per_tonne: float = Field(..., gt=0)
    month: int = Field(..., ge=1, le=12)
    crop: str
    province: str


class PredictResponse(BaseModel):
    risk_level: str
    risk_probabilities: dict
    action_recommendation: str


def _build_action(risk_level: str, req: PredictRequest) -> str:
    if risk_level == "Elevated":
        parts = ["ELEVATED RISK: act this week."]
        if req.pest_incidents_reported > 10:
            parts.append("Pest pressure high — inspect and treat.")
        if req.rainfall_mm < 40:
            parts.append("Rainfall low — prioritise available irrigation.")
        if req.input_availability_score_0_100 < 50:
            parts.append("Input access limited — contact extension officer.")
        return " ".join(parts)
    return "LOW RISK: continue standard practices, recheck next month."


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    row = pd.DataFrame([{
        "rainfall_mm": req.rainfall_mm,
        "ndvi_proxy_0_1": req.ndvi_proxy_0_1,
        "pest_incidents_reported": req.pest_incidents_reported,
        "irrigation_coverage_pct": req.irrigation_coverage_pct,
        "input_availability_score_0_100": req.input_availability_score_0_100,
        "avg_farmgate_price_usd_per_tonne": req.avg_farmgate_price_usd_per_tonne,
        "month_num": req.month,
        "crop": req.crop,
        "province": req.province,
    }])

    pipe = _bundle["pipeline"]
    risk = pipe.predict(row)[0]
    proba = pipe.predict_proba(row)[0]
    proba_dict = {cls: round(float(p), 3) for cls, p in zip(pipe.classes_, proba)}

    return PredictResponse(
        risk_level=risk,
        risk_probabilities=proba_dict,
        action_recommendation=_build_action(risk, req),
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "dataset_statement": (
            "Trained on AI4I Design Track synthetic sample data (360 rows, "
            "aggregate, not official statistics). A yield-prediction module "
            "was scoped and dropped after feature-importance analysis showed "
            "it was dominated by crop-type identity rather than climate "
            "signal. Production deployment requires real AGRITEX/FAO/CIMMYT "
            "field data before this risk score can be trusted operationally."
        ),
    }