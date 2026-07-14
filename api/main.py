"""
FastAPI backend for the AgriRisk classification service.
Endpoints:
  POST /predict    — risk classification from raw field inputs
  GET  /districts  — province-level risk summary (for internal review, not USSD)
  GET  /health     — model metrics + dataset disclosure
"""

import pathlib
import pickle

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
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
    _df["month_num"] = _df["month"].astype(str).str.split("-").str[1].astype(int)
    _df["risk_binary"] = _df["risk_level"].apply(lambda x: "Low" if x == "Low" else "Elevated")


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


@app.get("/districts")
def districts(province: str | None = Query(default=None)):
    """
    District-level descriptive statistics computed directly from recorded
    data — no model inference call. avg_yield_t_per_ha is the historical
    average of the recorded estimated_yield_t_per_ha column: a descriptive
    stat over past data, not a model prediction. This is distinct from the
    yield-REGRESSION MODEL, which was scoped and dropped (see /health) after
    a feature-importance audit showed it was dominated by crop identity
    rather than climate signal. No such model is invoked here.
    """
    if _df is None:
        raise HTTPException(status_code=503, detail="Data not loaded yet.")

    df = _df
    if province:
        df = df[df["province"] == province]
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No records for province '{province}'.")

    grouped = df.groupby(["province", "district"])
    results = []
    for (prov, dist), g in grouped:
        elevated_pct = round((g["risk_binary"] == "Elevated").mean() * 100, 1)
        dominant_level = "Elevated" if elevated_pct >= 50 else "Low"
        crop_elevated_rate = (
            g.groupby("crop")["risk_binary"]
            .apply(lambda s: (s == "Elevated").mean())
            .sort_values(ascending=False)
        )
        high_risk_crops = crop_elevated_rate[crop_elevated_rate > 0.5].index.tolist()

        results.append({
            "province": prov,
            "district": dist,
            "record_count": int(len(g)),
            "pct_elevated": elevated_pct,
            "dominant_risk_level": dominant_level,
            "high_risk_crops": high_risk_crops,
            "avg_yield_t_per_ha": round(float(g["estimated_yield_t_per_ha"].mean()), 2),
        })

    return results


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
