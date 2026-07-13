# AgriAI — Agriculture Climate & Market Signal Intelligence

Dual-output ML system for Zimbabwe district-level agricultural risk assessment
and yield prediction. Built for the AI4I Development Track (Track 3).

---

## Quick Start

```bash
# 1 — Install dependencies
pip install scikit-learn xgboost fastapi uvicorn pandas numpy

# 2 — Train models (creates models/classifier.pkl, models/regressor.pkl)
python models/train.py

# 3 — Start API server
uvicorn api.main:app --reload --port 8000

# 4 — Open dashboard
open frontend/index.html     # or double-click in Windows Explorer
```

---

## Project Structure

```
agri-ai/
├── data/
│   └── agriculture_climate_market_signals.csv   # Synthetic AI4I dataset
├── models/
│   ├── train.py                                 # ML training pipeline
│   ├── classifier.pkl                           # Risk level GBT (generated)
│   ├── regressor.pkl                            # Yield GBT (generated)
│   └── metrics.json                             # Evaluation metrics (generated)
├── api/
│   └── main.py                                  # FastAPI backend
├── frontend/
│   └── index.html                               # Single-page dashboard
└── docs/
    ├── architecture.md                          # System architecture + AI justification
    └── dataset_statement.md                     # Formal data provenance statement
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/predict` | Risk level + yield prediction |
| GET | `/districts` | All districts ranked by risk score |
| GET | `/trends` | Monthly aggregate signals |
| GET | `/health` | Model metrics + dataset statement |

### Example predict request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "rainfall_mm": 10,
    "ndvi_proxy_0_1": 0.2,
    "pest_incidents_reported": 18,
    "irrigation_coverage_pct": 5,
    "input_availability_score_0_100": 30,
    "avg_farmgate_price_usd_per_tonne": 400,
    "climate_crop_risk_score_0_100": 75,
    "month": 5,
    "crop": "Maize",
    "province": "Matabeleland North"
  }'
```

---

## Target User Personas

| Persona | What this system answers |
|---|---|
| Extension Officers | Which districts need urgent pest/input intervention right now? |
| Farmer Associations | What is the risk level and expected yield for my crop this month? |
| Food Security Planners | How are yields and risk scores trending across provinces? |

---

## Dataset Statement

Synthetic aggregate sample data provided by AI4I for challenge use only.
**Not official national statistics.** See `docs/dataset_statement.md`.

---

## Security

No API keys or secrets are committed. The API accepts unauthenticated requests
on localhost for development; add OAuth2 / API-key middleware before any
public deployment.
