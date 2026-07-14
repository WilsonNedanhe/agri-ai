# AgriRisk — Crop Risk Advisory API

AI-powered crop risk classification for Zimbabwe smallholder farmers, built for the POTRAZ AI4I Challenge, Track 3 — Development.

## About

Smallholder farmers in Zimbabwe make critical seasonal decisions — input purchase timing, irrigation prioritization, pest response — with limited access to consolidated climate and market signal data. AgriRisk is a lightweight FastAPI service that classifies crop risk (Low vs Elevated) from nine raw, farmer-answerable inputs, and returns a one-line actionable recommendation alongside the prediction.

The project deliberately scopes itself to what the data can actually support. A yield-prediction module was built, tested, and **dropped**: a feature-importance audit showed 92% of its apparent accuracy came from a single crop-identity flag rather than any real climate signal — a crop lookup table, not a climate-risk model. That decision, and the reasoning behind it, is disclosed in the `/health` endpoint and in the full technical proposal.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/predict` | Binary risk classification (Low/Elevated) + probabilities + action recommendation, from raw inputs |
| `GET` | `/districts` | District-level descriptive statistics (% elevated risk, dominant risk level, high-risk crops, historical average yield) — computed directly from recorded data, no model call |
| `GET` | `/health` | Model metrics and mandatory dataset provenance / limitation disclosure |

## Model

- **Algorithm:** Gradient Boosted Tree (scikit-learn `GradientBoostingClassifier`, 200 estimators, max depth 4)
- **Target:** Binary — Low / Elevated risk (collapsed from an original 3-class target; the High-risk class had only 15/360 rows and scored 0.00 precision/recall/F1 under cross-validation, so a 3-class model would have been indefensible)
- **Validation:** 5-fold cross-validated balanced accuracy of 0.804 ± 0.040
- **Why ML at all:** a single-variable rainfall threshold baseline scores 75.0% accuracy on the same data. The classifier's real value is case-level — it catches 92% of cases where rainfall alone would have signaled "safe" but pest pressure or poor irrigation access drove elevated risk anyway.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
uvicorn api.main:app --reload
```

Then open `index.html` in a browser (with the API running on `localhost:8000`) for the chat-style prediction and district-overview interface.

## Tests

```bash
pip install pytest
pytest tests/test_main.py -v
```

8 tests covering `/predict` schema and probability normalization, HTTP 422 validation rejection, unseen-category handling, `/districts` row counts and province filtering, and `/health`'s dataset disclosure.

## Known limitations (disclosed, not hidden)

- **Dataset:** trained on 360 synthetic AI4I Design Track sample rows, not official AGRITEX/FAO/CIMMYT statistics. Not yet validated against real field outcomes.
- **CORS:** currently `allow_origins=["*"]` — fine for a demo, must be restricted before any production deployment.
- **No authentication** on the API — acceptable for a Challenge submission, not for production.
- **No rate limiting.**
- **Stateless:** no persistent database; a Supabase/PostgreSQL schema is proposed in the full technical proposal for a future pilot phase.

## Roadmap

- USSD channel integration (AfricaTalking sandbox) for farmers without smartphone/data access
- Real AGRITEX/FAO/CIMMYT field data to replace the synthetic sample set
- ZCHPC compute for a full hyperparameter sweep at scale

## Full proposal

See `AgriRisk_AI4I_Proposal_Development.pdf` in this repository for the complete technical design, compliance framework, and roadmap submitted to the POTRAZ AI4I Challenge.
