"""
AI4I Track 3 — AgriRisk classifier training.

Scope decision: risk classification ONLY. An earlier yield regressor was
dropped after feature-importance analysis showed 92% of its R^2 came from
a single crop-type flag (Tomatoes ~14 t/ha vs ~1-3 t/ha for everything
else) — the model was a crop lookup table, not a climate-risk model.
Keeping a metric like that in the submission would fail our own AI-fit
justification, so it's cut.

Target reframe: risk_level (Low/Medium/High) collapsed to binary
(Low vs Elevated). The original 3-class target has only 15 High-risk
rows out of 360 — verified this produces 0.00 precision/recall/f1 on
High under 5-fold CV. Binary reframe is the only version we can defend.
"""

import pathlib
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

ROOT = pathlib.Path(__file__).parent.parent
DATA = ROOT / "data" / "agriculture_climate_market_signals.csv"
MODELS_DIR = ROOT / "models"

# climate_crop_risk_score_0_100 is deliberately excluded — it's a
# pre-computed composite that directly encodes the risk_level target
# (data leakage), and in production a farmer/USSD caller wouldn't have
# it available anyway.
FEATURES = [
    "rainfall_mm", "ndvi_proxy_0_1", "pest_incidents_reported",
    "irrigation_coverage_pct", "input_availability_score_0_100",
    "avg_farmgate_price_usd_per_tonne", "month_num", "crop", "province",
]
NUMERIC = FEATURES[:-2]
CATEGORICAL = ["crop", "province"]


def load_data():
    df = pd.read_csv(DATA)
    # Plain string split instead of pd.to_datetime — sidesteps a pandas
    # 3.0.2 / PyArrow string-backend access-violation crash on Windows
    # that occurs even with an explicit format= argument.
    df["month_num"] = df["month"].astype(str).str.split("-").str[1].astype(int)
    df["risk_binary"] = df["risk_level"].apply(lambda x: "Low" if x == "Low" else "Elevated")
    return df


def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])


def baseline_rule_accuracy(df):
    """Naive rainfall-only threshold — the thing we have to beat to justify ML."""
    rule = pd.cut(df["rainfall_mm"], bins=[-1, 40, 9999], labels=["Elevated", "Low"])
    acc = (rule == df["risk_binary"]).mean()
    print(f"[Baseline] Rainfall-threshold accuracy: {acc:.1%} (justify ML against this)")
    return acc


def main():
    df = load_data()
    baseline_rule_accuracy(df)

    X = df[FEATURES]
    y = df["risk_binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    sw = compute_sample_weight("balanced", y_train)

    pipe = Pipeline([
        ("pre", build_preprocessor()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42
        )),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="balanced_accuracy")
    print(f"\n5-fold CV balanced accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    pipe.fit(X_train, y_train, clf__sample_weight=sw)
    preds = pipe.predict(X_test)
    print("\nTest set report:")
    print(classification_report(y_test, preds))

    cat_names = pipe.named_steps["pre"].named_transformers_["cat"].get_feature_names_out(CATEGORICAL)
    feat_names = NUMERIC + list(cat_names)
    imp = pipe.named_steps["clf"].feature_importances_
    print("\nTop features:")
    for f, i in sorted(zip(feat_names, imp), key=lambda x: -x[1])[:8]:
        print(f"  {f:35s} {i:.3f}")

    MODELS_DIR.mkdir(exist_ok=True)
    with open(MODELS_DIR / "classifier.pkl", "wb") as f:
        pickle.dump({"pipeline": pipe}, f)
    print("\n[Save] classifier.pkl written")


if __name__ == "__main__":
    main()
