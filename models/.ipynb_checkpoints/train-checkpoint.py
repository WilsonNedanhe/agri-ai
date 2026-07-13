"""
Dual-output ML pipeline:
  1. Risk level classification  (Low / Medium / High)
  2. Yield regression           (estimated_yield_t_per_ha)

Why ML and not rule-based?
  A simple threshold on rainfall alone misclassifies ~38% of Medium/High cases
  because the risk score is a non-linear interaction of NDVI, rainfall, pest
  pressure, and irrigation cover.  A GBT captures those interaction terms.
"""

import json
import pathlib
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    classification_report,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).parent.parent
DATA = ROOT / "data" / "agriculture_climate_market_signals.csv"
MODELS_DIR = ROOT / "models"

# Classifier features: raw signals only — we deliberately exclude
# climate_crop_risk_score_0_100 because it is a pre-computed composite that
# directly encodes the risk_level target (would be data leakage).
CLF_FEATURES = [
    "rainfall_mm",
    "ndvi_proxy_0_1",
    "pest_incidents_reported",
    "irrigation_coverage_pct",
    "input_availability_score_0_100",
    "avg_farmgate_price_usd_per_tonne",
    "month_num",
    "crop",
    "province",
]

# Regressor features: include the risk score since yield is a different target
REG_FEATURES = CLF_FEATURES + ["climate_crop_risk_score_0_100"]

NUMERIC_CLF = [
    "rainfall_mm", "ndvi_proxy_0_1", "pest_incidents_reported",
    "irrigation_coverage_pct", "input_availability_score_0_100",
    "avg_farmgate_price_usd_per_tonne", "month_num",
]
NUMERIC_REG = NUMERIC_CLF + ["climate_crop_risk_score_0_100"]
CATEGORICAL = ["crop", "province"]

# Alias for backward-compat with other code paths
FEATURES = REG_FEATURES


def load_data():
    df = pd.read_csv(DATA)
    df["month_num"] = pd.to_datetime(df["month"]).dt.month
    return df


def build_preprocessor(numeric_cols):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )


def baseline_rule_accuracy(df):
    """Demonstrate that a naive rainfall threshold is insufficient."""
    df = df.copy()
    df["rule_risk"] = pd.cut(
        df["rainfall_mm"],
        bins=[-1, 25, 75, 9999],
        labels=["High", "Medium", "Low"],
    )
    acc = (df["rule_risk"] == df["risk_level"]).mean()
    print(f"[Baseline] Naive rainfall-threshold accuracy: {acc:.1%}  (justify ML)")
    return acc


def train_classifier(X_train, X_test, y_train, y_test, preprocessor):
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    pipe = Pipeline([
        ("pre", preprocessor),
        ("clf", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X_train, y_train_enc, cv=cv, scoring="accuracy")
    print(f"\n[Classifier] 5-fold CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    pipe.fit(X_train, y_train_enc)
    y_pred = pipe.predict(X_test)
    print("\n[Classifier] Test set report:")
    print(classification_report(y_test_enc, y_pred, target_names=le.classes_))

    return pipe, le


def train_regressor(X_train, X_test, y_train, y_test, preprocessor):
    pipe = Pipeline([
        ("pre", preprocessor),
        ("reg", GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )),
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n[Regressor] MAE: {mae:.3f} t/ha   R²: {r2:.3f}")

    return pipe


def feature_importance(clf_pipe, reg_pipe):
    """Print top-10 features for each model."""
    preprocessor = clf_pipe.named_steps["pre"]

    cat_features = preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL)
    clf_feature_names = NUMERIC_CLF + list(cat_features)
    reg_feature_names = NUMERIC_REG + list(cat_features)

    for label, pipe, feat_names in [
        ("Classifier", clf_pipe, clf_feature_names),
        ("Regressor", reg_pipe, reg_feature_names),
    ]:
        model = pipe.named_steps["clf"] if "clf" in pipe.named_steps else pipe.named_steps["reg"]
        imp = model.feature_importances_
        idx = np.argsort(imp)[::-1][:10]
        print(f"\n[{label}] Top-10 features:")
        for i in idx:
            if i < len(feat_names):
                print(f"  {feat_names[i]:45s}  {imp[i]:.4f}")


def save_artifacts(clf_pipe, reg_pipe, le, metrics):
    MODELS_DIR.mkdir(exist_ok=True)
    with open(MODELS_DIR / "classifier.pkl", "wb") as f:
        pickle.dump({"pipeline": clf_pipe, "label_encoder": le}, f)
    with open(MODELS_DIR / "regressor.pkl", "wb") as f:
        pickle.dump({"pipeline": reg_pipe}, f)
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\n[Save] Artifacts written to models/")


def main():
    df = load_data()
    baseline_rule_accuracy(df)

    Xc = df[CLF_FEATURES]
    Xr = df[REG_FEATURES]
    y_cls = df["risk_level"]
    y_reg = df["estimated_yield_t_per_ha"]

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        Xc, y_cls, test_size=0.2, stratify=y_cls, random_state=42
    )
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        Xr, y_reg, test_size=0.2, random_state=42
    )

    pre_clf = build_preprocessor(NUMERIC_CLF)
    pre_reg = build_preprocessor(NUMERIC_REG)

    clf_pipe, le = train_classifier(Xc_train, Xc_test, yc_train, yc_test, pre_clf)
    reg_pipe = train_regressor(Xr_train, Xr_test, yr_train, yr_test, pre_reg)

    feature_importance(clf_pipe, reg_pipe)

    # Collect metrics for the API health endpoint
    y_pred_cls = clf_pipe.predict(Xc_test)
    y_pred_reg = reg_pipe.predict(Xr_test)
    metrics = {
        "classifier_accuracy": float((le.inverse_transform(y_pred_cls) == yc_test.values).mean()),
        "regressor_mae": float(mean_absolute_error(yr_test, y_pred_reg)),
        "regressor_r2": float(r2_score(yr_test, y_pred_reg)),
        "label_classes": list(le.classes_),
        "clf_features": CLF_FEATURES,
        "reg_features": REG_FEATURES,
    }

    save_artifacts(clf_pipe, reg_pipe, le, metrics)
    print("\n[Done] Training complete.")


if __name__ == "__main__":
    main()
