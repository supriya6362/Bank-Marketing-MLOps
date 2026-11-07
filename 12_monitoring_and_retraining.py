from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "bank-additional-full.csv"
OUT = ROOT / "outputs"
MODELS = ROOT / "models"
TARGET = "y"
RANDOM_STATE = 42

def load_data():
    # This UCI Bank Marketing file is semicolon-separated.
    df = pd.read_csv(DATA, sep=";")
    df[TARGET] = df[TARGET].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    if df[TARGET].isna().any():
        raise ValueError("Target column y contains unexpected values.")
    return df

def make_pipeline(X):
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    prep = ColumnTransformer([
        ("num", num_pipe, numeric),
        ("cat", cat_pipe, categorical)
    ])
    return Pipeline([
        ("preprocess", prep),
        ("model", RandomForestClassifier(
            n_estimators=160, max_depth=14, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

def psi_numeric(reference, current, bins=10):
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()
    if ref.empty or cur.empty or ref.nunique() < 2:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_pct = np.histogram(cur, bins=edges)[0] / len(cur)
    eps = 1e-6
    return float(np.sum((cur_pct-ref_pct) * np.log((cur_pct+eps)/(ref_pct+eps))))

def categorical_shift(reference, current):
    a = reference.fillna("MISSING").astype(str).value_counts(normalize=True)
    b = current.fillna("MISSING").astype(str).value_counts(normalize=True)
    keys = a.index.union(b.index)
    return float(0.5 * np.abs(a.reindex(keys, fill_value=0) - b.reindex(keys, fill_value=0)).sum())

def main():
    OUT.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    df = load_data()
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Baseline model and baseline evaluation.
    model = make_pipeline(X_train)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, prob))
    }
    joblib.dump(model, MODELS / "bank_marketing_model_v1.joblib")

    # Simulated incoming batch: feature distribution is changed deliberately.
    rng = np.random.default_rng(RANDOM_STATE)
    current = X_test.sample(n=min(5000, len(X_test)), random_state=RANDOM_STATE).copy()
    if "campaign" in current:
        current["campaign"] = (current["campaign"] + rng.integers(1, 5, len(current))).clip(1, None)
    if "age" in current:
        current["age"] = (current["age"] + rng.normal(4, 7, len(current))).clip(18, 95).round()
    if "job" in current:
        current["job"] = rng.choice(["admin.", "blue-collar", "technician", "services"], len(current))

    drift_rows = []
    for col in X_train.columns:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            score = psi_numeric(X_train[col], current[col])
            method = "PSI"
            threshold = 0.20
        else:
            score = categorical_shift(X_train[col], current[col])
            method = "Total variation distance"
            threshold = 0.10
        drift_rows.append({"feature": col, "method": method, "score": score,
                           "threshold": threshold, "drift_detected": bool(score > threshold)})
    drift = pd.DataFrame(drift_rows).sort_values("score", ascending=False)
    drift.to_csv(OUT / "lab12_drift_report.csv", index=False)

    drift_detected = bool(drift["drift_detected"].any())
    # For demonstration, use the full labelled source data for retraining after drift is detected.
    # The held-out test set is not used for this retraining step.
    if drift_detected:
        updated_model = make_pipeline(X)
        updated_model.fit(X, y)
        joblib.dump(updated_model, MODELS / "bank_marketing_model_v2.joblib")
        selected = "V2"
    else:
        selected = "V1"

    report = {
        "lab": "12 - Data Drift Monitoring and Model Retraining",
        "dataset": DATA.name,
        "rows": int(len(df)),
        "features": int(X.shape[1]),
        "target_column": TARGET,
        "target_distribution": {str(k): int(v) for k, v in y.value_counts().items()},
        "baseline_test_metrics": metrics,
        "drift_detected": drift_detected,
        "drifted_features": drift.loc[drift["drift_detected"], "feature"].tolist(),
        "selected_model": selected,
        "retraining_performed": drift_detected,
        "note": "Incoming feature batch is simulated for the lab demonstration; drift thresholds are illustrative."
    }
    (OUT / "lab12_monitoring_report.json").write_text(json.dumps(report, indent=2))
    print("\\nLAB 12 COMPLETED")
    print("Baseline metrics:", json.dumps(metrics, indent=2))
    print("Drift detected:", drift_detected)
    print("Drift report:", OUT / "lab12_drift_report.csv")
    print("Monitoring report:", OUT / "lab12_monitoring_report.json")
    print("Model files:", MODELS)

if __name__ == "__main__":
    main()
