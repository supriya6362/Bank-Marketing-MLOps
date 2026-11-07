from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "bank-additional-full.csv"
MODELS = ROOT / "models"
OUT = ROOT / "outputs"
LOG = OUT / "lab13_prediction_log.csv"

def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(DATA, sep=";")
    y = df["y"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    X = df.drop(columns=["y"])

    v2 = MODELS / "bank_marketing_model_v2.joblib"
    v1 = MODELS / "bank_marketing_model_v1.joblib"
    model_path = v2 if v2.exists() else v1
    if not model_path.exists():
        raise FileNotFoundError("Run Lab 12 first to train the model.")

    model = joblib.load(model_path)
    # Process records in sequential batches to mimic continuous incoming predictions.
    batch_size = 1000
    rows = []
    for batch_no, start in enumerate(range(0, len(X), batch_size), start=1):
        xb = X.iloc[start:start + batch_size]
        yb = y.iloc[start:start + batch_size]
        predictions = model.predict(xb)
        probabilities = model.predict_proba(xb)[:, 1]
        rows.append({
            "batch": batch_no,
            "records": int(len(xb)),
            "predicted_subscriptions_yes": int((predictions == 1).sum()),
            "predicted_subscriptions_no": int((predictions == 0).sum()),
            "average_prediction_probability": float(np.mean(probabilities)),
            "accuracy": float(accuracy_score(yb, predictions)),
            "precision": float(precision_score(yb, predictions, zero_division=0)),
            "recall": float(recall_score(yb, predictions, zero_division=0)),
            "f1": float(f1_score(yb, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(yb, probabilities)) if yb.nunique() == 2 else None
        })
    result = pd.DataFrame(rows)
    result.to_csv(LOG, index=False)
    summary = {
        "lab": "13 - Continuous Prediction Monitoring",
        "model_used": model_path.name,
        "dataset": DATA.name,
        "total_records_processed": int(len(X)),
        "batch_size": batch_size,
        "batches_processed": int(len(result)),
        "overall_accuracy": float(accuracy_score(y, model.predict(X))),
        "overall_f1": float(f1_score(y, model.predict(X), zero_division=0)),
        "prediction_log": str(LOG.relative_to(ROOT)),
        "note": "This is batch-based local monitoring over the supplied labelled dataset, not a live production feed."
    }
    (OUT / "lab13_monitoring_summary.json").write_text(json.dumps(summary, indent=2))
    print("\\nLAB 13 COMPLETED")
    print("Model:", model_path.name)
    print("Records processed:", len(X))
    print("Batches:", len(result))
    print(result.head(10).to_string(index=False))
    print("Prediction log:", LOG)
    print("Summary:", OUT / "lab13_monitoring_summary.json")

if __name__ == "__main__":
    main()
