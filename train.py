"""Train once on Scania training data and export reproducible demo artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from repairroute.decision import capacity_curve, outcome_at_capacity, outcome_at_threshold

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
RANDOM_STATE = 42


def read_scania(path: Path):
    # The original UCI CSV begins with 20 descriptive lines before the header.
    frame = pd.read_csv(path, skiprows=20, na_values="na", low_memory=False)
    if "class" not in frame or len(frame.columns) != 171:
        raise ValueError(f"Unexpected Scania schema in {path}: {len(frame.columns)} columns")
    labels = frame.pop("class").eq("pos").to_numpy()
    features = frame.apply(pd.to_numeric, errors="coerce").astype("float32")
    return features, labels


def main():
    train_path = DATA / "aps_failure_training_set.csv"
    test_path = DATA / "aps_failure_test_set.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Download the original UCI CSV files into data/ first")

    x, y = read_scania(train_path)
    x_test, y_test = read_scania(test_path)
    if list(x.columns) != list(x_test.columns):
        raise ValueError("Training and test feature columns differ")

    x_fit, x_validation, y_fit, y_validation = train_test_split(
        x, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    model = HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.07,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        early_stopping=True,
        random_state=RANDOM_STATE,
    )
    model.fit(x_fit, y_fit)
    validation_scores = model.predict_proba(x_validation)[:, 1]
    validation_curve = capacity_curve(validation_scores, y_validation)
    best_k = int(np.argmin(validation_curve["cost"]))
    # The validation set alone determines the deployed default policy.
    if best_k == 0:
        threshold = float("inf")
    else:
        ranked_scores = validation_scores[validation_curve["order"]]
        threshold = float(ranked_scores[best_k - 1])

    test_scores = model.predict_proba(x_test)[:, 1]
    test_policy = outcome_at_threshold(test_scores, y_test, threshold)
    all_cost = int(10 * np.count_nonzero(~y_test))
    none_cost = int(500 * np.count_nonzero(y_test))

    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump({"model": model, "feature_columns": list(x.columns)}, ARTIFACTS / "model.joblib", compress=3)
    pd.DataFrame(
        {
            "record_id": [f"RR-{i+1:05d}" for i in range(len(y_test))],
            "priority_score": test_scores,
            "is_aps": y_test.astype(int),
        }
    ).to_csv(ARTIFACTS / "historical_scores.csv", index=False)
    x_test.head(5).to_csv(ARTIFACTS / "sample_batch.csv", index=False)

    metrics = {
        "training_rows": int(len(x_fit)),
        "validation_rows": int(len(x_validation)),
        "test_rows": int(len(x_test)),
        "test_aps_faults": int(y_test.sum()),
        "validation_selected_capacity": best_k,
        "validation_selected_threshold": threshold,
        "validation_policy": outcome_at_capacity(validation_curve, best_k),
        "fixed_test_policy": test_policy,
        "test_inspect_all_cost": all_cost,
        "test_inspect_none_cost": none_cost,
        "test_average_precision": float(average_precision_score(y_test, test_scores)),
        "test_roc_auc": float(roc_auc_score(y_test, test_scores)),
        "model": "HistGradientBoostingClassifier",
        "seed": RANDOM_STATE,
        "data_source": "UCI APS Failure at Scania Trucks, DOI 10.24432/C51S51",
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
