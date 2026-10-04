"""Compare candidate APS ranking models using training/validation data only.

Run with the original UCI training CSV already downloaded to data/. This script
never reads the official test CSV or the held-out score artifact.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import joblib
from lightgbm import LGBMClassifier
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from repairroute.decision import capacity_curve, outcome_at_capacity
from train import RANDOM_STATE, read_scania

ROOT = Path(__file__).resolve().parent.parent


def evaluate(name, model, x_fit, y_fit, x_validation, y_validation):
    started = perf_counter()
    if model is not None:
        model.fit(x_fit, y_fit)
    else:
        model = joblib.load(ROOT / "artifacts" / "model.joblib")["model"]
    scores = model.predict_proba(x_validation)[:, 1]
    curve = capacity_curve(scores, y_validation)
    k = int(np.argmin(curve["cost"]))
    outcome = outcome_at_capacity(curve, k)
    result = {
        "model": name,
        "validation_cost": outcome["cost"],
        "inspections": outcome["inspections"],
        "caught": outcome["caught"],
        "missed": outcome["missed"],
        "unnecessary": outcome["unnecessary"],
        "average_precision": float(average_precision_score(y_validation, scores)),
        "fit_and_score_seconds": round(perf_counter() - started, 2),
    }
    return result


def main():
    x, y = read_scania(ROOT / "data" / "aps_failure_training_set.csv")
    x_fit, x_validation, y_fit, y_validation = train_test_split(
        x, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    candidates = [
        ("current HistGradientBoosting", None),
        ("LightGBM 200 trees / 31 leaves", LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            min_child_samples=30, reg_lambda=1.0, verbosity=-1,
            n_jobs=2, random_state=RANDOM_STATE,
        )),
        ("LightGBM 350 trees / 15 leaves", LGBMClassifier(
            n_estimators=350, learning_rate=0.04, num_leaves=15,
            min_child_samples=40, reg_lambda=3.0, verbosity=-1,
            n_jobs=2, random_state=RANDOM_STATE,
        )),
        ("LightGBM 150 trees / 63 leaves", LGBMClassifier(
            n_estimators=150, learning_rate=0.05, num_leaves=63,
            min_child_samples=25, reg_lambda=2.0, verbosity=-1,
            n_jobs=2, random_state=RANDOM_STATE,
        )),
        ("LightGBM 200 trees / positive weight 5", LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            min_child_samples=30, reg_lambda=1.0, scale_pos_weight=5,
            verbosity=-1, n_jobs=2, random_state=RANDOM_STATE,
        )),
        ("LightGBM 200 trees / positive weight 20", LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            min_child_samples=30, reg_lambda=1.0, scale_pos_weight=20,
            verbosity=-1, n_jobs=2, random_state=RANDOM_STATE,
        )),
        ("HistGradientBoosting 200 iterations", HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.07, max_leaf_nodes=31,
            min_samples_leaf=30, l2_regularization=1.0,
            early_stopping=True, random_state=RANDOM_STATE,
        )),
    ]
    results = [
        evaluate(name, model, x_fit, y_fit, x_validation, y_validation)
        for name, model in candidates
    ]
    output = ROOT / "experiments" / "validation_comparison.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
