"""Check whether the 200-iteration setting is robust across training splits.

This experiment uses only the official training CSV, never the official test CSV.
Each split selects its own best capacity for a paired cost comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
import numpy as np

from repairroute.decision import capacity_curve, outcome_at_capacity
from train import read_scania

ROOT = Path(__file__).resolve().parent.parent
SEEDS = (42, 17, 99, 123, 7)


def main():
    x, y = read_scania(ROOT / "data" / "aps_failure_training_set.csv")
    results = []
    for seed in SEEDS:
        x_fit, x_validation, y_fit, y_validation = train_test_split(
            x, y, test_size=0.20, stratify=y, random_state=seed
        )
        row = {"split_seed": seed}
        for iterations in (120, 200):
            model = HistGradientBoostingClassifier(
                max_iter=iterations, learning_rate=0.07, max_leaf_nodes=31,
                min_samples_leaf=30, l2_regularization=1.0,
                early_stopping=True, random_state=42,
            )
            model.fit(x_fit, y_fit)
            scores = model.predict_proba(x_validation)[:, 1]
            curve = capacity_curve(scores, y_validation)
            k = int(np.argmin(curve["cost"]))
            row[f"iterations_{iterations}"] = outcome_at_capacity(curve, k)
        row["cost_change_200_minus_120"] = (
            row["iterations_200"]["cost"] - row["iterations_120"]["cost"]
        )
        results.append(row)
        print(json.dumps(row), flush=True)
    summary = {
        "splits": results,
        "mean_cost_change": float(np.mean([row["cost_change_200_minus_120"] for row in results])),
        "wins_for_200": sum(row["cost_change_200_minus_120"] < 0 for row in results),
    }
    (ROOT / "experiments" / "stability_results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({"mean_cost_change": summary["mean_cost_change"], "wins_for_200": summary["wins_for_200"]}))


if __name__ == "__main__":
    main()
