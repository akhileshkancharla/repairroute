"""Cost-aware ranking for historical APS inspection decisions."""

from __future__ import annotations

import numpy as np

FALSE_INSPECTION_COST = 10
MISSED_APS_COST = 500


def capacity_curve(scores, labels):
    """Return outcomes for inspecting the top k records, for every k from 0 to n."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    if scores.ndim != 1 or labels.ndim != 1 or len(scores) != len(labels):
        raise ValueError("Scores and labels must be equally sized one-dimensional arrays")
    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite")

    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    n = len(labels)
    capacity = np.arange(n + 1, dtype=int)
    caught = np.concatenate(([0], np.cumsum(sorted_labels, dtype=int)))
    unnecessary = capacity - caught
    missed = int(labels.sum()) - caught
    cost = FALSE_INSPECTION_COST * unnecessary + MISSED_APS_COST * missed
    return {
        "order": order,
        "capacity": capacity,
        "caught": caught,
        "unnecessary": unnecessary,
        "missed": missed,
        "cost": cost,
    }


def outcome_at_capacity(curve, capacity):
    k = int(capacity)
    if k < 0 or k >= len(curve["capacity"]):
        raise ValueError("Capacity is outside the batch size")
    caught = int(curve["caught"][k])
    return {
        "inspections": k,
        "caught": caught,
        "unnecessary": int(curve["unnecessary"][k]),
        "missed": int(curve["missed"][k]),
        "cost": int(curve["cost"][k]),
        "recall": caught / (caught + int(curve["missed"][k])) if (caught + int(curve["missed"][k])) else 0.0,
        "precision": caught / k if k else 0.0,
    }


def outcome_at_threshold(scores, labels, threshold):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    inspect = scores >= threshold
    caught = int(np.count_nonzero(inspect & labels))
    unnecessary = int(np.count_nonzero(inspect & ~labels))
    missed = int(np.count_nonzero(~inspect & labels))
    inspections = caught + unnecessary
    return {
        "inspections": inspections,
        "caught": caught,
        "unnecessary": unnecessary,
        "missed": missed,
        "cost": FALSE_INSPECTION_COST * unnecessary + MISSED_APS_COST * missed,
        "recall": caught / (caught + missed) if caught + missed else 0.0,
        "precision": caught / inspections if inspections else 0.0,
    }


def random_expected_cost(total, positives, capacity):
    """Expected benchmark cost of a uniformly random queue at the same capacity."""
    if total == 0:
        return 0.0
    fraction = capacity / total
    return (
        FALSE_INSPECTION_COST * (total - positives) * fraction
        + MISSED_APS_COST * positives * (1 - fraction)
    )
