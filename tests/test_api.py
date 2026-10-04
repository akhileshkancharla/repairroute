import unittest
from pathlib import Path

import pandas as pd
from fastapi import HTTPException

from repairroute.api import ScoreRequest, health, historical_curve, historical_decision, historical_queue, metadata, sample_batch, score_batch, website


class ApiTests(unittest.TestCase):
    def test_health_and_metadata(self):
        self.assertEqual(health(), {"status": "ok"})
        meta = metadata()
        self.assertEqual(meta["historical_records"], 16000)
        self.assertEqual(meta["feature_count"], 170)
        self.assertAlmostEqual(meta["reference_threshold"], 0.022607538574107376)
        self.assertEqual(meta["fixed_test_policy"]["cost"], 14970)
        self.assertAlmostEqual(meta["test_f1"], 0.720164609053498)
        self.assertEqual(meta["test_confusion_matrix"]["false_negative"], 25)

    def test_decision_and_queue_use_same_capacity(self):
        decision = historical_decision(capacity=10)
        queue = historical_queue(capacity=10, limit=20, offset=0)
        self.assertEqual(decision["outcome"]["inspections"], 10)
        self.assertEqual(queue["total_selected"], 10)
        self.assertEqual(len(queue["records"]), 10)
        self.assertNotIn("is_aps", queue["records"][0])
        self.assertEqual(
            decision["outcome"]["cost"],
            10 * decision["outcome"]["unnecessary"] + 500 * decision["outcome"]["missed"],
        )

    def test_capacity_limits(self):
        with self.assertRaises(HTTPException):
            historical_decision(capacity=16001)
        with self.assertRaises(HTTPException):
            historical_queue(capacity=16001, limit=20, offset=0)

    def test_frontend_and_curve_artifacts(self):
        self.assertEqual(Path(website().path).name, "index.html")
        self.assertEqual(Path(sample_batch().path).name, "sample_batch.csv")
        self.assertEqual(Path(sample_batch("high").path).name, "sample_high_priority.csv")
        self.assertEqual(Path(sample_batch("borderline").path).name, "sample_borderline.csv")
        self.assertEqual(Path(sample_batch("low").path).name, "sample_low_priority.csv")
        curve = historical_curve(points=11, max_capacity=1600)
        self.assertEqual(curve["max_capacity"], 1600)
        self.assertEqual(curve["points"][0]["capacity"], 0)
        self.assertEqual(curve["points"][-1]["capacity"], 1600)

    def test_live_scoring_and_schema_validation(self):
        sample = pd.read_csv(Path(__file__).resolve().parent.parent / "artifacts" / "sample_batch.csv")
        record = {key: (None if pd.isna(value) else float(value)) for key, value in sample.iloc[0].items()}
        result = score_batch(ScoreRequest(records=[record]))
        self.assertEqual(result["count"], 1)
        self.assertGreaterEqual(result["records"][0]["priority_score"], 0)
        with self.assertRaises(HTTPException):
            score_batch(ScoreRequest(records=[{}]))

    def test_mixed_demo_batch_contains_distinct_priorities(self):
        sample = pd.read_csv(Path(__file__).resolve().parent.parent / "artifacts" / "sample_batch.csv")
        records = [
            {key: (None if pd.isna(value) else float(value)) for key, value in row.items()}
            for _, row in sample.iterrows()
        ]
        scores = [item["priority_score"] for item in score_batch(ScoreRequest(records=records))["records"]]
        self.assertEqual(len(scores), 30)
        self.assertGreater(max(scores), 0.9)
        self.assertLess(min(scores), 0.001)
        self.assertGreater(sum(score >= metadata()["reference_threshold"] for score in scores), 5)


if __name__ == "__main__":
    unittest.main()
