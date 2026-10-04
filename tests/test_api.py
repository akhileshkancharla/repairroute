import unittest
from pathlib import Path

import pandas as pd
from fastapi import HTTPException

from repairroute.api import ScoreRequest, health, historical_decision, historical_queue, metadata, score_batch


class ApiTests(unittest.TestCase):
    def test_health_and_metadata(self):
        self.assertEqual(health(), {"status": "ok"})
        meta = metadata()
        self.assertEqual(meta["historical_records"], 16000)
        self.assertEqual(meta["feature_count"], 170)
        self.assertEqual(meta["fixed_test_policy"]["cost"], 14970)

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

    def test_live_scoring_and_schema_validation(self):
        sample = pd.read_csv(Path(__file__).resolve().parent.parent / "artifacts" / "sample_batch.csv")
        record = {key: (None if pd.isna(value) else float(value)) for key, value in sample.iloc[0].items()}
        result = score_batch(ScoreRequest(records=[record]))
        self.assertEqual(result["count"], 1)
        self.assertGreaterEqual(result["records"][0]["priority_score"], 0)
        with self.assertRaises(HTTPException):
            score_batch(ScoreRequest(records=[{}]))


if __name__ == "__main__":
    unittest.main()
