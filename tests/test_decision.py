import unittest

from repairroute.decision import capacity_curve, outcome_at_capacity, outcome_at_threshold, random_expected_cost


class DecisionTests(unittest.TestCase):
    def test_capacity_curve_tracks_both_error_costs(self):
        curve = capacity_curve([0.1, 0.9, 0.8], [True, False, True])
        self.assertEqual(outcome_at_capacity(curve, 0)["cost"], 1000)
        self.assertEqual(outcome_at_capacity(curve, 1)["cost"], 1010)
        self.assertEqual(outcome_at_capacity(curve, 2)["cost"], 510)
        self.assertEqual(outcome_at_capacity(curve, 3)["cost"], 10)
        self.assertEqual(outcome_at_capacity(curve, 2)["missed"], 1)

    def test_threshold_matches_capacity_when_scores_are_unique(self):
        scores = [0.1, 0.9, 0.8]
        labels = [True, False, True]
        curve = capacity_curve(scores, labels)
        self.assertEqual(outcome_at_threshold(scores, labels, 0.8), outcome_at_capacity(curve, 2))

    def test_random_baseline_at_extremes(self):
        self.assertEqual(random_expected_cost(100, 2, 0), 1000)
        self.assertEqual(random_expected_cost(100, 2, 100), 980)


if __name__ == "__main__":
    unittest.main()
