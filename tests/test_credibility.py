import unittest

import pandas as pd

from evaluation.metrics import calculate_metrics, calculate_value_capture
from evaluation.policies import always_accept, always_defend, oracle_ceiling
from models.baseline import ACCEPT, DEFEND


class EvaluationPolicyTests(unittest.TestCase):
    def test_always_defend_and_accept(self):
        cases = pd.DataFrame({"case_id": ["CB-1", "CB-2"]})
        self.assertEqual(always_defend(cases).tolist(), [DEFEND, DEFEND])
        self.assertEqual(always_accept(cases).tolist(), [ACCEPT, ACCEPT])

    def test_oracle_isolated_to_hidden_actual_won_label(self):
        actual_won = pd.Series([True, False, True])
        self.assertEqual(
            oracle_ceiling(actual_won).tolist(), [DEFEND, ACCEPT, DEFEND]
        )

    def test_foregone_recovery_and_oracle_value_capture(self):
        outcomes = pd.DataFrame(
            {
                "actual_won": [True, False],
                "actual_recovery_amount": [1_000.0, 0.0],
                "defense_cost": [100.0, 200.0],
            }
        )
        accepted_metrics = calculate_metrics(always_accept(outcomes), outcomes)
        oracle_metrics = calculate_metrics(oracle_ceiling(outcomes["actual_won"]), outcomes)
        self.assertEqual(accepted_metrics.foregone_recovery, 1_000.0)
        self.assertEqual(oracle_metrics.net_economic_value, 900.0)
        self.assertEqual(
            calculate_value_capture(oracle_metrics.net_economic_value, oracle_metrics.net_economic_value),
            1.0,
        )
        self.assertIsNone(calculate_value_capture(10.0, 0.0))


if __name__ == "__main__":
    unittest.main()
