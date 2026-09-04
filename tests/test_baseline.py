import unittest

import pandas as pd

from evaluation.metrics import calculate_metrics
from models.baseline import ACCEPT, DEFEND, decide, predict_decisions


class BaselineRuleTests(unittest.TestCase):
    def test_rules_cover_each_supported_dispute_type(self):
        self.assertEqual(
            decide({"dispute_type": "ITEM_NOT_RECEIVED", "delivery_confirmed": True}),
            DEFEND,
        )
        self.assertEqual(
            decide(
                {
                    "dispute_type": "UNAUTHORIZED_TRANSACTION",
                    "device_seen_before": True,
                    "location_consistent": True,
                    "velocity_24h": 5,
                }
            ),
            DEFEND,
        )
        self.assertEqual(
            decide(
                {
                    "dispute_type": "PRODUCT_NOT_AS_DESCRIBED",
                    "delivered": True,
                    "customer_contacted": True,
                    "merchant_response_time_hours": 24,
                }
            ),
            DEFEND,
        )

    def test_hidden_columns_do_not_change_decisions(self):
        observable = pd.DataFrame(
            [
                {
                    "case_id": "CB-1",
                    "dispute_type": "ITEM_NOT_RECEIVED",
                    "delivered": True,
                    "delivery_confirmed": True,
                    "device_seen_before": False,
                    "location_consistent": False,
                    "velocity_24h": 0,
                    "customer_contacted": False,
                    "merchant_response_time_hours": 99,
                }
            ]
        )
        with_hidden = observable.assign(
            actual_defensible=False,
            actual_won=False,
            actual_recovery_amount=0.0,
            defense_cost=99999.0,
        )
        self.assertEqual(
            predict_decisions(observable).tolist(),
            predict_decisions(with_hidden).tolist(),
        )


class MetricsTests(unittest.TestCase):
    def test_economic_metrics_only_charge_defended_cases(self):
        decisions = pd.Series([DEFEND, DEFEND, ACCEPT, ACCEPT])
        outcomes = pd.DataFrame(
            {
                "actual_won": [True, False, True, False],
                "actual_recovery_amount": [1000.0, 0.0, 2500.0, 0.0],
                "defense_cost": [100.0, 200.0, 300.0, 400.0],
            }
        )
        metrics = calculate_metrics(decisions, outcomes)
        self.assertEqual((metrics.true_positives, metrics.false_positives), (1, 1))
        self.assertEqual((metrics.false_negatives, metrics.true_negatives), (1, 1))
        self.assertEqual(metrics.amount_recovered, 1000.0)
        self.assertEqual(metrics.defense_cost, 300.0)
        self.assertEqual(metrics.net_economic_value, 700.0)
        self.assertEqual(metrics.foregone_recovery, 2500.0)

    def test_losing_defense_still_incurred_its_defense_cost(self):
        decisions = pd.Series([DEFEND])
        outcomes = pd.DataFrame(
            {
                "actual_won": [False],
                "actual_recovery_amount": [0.0],
                "defense_cost": [275.0],
            }
        )
        metrics = calculate_metrics(decisions, outcomes)
        self.assertEqual(metrics.false_positives, 1)
        self.assertEqual(metrics.defense_cost, 275.0)
        self.assertEqual(metrics.net_economic_value, -275.0)


if __name__ == "__main__":
    unittest.main()
