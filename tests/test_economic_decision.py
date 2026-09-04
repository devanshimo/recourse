import inspect
import unittest

from models.baseline import ACCEPT, DEFEND
from models.economic_decision import (
    REVIEW,
    EconomicPolicy,
    decide_from_expected_value,
)


class EconomicDecisionTests(unittest.TestCase):
    def setUp(self):
        self.policy = EconomicPolicy(minimum_expected_net_value=250.0)

    def test_expected_value_arithmetic(self):
        record = decide_from_expected_value(0.8, 1_000.0, 200.0, self.policy)
        self.assertEqual(record.expected_recovery, 800.0)
        self.assertEqual(record.expected_net_value, 600.0)

    def test_obviously_profitable_case_is_defended(self):
        record = decide_from_expected_value(0.9, 5_000.0, 200.0, self.policy)
        self.assertEqual(record.decision, DEFEND)

    def test_obviously_unprofitable_case_is_accepted(self):
        record = decide_from_expected_value(0.1, 1_000.0, 500.0, self.policy)
        self.assertEqual(record.decision, ACCEPT)

    def test_uncertain_profitable_case_is_reviewed(self):
        record = decide_from_expected_value(0.5, 2_000.0, 200.0, self.policy)
        self.assertEqual(record.decision, REVIEW)

    def test_decision_function_accepts_no_hidden_outcome_fields(self):
        parameters = set(inspect.signature(decide_from_expected_value).parameters)
        self.assertFalse(
            parameters & {"actual_won", "actual_defensible", "actual_recovery_amount"}
        )


if __name__ == "__main__":
    unittest.main()
