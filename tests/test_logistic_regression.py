import unittest
from pathlib import Path

import pandas as pd

from data.split import load_split_case_ids
from models.logistic_regression import (
    FEATURE_COLUMNS,
    HIDDEN_OUTCOME_COLUMNS,
    predict_win_probabilities,
    select_features,
    train_model,
)


class LogisticRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.observable = pd.read_csv(root / "data" / "chargebacks_observable.csv")
        cls.complete = pd.read_csv(root / "data" / "chargebacks_complete.csv")
        cls.train_ids = load_split_case_ids("train")
        cls.validation_ids = load_split_case_ids("validation")
        cls.test_ids = load_split_case_ids("test")
        cls.train_cases = cls.observable.set_index("case_id").loc[cls.train_ids].reset_index()
        train_labels = cls.complete.set_index("case_id").loc[
            cls.train_ids, "actual_won"
        ]
        cls.model = train_model(cls.train_cases, train_labels)

    def test_hidden_outcomes_are_not_features(self):
        self.assertTrue(HIDDEN_OUTCOME_COLUMNS.isdisjoint(FEATURE_COLUMNS))
        with_hidden = self.train_cases.assign(actual_won=True)
        with self.assertRaisesRegex(ValueError, "Hidden outcome"):
            select_features(with_hidden)

    def test_splits_remain_separate_for_model_experiment(self):
        train_ids = set(self.train_ids)
        validation_ids = set(self.validation_ids)
        test_ids = set(self.test_ids)
        self.assertFalse(train_ids & validation_ids)
        self.assertFalse(train_ids & test_ids)
        self.assertFalse(validation_ids & test_ids)

    def test_probabilities_are_bounded(self):
        probabilities = predict_win_probabilities(self.model, self.train_cases)
        self.assertTrue(probabilities.between(0, 1).all())

    def test_inference_handles_unseen_rows(self):
        unseen_case = self.train_cases.iloc[[0]].copy()
        unseen_case.loc[:, "payment_method"] = "new_payment_method"
        probabilities = predict_win_probabilities(self.model, unseen_case)
        self.assertEqual(len(probabilities), 1)
        self.assertTrue(probabilities.between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
