import unittest
from pathlib import Path

import pandas as pd

from data.split import load_split_case_ids
from evaluation.llm_ablation import (
    HeuristicMockEvidenceProvider,
    llm_feature_frame,
    run_llm_ablation,
)
from models.llm_evidence import EvidenceRequest


class UngroundedProvider:
    def extract(self, request: EvidenceRequest):
        return {
            "customer_claim": "item_not_received",
            "claim_confidence": 0.9,
            "contradicts_merchant_evidence": True,
            "contradiction_confidence": 0.9,
            "contradiction_detail": "tracking_number=missing conflicts.",
            "new_signal_present": True,
        }


class LLMAbationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.observable = pd.read_csv(root / "data" / "chargebacks_observable.csv")

    def test_llm_features_reject_hidden_columns(self):
        cases = self.observable.iloc[:1].assign(actual_won=True)
        with self.assertRaisesRegex(ValueError, "Hidden columns"):
            llm_feature_frame(cases, HeuristicMockEvidenceProvider())

    def test_ungrounded_contradiction_feature_is_zero(self):
        cases = self.observable.iloc[:1].copy()
        features, _ = llm_feature_frame(cases, UngroundedProvider())
        self.assertEqual(features.loc[cases.index[0], "contradiction_confidence"], 0.0)

    def test_new_signal_feature_is_boolean_and_preserved(self):
        cases = self.observable[self.observable["dispute_text"].str.contains("still waiting")].iloc[:1]
        features, _ = llm_feature_frame(cases, HeuristicMockEvidenceProvider())
        self.assertTrue(bool(features.iloc[0]["new_signal_present"]))

    def test_ablation_uses_non_overlapping_train_and_test_ids(self):
        self.assertFalse(set(load_split_case_ids("train")) & set(load_split_case_ids("test")))

    def test_ablation_is_deterministic(self):
        first = run_llm_ablation()
        second = run_llm_ablation()
        self.assertEqual(first.lr_only_threshold, second.lr_only_threshold)
        self.assertEqual(first.lr_llm_threshold, second.lr_llm_threshold)
        self.assertEqual(first.delta, second.delta)
        self.assertEqual(first.decision_flips, second.decision_flips)


if __name__ == "__main__":
    unittest.main()
