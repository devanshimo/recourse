import copy
import unittest

from models.baseline import DEFEND
from models.economic_decision import EconomicDecision
from models.llm_evidence import (
    REVIEW_CONTRADICTION_CONFIDENCE,
    apply_llm_review_trigger,
    build_evidence_request,
    extract_evidence,
)


class FakeProvider:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def extract(self, request):
        self.requests.append(request)
        return self.response


class LLMEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.case = {
            "dispute_text": "I never received this order.",
            "dispute_type": "ITEM_NOT_RECEIVED",
            "delivered": True,
            "delivery_confirmed": True,
            "delivery_age_days": 8,
            "device_seen_before": True,
            "location_consistent": True,
            "customer_contacted": False,
            "refund_requested": False,
            "merchant_response_time_hours": 12.0,
            "actual_won": False,
            "actual_defensible": False,
            "actual_recovery_amount": 0.0,
            "defense_cost": 999.0,
            "split": "test",
        }
        self.valid_response = {
            "customer_claim": "item_not_received",
            "claim_confidence": 0.95,
            "contradicts_merchant_evidence": True,
            "contradiction_confidence": 0.8,
            "contradiction_detail": "delivery_confirmed=true conflicts with the claim.",
            "new_signal_present": False,
        }

    def _base_decision(self):
        return EconomicDecision(DEFEND, 0.8, 800.0, 100.0, 700.0, "Base decision.")

    def test_valid_extraction_and_grounding(self):
        result = extract_evidence(self.case, FakeProvider(self.valid_response))
        self.assertFalse(result.failed)
        self.assertTrue(result.evidence.contradicts_merchant_evidence)
        self.assertEqual(result.evidence.customer_claim, "item_not_received")

    def test_malformed_extraction_falls_back(self):
        result = extract_evidence(self.case, FakeProvider({"not": "the schema"}))
        self.assertTrue(result.failed)
        self.assertFalse(result.evidence.contradicts_merchant_evidence)
        self.assertEqual(apply_llm_review_trigger(self._base_decision(), result).decision, "REVIEW")

    def test_missing_text_falls_back_without_calling_provider(self):
        provider = FakeProvider(self.valid_response)
        result = extract_evidence({**self.case, "dispute_text": "  "}, provider)
        self.assertTrue(result.failed)
        self.assertEqual(provider.requests, [])

    def test_ungrounded_contradiction_is_not_accepted(self):
        response = {**self.valid_response, "contradiction_detail": "tracking_number=ABC conflicts."}
        result = extract_evidence(self.case, FakeProvider(response))
        self.assertFalse(result.evidence.contradicts_merchant_evidence)
        self.assertEqual(result.evidence.contradiction_confidence, 0.0)

    def test_high_confidence_grounded_contradiction_routes_to_review(self):
        result = extract_evidence(self.case, FakeProvider(self.valid_response))
        self.assertGreaterEqual(
            result.evidence.contradiction_confidence, REVIEW_CONTRADICTION_CONFIDENCE
        )
        self.assertEqual(apply_llm_review_trigger(self._base_decision(), result).decision, "REVIEW")

    def test_low_confidence_contradiction_does_not_override_decision(self):
        response = {**self.valid_response, "contradiction_confidence": 0.59}
        result = extract_evidence(self.case, FakeProvider(response))
        self.assertEqual(apply_llm_review_trigger(self._base_decision(), result).decision, DEFEND)

    def test_hidden_fields_are_never_sent_to_provider(self):
        provider = FakeProvider(self.valid_response)
        extract_evidence(self.case, provider)
        request = provider.requests[0]
        self.assertFalse(set(request.merchant_evidence) & {
            "actual_won", "actual_defensible", "actual_recovery_amount", "defense_cost", "split"
        })
        self.assertNotIn("actual_won", request.prompt())
        self.assertNotIn("split=test", request.prompt())

    def test_extraction_never_overwrites_structured_evidence(self):
        original = copy.deepcopy(self.case)
        extract_evidence(self.case, FakeProvider(self.valid_response))
        self.assertEqual(self.case, original)
        self.assertEqual(build_evidence_request(self.case).merchant_evidence["delivered"], True)


if __name__ == "__main__":
    unittest.main()
