"""Provider-agnostic, grounded LLM evidence extraction for dispute text."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from typing import Any, Protocol

from models.economic_decision import EconomicDecision, REVIEW

ALLOWED_CLAIMS = {
    "item_not_received",
    "unauthorized_transaction",
    "product_not_as_described",
    "other",
}
MERCHANT_EVIDENCE_FIELDS = (
    "dispute_type",
    "delivered",
    "delivery_confirmed",
    "delivery_age_days",
    "device_seen_before",
    "location_consistent",
    "customer_contacted",
    "refund_requested",
    "merchant_response_time_hours",
)
HIDDEN_FIELDS = {
    "actual_won",
    "actual_defensible",
    "actual_recovery_amount",
    "defense_cost",
    "split",
}
REVIEW_CONTRADICTION_CONFIDENCE = 0.60


@dataclass(frozen=True)
class EvidenceRequest:
    """The complete, allowlisted input sent to an LLM provider."""

    dispute_text: str
    merchant_evidence: dict[str, object]

    def prompt(self) -> str:
        """Render a provider-ready instruction with a strict JSON contract."""
        evidence = ", ".join(
            f"{field}={_display_value(value)}"
            for field, value in self.merchant_evidence.items()
        )
        return (
            "Analyze the customer dispute text for a customer claim and any conflict "
            "with merchant evidence. Do not restate structured fields or invent facts. "
            "Return exactly this JSON schema: "
            '{"customer_claim":"item_not_received | unauthorized_transaction | '
            'product_not_as_described | other","claim_confidence":0.0,'
            '"contradicts_merchant_evidence":true,"contradiction_confidence":0.0,'
            '"contradiction_detail":"one short sentence","new_signal_present":true}. '
            "If a contradiction exists, contradiction_detail must contain an exact "
            "field=value token from the merchant evidence.\n"
            f"Customer dispute text: {self.dispute_text}\n"
            f"Merchant evidence: {evidence}"
        )


class EvidenceProvider(Protocol):
    """Minimal provider interface; adapters can call any LLM behind it."""

    def extract(self, request: EvidenceRequest) -> str | dict[str, Any]:
        """Return the JSON object requested by ``request.prompt()``."""


@dataclass(frozen=True)
class EvidenceExtraction:
    """The exact six-field LLM output schema."""

    customer_claim: str
    claim_confidence: float
    contradicts_merchant_evidence: bool
    contradiction_confidence: float
    contradiction_detail: str
    new_signal_present: bool


@dataclass(frozen=True)
class ExtractionResult:
    """Validated evidence plus a separate safe-fallback status."""

    evidence: EvidenceExtraction
    failed: bool
    failure_reason: str | None = None


class EvidenceValidationError(ValueError):
    """Raised internally when provider output violates the strict schema."""


def build_evidence_request(case: dict[str, object]) -> EvidenceRequest:
    """Allowlist only text and relevant observable evidence for the provider."""
    dispute_text = case.get("dispute_text")
    evidence = {
        field: case[field]
        for field in MERCHANT_EVIDENCE_FIELDS
        if field in case
    }
    return EvidenceRequest(
        dispute_text=dispute_text if isinstance(dispute_text, str) else "",
        merchant_evidence=evidence,
    )


def extract_evidence(
    case: dict[str, object],
    provider: EvidenceProvider,
) -> ExtractionResult:
    """Extract validated, grounded evidence; failures return a safe fallback."""
    request = build_evidence_request(case)
    if not request.dispute_text.strip():
        return _fallback("Missing or empty dispute text.")

    try:
        raw_output = provider.extract(request)
        evidence = _validate_schema(raw_output)
        evidence = _ground_contradiction(evidence, request.merchant_evidence)
        return ExtractionResult(evidence=evidence, failed=False)
    except (EvidenceValidationError, TypeError, ValueError, json.JSONDecodeError) as error:
        return _fallback(f"Extraction failed: {error}")
    except Exception as exc:
        print(f"\n!!! LLM Extraction failed: {type(exc).__name__}: {exc}\n")
        # Provider failures must be contained; do not expose implementation details.
        return _fallback("Provider extraction failed.")


def apply_llm_review_trigger(
    base_decision: EconomicDecision,
    extraction: ExtractionResult,
) -> EconomicDecision:
    """Overlay deterministic REVIEW safeguards without changing P(win)."""
    if extraction.failed:
        return replace(
            base_decision,
            decision=REVIEW,
            reason="LLM evidence extraction unavailable; route to human review.",
        )
    evidence = extraction.evidence
    if (
        evidence.contradicts_merchant_evidence
        and evidence.contradiction_confidence >= REVIEW_CONTRADICTION_CONFIDENCE
    ):
        return replace(
            base_decision,
            decision=REVIEW,
            reason="High-confidence grounded evidence contradiction requires review.",
        )
    return base_decision


def model_feature_signals(extraction: ExtractionResult) -> dict[str, float | bool]:
    """Return only the future-model-eligible LLM fields, safely gated."""
    evidence = extraction.evidence
    return {
        "contradiction_confidence": (
            evidence.contradiction_confidence
            if evidence.contradicts_merchant_evidence and not extraction.failed
            else 0.0
        ),
        "new_signal_present": evidence.new_signal_present and not extraction.failed,
    }


def _fallback(reason: str) -> ExtractionResult:
    return ExtractionResult(
        evidence=EvidenceExtraction(
            customer_claim="other",
            claim_confidence=0.0,
            contradicts_merchant_evidence=False,
            contradiction_confidence=0.0,
            contradiction_detail="Extraction unavailable; human review required.",
            new_signal_present=False,
        ),
        failed=True,
        failure_reason=reason,
    )


def _validate_schema(raw_output: str | dict[str, Any]) -> EvidenceExtraction:
    if isinstance(raw_output, str):
        raw_output = json.loads(raw_output)
    if not isinstance(raw_output, dict):
        raise EvidenceValidationError("Output must be a JSON object.")
    expected_fields = set(EvidenceExtraction.__annotations__)
    if set(raw_output) != expected_fields:
        raise EvidenceValidationError("Output fields do not match the required schema.")

    claim = raw_output["customer_claim"]
    detail = raw_output["contradiction_detail"]
    if not isinstance(claim, str) or claim not in ALLOWED_CLAIMS:
        raise EvidenceValidationError("customer_claim is invalid.")
    if not isinstance(detail, str) or not detail.strip() or len(detail) > 240:
        raise EvidenceValidationError("contradiction_detail must be a short non-empty string.")
    for field in ("claim_confidence", "contradiction_confidence"):
        value = raw_output[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EvidenceValidationError(f"{field} must be numeric.")
        if not math.isfinite(value) or not 0 <= float(value) <= 1:
            raise EvidenceValidationError(f"{field} must be between 0 and 1.")
    for field in ("contradicts_merchant_evidence", "new_signal_present"):
        if type(raw_output[field]) is not bool:
            raise EvidenceValidationError(f"{field} must be boolean.")
    return EvidenceExtraction(
        customer_claim=claim,
        claim_confidence=float(raw_output["claim_confidence"]),
        contradicts_merchant_evidence=raw_output["contradicts_merchant_evidence"],
        contradiction_confidence=float(raw_output["contradiction_confidence"]),
        contradiction_detail=detail.strip(),
        new_signal_present=raw_output["new_signal_present"],
    )


def _ground_contradiction(
    evidence: EvidenceExtraction,
    merchant_evidence: dict[str, object],
) -> EvidenceExtraction:
    if not evidence.contradicts_merchant_evidence:
        return evidence
    detail = evidence.contradiction_detail.lower()
    grounded_tokens = {
        f"{field}={_display_value(value)}".lower()
        for field, value in merchant_evidence.items()
    }
    if any(token in detail for token in grounded_tokens):
        return evidence
    return replace(
        evidence,
        contradicts_merchant_evidence=False,
        contradiction_confidence=0.0,
        contradiction_detail="No grounded contradiction identified.",
    )


def _display_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)
