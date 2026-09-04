import os
from pathlib import Path
import pandas as pd

from models.gemini_provider import GeminiEvidenceProvider

from models.economic_decision import (
    EconomicPolicy,
    decide_from_expected_value,
    estimate_defense_cost,
)
from models.llm_evidence import (
    EvidenceProvider,
    apply_llm_review_trigger,
    build_evidence_request,
    extract_evidence,
)
from models.logistic_regression import (
    predict_win_probabilities,
    train_model,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OBSERVABLE_PATH = DATA_DIR / "chargebacks_observable.csv"
SPLIT_PATH = DATA_DIR / "split_assignments.csv"


POLICY = EconomicPolicy(
    minimum_expected_net_value=0.0,
    review_probability_lower=0.45,
    review_probability_upper=0.60,
)


def _load_model():
    cases = pd.read_csv(OBSERVABLE_PATH)
    splits = pd.read_csv(SPLIT_PATH)

    train_ids = set(splits.loc[splits["split"] == "train", "case_id"])
    train_cases = cases[cases["case_id"].isin(train_ids)].copy()

    # actual_won is deliberately obtained only from the complete dataset
    # for training. It is never part of the public API or LLM request.
    complete_path = DATA_DIR / "chargebacks_complete.csv"
    complete = pd.read_csv(complete_path)
    labels = (
        complete.set_index("case_id")
        .loc[train_cases["case_id"], "actual_won"]
        .astype(bool)
    )

    return train_model(train_cases, labels)


MODEL = _load_model()
GEMINI_PROVIDER = (
    GeminiEvidenceProvider()
    if os.environ.get("GEMINI_API_KEY")
    else None
)


def _case_to_dict(request) -> dict[str, object]:
    if hasattr(request, "model_dump"):
        return request.model_dump()
    return request.dict()


class FakeEvidenceProvider:
    """Deterministic provider for local demos/tests."""

    def extract(self, request):
        text = request.dispute_text.lower()

        if "tracking says delivered" in text and "never received" in text:
            return {
                "customer_claim": "item_not_received",
                "claim_confidence": 0.96,
                "contradicts_merchant_evidence": True,
                "contradiction_confidence": 0.87,
                "contradiction_detail": (
                    "delivery_confirmed=true conflicts with the customer's claim"
                ),
                "new_signal_present": True,
            }

        if "different from the listing" in text:
            return {
                "customer_claim": "product_not_as_described",
                "claim_confidence": 0.95,
                "contradicts_merchant_evidence": False,
                "contradiction_confidence": 0.05,
                "contradiction_detail": "No grounded contradiction identified.",
                "new_signal_present": False,
            }

        return {
            "customer_claim": "item_not_received",
            "claim_confidence": 0.95,
            "contradicts_merchant_evidence": False,
            "contradiction_confidence": 0.05,
            "contradiction_detail": "No grounded contradiction identified.",
            "new_signal_present": False,
        }


def decide_chargeback(
    request,
    evidence_provider: EvidenceProvider | None = None,
):
    case = _case_to_dict(request)

    # Never expose case_id/outcomes to the application decision path.
    model_frame = pd.DataFrame([case])
    p_win = float(predict_win_probabilities(MODEL, model_frame).iloc[0])

    # Existing economic cost estimator.
    estimated_cost = float(
        estimate_defense_cost(model_frame).iloc[0]
    )

    base_decision = decide_from_expected_value(
        win_probability=p_win,
        amount=float(case["amount"]),
        defense_cost=estimated_cost,
        policy=POLICY,
    )

    if evidence_provider is not None:
        provider = evidence_provider
    elif GEMINI_PROVIDER is not None:
        provider = GEMINI_PROVIDER
    else:
        provider = FakeEvidenceProvider()
    extraction = extract_evidence(case, provider)

    final_decision = apply_llm_review_trigger(
        base_decision,
        extraction,
    )

    evidence = extraction.evidence

    reasoning = [
        f"Structured evidence produced P(win) = {p_win:.3f}.",
        (
            f"Expected recovery = INR "
            f"{base_decision.expected_recovery:.2f} "
            f"(P(win) × amount)."
        ),
        f"Estimated defense cost = INR {estimated_cost:.2f}.",
        (
            f"Expected net value = INR "
            f"{base_decision.expected_net_value:.2f}."
        ),
        (
            f"LLM claim: {evidence.customer_claim} "
            f"(confidence {evidence.claim_confidence:.2f})."
        ),
    ]

    if evidence.contradicts_merchant_evidence:
        reasoning.append(
            "LLM detected a grounded contradiction: "
            f"{evidence.contradiction_detail}"
        )
    else:
        reasoning.append("LLM found no grounded contradiction.")

    if final_decision.decision == "REVIEW":
        reasoning.append(f"Review trigger: {final_decision.reason}")
    else:
        reasoning.append(
            f"Final decision from the existing economic policy: "
            f"{final_decision.decision}."
        )

    return {
        "decision": final_decision.decision,
        "p_win": p_win,
        "amount": float(case["amount"]),
        "expected_recovery": float(base_decision.expected_recovery),
        "estimated_defense_cost": estimated_cost,
        "expected_net_value": float(base_decision.expected_net_value),
        "llm_evidence": {
            "customer_claim": evidence.customer_claim,
            "claim_confidence": evidence.claim_confidence,
            "contradicts_merchant_evidence": (
                evidence.contradicts_merchant_evidence
            ),
            "contradiction_confidence": evidence.contradiction_confidence,
            "contradiction_detail": evidence.contradiction_detail,
            "new_signal_present": evidence.new_signal_present,
        },
        "review_reason": (
            final_decision.reason
            if final_decision.decision == "REVIEW"
            else None
        ),
        "reasoning": reasoning,
    }