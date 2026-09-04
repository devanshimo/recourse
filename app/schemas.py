from typing import Literal

from pydantic import BaseModel, Field


class ChargebackRequest(BaseModel):
    amount: float = Field(gt=0)
    transaction_age_days: int = Field(ge=0)
    payment_method: str
    account_age_days: int = Field(ge=0)
    previous_orders: int = Field(ge=0)
    previous_chargebacks: int = Field(ge=0)
    previous_refunds: int = Field(ge=0)
    device_seen_before: bool
    location_consistent: bool
    velocity_24h: int = Field(ge=0)
    delivered: bool
    delivery_confirmed: bool
    delivery_age_days: int = Field(ge=0)
    customer_contacted: bool
    merchant_response_time_hours: float = Field(ge=0)
    refund_requested: bool
    dispute_type: Literal[
        "ITEM_NOT_RECEIVED",
        "UNAUTHORIZED_TRANSACTION",
        "PRODUCT_NOT_AS_DESCRIBED",
    ]
    dispute_text: str = Field(min_length=1)


class LLMEvidenceResponse(BaseModel):
    customer_claim: str
    claim_confidence: float
    contradicts_merchant_evidence: bool
    contradiction_confidence: float
    contradiction_detail: str
    new_signal_present: bool


class DecisionResponse(BaseModel):
    decision: Literal["DEFEND", "ACCEPT", "REVIEW"]
    p_win: float
    amount: float
    expected_recovery: float
    estimated_defense_cost: float
    expected_net_value: float
    llm_evidence: LLMEvidenceResponse
    review_reason: str | None
    reasoning: list[str]