import os
from typing import Literal

from google import genai
from pydantic import BaseModel

from models.llm_evidence import EvidenceRequest


class GeminiEvidence(BaseModel):
    customer_claim: Literal[
        "item_not_received",
        "unauthorized_transaction",
        "product_not_as_described",
        "other",
    ]
    claim_confidence: float
    contradicts_merchant_evidence: bool
    contradiction_confidence: float
    contradiction_detail: str
    new_signal_present: bool


class GeminiEvidenceProvider:
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")

        self.client = genai.Client(api_key=api_key)
        self.model = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

    def extract(self, request: EvidenceRequest):
        prompt = f"""
You are an evidence extraction assistant for a chargeback defense system.

Interpret the customer's dispute narrative against the merchant evidence.

Do NOT:
- predict whether the merchant will win
- make the final DEFEND, ACCEPT, or REVIEW decision
- invent evidence
- use hidden outcome information

CUSTOMER DISPUTE:
{request.dispute_text}

MERCHANT EVIDENCE:
{request.merchant_evidence}

Rules:

1. Identify the customer's main claim.
2. Give a confidence from 0.0 to 1.0.
3. A contradiction is valid ONLY when the customer's claim directly conflicts
   with a supplied merchant evidence field.
4. If there is a contradiction, contradiction_detail MUST explicitly contain
   the exact evidence field=value that creates the contradiction.
5. If there is no grounded contradiction, set contradiction confidence low
   and say that no grounded contradiction was identified.
6. Set new_signal_present=true only when the narrative contains meaningful
   information that is not represented in the merchant evidence.
"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": GeminiEvidence,
            },
        )

        return GeminiEvidence.model_validate_json(response.text).model_dump()