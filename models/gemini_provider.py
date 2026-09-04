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
3. Before setting contradicts_merchant_evidence, identify the SPECIFIC fact
   the customer is asserting, such as:
   - "the package was not received"
   - "the item was defective"
   - "the transaction was not authorized by me"

4. A contradiction exists ONLY if a supplied structured merchant evidence
   field asserts the DIRECT OPPOSITE of that same specific fact.

   Do NOT treat the following as contradictions:
   - related or thematically adjacent evidence
   - evidence that merely makes the customer's claim less likely
   - evidence that supports the merchant without directly denying the claim
   - missing or absent evidence
   - evidence about a different fact

   Examples:
   - Customer says "I did not authorize this transaction" +
     device_seen_before=True → NOT a contradiction.
   - Customer says "I did not authorize this transaction" +
     location_consistent=True → NOT a contradiction.
   - Customer says "the item was defective" +
     delivered=False → NOT a contradiction.
   - Customer says "I never received the package" +
     delivered=True → contradiction.
   - Customer says "I never received the package" +
     delivery_confirmed=True → contradiction.

5. If you cannot name a specific supplied merchant evidence field that
   directly asserts the opposite of the customer's specific claim, set:
   contradicts_merchant_evidence=false
   contradiction_confidence=0.0
   contradiction_detail="No grounded contradiction identified."

6. If there is a contradiction, contradiction_detail MUST explicitly contain
   the exact supplied evidence field=value that creates the contradiction.

7. When uncertain whether evidence is a direct contradiction or merely
   related/supporting evidence, default to:
   contradicts_merchant_evidence=false.

8. contradiction_confidence reflects how directly the evidence denies the
   specific customer claim, NOT how much merchant evidence exists overall.
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