Recourse

AI-Assisted Chargeback Defense & Decision Engine

Recourse helps merchants decide whether a chargeback is worth defending.

Given a disputed transaction, Recourse combines structured transaction evidence, customer history, fulfillment signals, merchant interactions, and the customer’s dispute narrative to produce an auditable DEFEND, ACCEPT, or REVIEW decision.

The core question is simple:

Should a merchant spend resources defending this chargeback?

⸻

How It Works

Recourse deliberately separates prediction, economics, and language-model reasoning.

                    CHARGEBACK
                        │
          ┌─────────────┴─────────────┐
          │                           │
          ▼                           ▼
  Structured evidence           Dispute narrative
          │                           │
          ▼                           ▼
 Logistic Regression               Gemini
          │                           │
          ▼                    Evidence extraction
       P(win)                         │
          │                    ┌──────┴──────┐
          │                    │             │
          │              contradiction   new signal
          │                    │
          └────────────┬───────┘
                       ▼
                Economic layer
                       │
                       ▼
              DEFEND / ACCEPT
                       │
          high-confidence grounded
             contradiction?
                       │
                       ▼
                    REVIEW

1. Structured risk model

A Logistic Regression model estimates:

P(win) = probability that the merchant successfully defends the chargeback.

The model uses only observable transaction and merchant evidence.

Hidden outcome fields such as the actual outcome, recovery amount, and defense cost are never exposed to the inference model.

2. Economic decision layer

A prediction alone is not enough.

Recourse estimates:

* Expected recovery
* Estimated defense cost
* Expected net value

The economic layer converts the predicted probability into an actionable merchant decision.

3. Gemini evidence layer

Customer dispute narratives contain information that is difficult to represent as structured features.

Gemini is therefore used as an evidence interpreter, not as the primary classifier.

It extracts:

* Customer claim
* Claim confidence
* Whether the narrative contradicts merchant evidence
* Contradiction confidence
* Grounded contradiction detail
* Whether the narrative contains a meaningful new signal

The LLM does not receive hidden outcomes and does not directly decide DEFEND or ACCEPT.

4. Grounded REVIEW safeguard

LLM output is never blindly trusted.

A contradiction is accepted only when the explanation can be grounded against an actual supplied merchant evidence field.

A high-confidence grounded contradiction routes the case to:

REVIEW — human verification required

If the LLM fails, produces invalid output, or cannot be grounded, Recourse safely falls back to human review rather than making an unsupported automated decision.

⸻

Decision Outcomes

DEFEND

The structured model and economic layer indicate that defending the chargeback has sufficient expected value, with no high-confidence grounded contradiction requiring review.

ACCEPT

The expected recovery does not justify the estimated defense cost.

REVIEW

The case requires human attention, for example because:

* The customer’s narrative directly contradicts merchant evidence with high confidence.
* The predicted win probability falls into the configured review band.
* LLM evidence extraction or validation fails.

This makes REVIEW a safety mechanism rather than treating every decision as a binary prediction.

⸻

Why Use an LLM Separately?

We tested whether narrative-derived signals should simply be added to the predictive classifier.

A deterministic heuristic proxy for the LLM-derived signals was evaluated on the same frozen train/validation/test methodology. The additional signals did not improve held-out classification performance.

Rather than forcing the LLM into the classifier, Recourse deliberately uses it where it adds a different capability:

The statistical model predicts risk. The LLM interprets the narrative. The economic layer makes the decision. The grounding layer controls when human review is required.

This separation also prevents an LLM’s uncalibrated confidence from being mistaken for a probability of winning.

⸻

Evaluation

Recourse was evaluated using a frozen synthetic dataset of 600 chargebacks.

The data is split into:

* 360 training cases
* 120 validation cases
* 120 held-out test cases

The split is fixed and stratified by chargeback outcome.

Thresholds and model choices are selected using training/validation data. The held-out test set remains untouched until final evaluation.

Held-out test results

The baseline comparison includes:

Strategy	Net Merchant Recovery
Always Accept	₹0.00
Rules baseline	₹107,843.32
Logistic Regression	₹152,451.19
Always Defend	₹151,812.20
Oracle	₹164,654.43

For the deployed Logistic Regression configuration on the held-out test set:

* Precision: 54.8%
* Recall: 98.4%
* FPR: 92.9%
* Defend rate: 95.8%
* Recovered: ₹178,876.88
* Defense cost: ₹26,425.69
* Net merchant recovery: ₹152,451.19
* Foregone recovery: ₹525.37

These numbers are reported on the held-out test set and should not be interpreted as production performance.

Important limitation

The current synthetic dataset produces a Logistic Regression policy that is highly defense-heavy. Its held-out net recovery is only modestly above the always-defend baseline.

Recourse therefore does not claim that the predictive model is already optimal.

Instead, the project demonstrates a complete decision framework with:

* Reproducible evaluation
* Leakage controls
* Economic decision-making
* Unstructured evidence interpretation
* Grounded contradiction detection
* Human-review escalation

⸻

Initial Dispute Types

Recourse currently supports:

* ITEM_NOT_RECEIVED
* UNAUTHORIZED_TRANSACTION
* PRODUCT_NOT_AS_DESCRIBED

The Gemini evidence layer can also return other when the narrative does not cleanly fit these categories.

⸻

Safety & Data Boundaries

The inference path intentionally separates observable evidence from hidden outcomes.

Observable evidence

Examples include:

* Transaction amount and age
* Payment method
* Account history
* Previous chargebacks/refunds
* Device familiarity
* Location consistency
* Transaction velocity
* Delivery confirmation
* Customer contact
* Merchant response time
* Refund request
* Dispute type
* Dispute narrative

Hidden outcome data

Used only for offline evaluation/training:

* Actual defensibility
* Actual win/loss outcome
* Actual recovery amount
* Defense cost
* Dataset split assignment

Hidden outcomes are not sent to Gemini and are not accepted through the public decision API.

⸻

API

Start the service:

python3 -m uvicorn app.main:app --reload

Health check:

GET /health

Decision endpoint:

POST /decide

Interactive API documentation:

GET /docs

The API returns:

* Final decision
* P(win)
* Expected recovery
* Estimated defense cost
* Expected net value
* Gemini evidence
* Review reason
* Auditable reasoning

⸻

Example

A customer claims:

“Tracking says delivered, but I never received the package.”

while merchant evidence contains:

delivered = true
delivery_confirmed = true

Gemini identifies the customer’s claim and detects the contradiction.

Recourse verifies that the contradiction is grounded in the supplied evidence and routes the case to:

REVIEW

The statistical P(win) and economic calculations remain visible rather than being overwritten by the LLM.

⸻

Project Structure

recourse/
├── app/
│   ├── main.py
│   ├── schemas.py
│   └── service.py
│
├── models/
│   ├── economic_decision.py
│   ├── logistic_regression.py
│   ├── llm_evidence.py
│   └── gemini_provider.py
│
├── evaluation/
│   ├── llm_ablation.py
│   ├── evaluate_model.py
│   ├── evaluate_economic_policy.py
│   ├── credibility.py
│   └── ...
│
├── data/
│   ├── chargebacks_observable.csv
│   ├── chargebacks_complete.csv
│   ├── split_assignments.csv
│   └── schema.md
│
└── README.md

⸻

Design Principles

AI should only be used where it adds value.

Recourse intentionally avoids making the LLM responsible for everything.

Logistic Regression handles structured risk prediction.

Economic logic handles expected-value decisions.

Gemini interprets unstructured dispute narratives.

Grounding and deterministic rules prevent unsupported LLM output from silently changing the decision.

Human review is used when automated evidence is ambiguous or contradictory.

The goal is not to make the system maximally AI-heavy.

The goal is to make it measurable, auditable, economically useful, and safe to automate.