"""
Recourse — Synthetic Chargeback Dataset Generator
Generates a synthetic chargeback dataset with:
- observable transaction/customer/fulfillment evidence
- unstructured dispute text
- hidden ground-truth outcomes
- realistic noise and conflicting signals
IMPORTANT:
The hidden outcome is generated from latent variables rather than
directly from the observable features. This helps avoid label leakage.
"""
from __future__ import annotations
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
SEED = 42
N_RECORDS = 600
OUTPUT_DIR = Path(__file__).parent
random.seed(SEED)
np.random.seed(SEED)
DISPUTE_TYPES = [
    "ITEM_NOT_RECEIVED",
    "UNAUTHORIZED_TRANSACTION",
    "PRODUCT_NOT_AS_DESCRIBED",
]
PAYMENT_METHODS = [
    "card",
    "upi",
    "netbanking",
    "wallet",
]
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))
def generate_latent_world() -> dict:
    """
    Generate the underlying reality of a transaction.
    These variables are NOT given to the prediction model.
    They are used only to generate a noisy ground-truth outcome.
    """
    customer_reliability = np.random.beta(7, 2)
    fulfillment_quality = np.random.beta(8, 2)
    transaction_legitimacy = np.random.beta(7, 2)
    merchant_evidence_quality = np.random.beta(7, 2)
    return {
        "customer_reliability": customer_reliability,
        "fulfillment_quality": fulfillment_quality,
        "transaction_legitimacy": transaction_legitimacy,
        "merchant_evidence_quality": merchant_evidence_quality,
    }
def generate_observable_case(latent: dict) -> dict:
    """
    Generate the evidence visible to Recourse.
    Evidence is noisy and only partially reflects the latent reality.
    """
    dispute_type = random.choice(DISPUTE_TYPES)
    amount = round(
        np.random.lognormal(mean=np.log(2200), sigma=0.75),
        2,
    )
    amount = min(max(amount, 200), 25000)
    transaction_age_days = int(
        np.random.gamma(shape=2.5, scale=12)
    )
    transaction_age_days = min(max(transaction_age_days, 1), 120)
    payment_method = random.choice(PAYMENT_METHODS)
    customer_reliability = latent["customer_reliability"]
    transaction_legitimacy = latent["transaction_legitimacy"]
    fulfillment_quality = latent["fulfillment_quality"]
    # Customer history contains signal, but isn't a direct copy
    # of the latent variables.
    account_age_days = int(
        np.clip(
            np.random.normal(
                500 + customer_reliability * 500,
                180,
            ),
            5,
            2500,
        )
    )
    previous_orders = int(
        np.clip(
            np.random.poisson(
                3 + customer_reliability * 12
            ),
            0,
            50,
        )
    )
    previous_chargebacks = int(
        np.clip(
            np.random.poisson(
                max(0.05, (1 - customer_reliability) * 2)
            ),
            0,
            8,
        )
    )
    previous_refunds = int(
        np.clip(
            np.random.poisson(
                0.5 + (1 - customer_reliability) * 3
            ),
            0,
            12,
        )
    )
    device_seen_before = (
        np.random.random()
        < 0.65 + 0.25 * transaction_legitimacy
    )
    location_consistent = (
        np.random.random()
        < 0.65 + 0.25 * transaction_legitimacy
    )
    velocity_24h = int(
        np.clip(
            np.random.poisson(
                1.5 + (1 - transaction_legitimacy) * 5
            ),
            0,
            20,
        )
    )
    delivered = (
        np.random.random()
        < 0.55 + 0.4 * fulfillment_quality
    )
    delivery_confirmed = (
        delivered
        and (
            np.random.random()
            < 0.55 + 0.35 * fulfillment_quality
        )
    )
    delivery_age_days = (
        int(
            np.clip(
                np.random.normal(
                    8 + fulfillment_quality * 18,
                    6,
                ),
                1,
                60,
            )
        )
        if delivered
        else 0
    )
    customer_contacted = (
        np.random.random()
        < 0.25 + 0.5 * (1 - customer_reliability)
    )
    merchant_response_time_hours = round(
        np.clip(
            np.random.lognormal(
                mean=np.log(12),
                sigma=0.8,
            ),
            1,
            120,
        ),
        1,
    )
    refund_requested = (
        customer_contacted
        and np.random.random()
        < 0.35 + 0.25 * (1 - customer_reliability)
    )
    dispute_text = generate_dispute_text(
        dispute_type=dispute_type,
        delivered=delivered,
        delivery_confirmed=delivery_confirmed,
        customer_contacted=customer_contacted,
        refund_requested=refund_requested,
    )
    return {
        "amount": amount,
        "transaction_age_days": transaction_age_days,
        "payment_method": payment_method,
        "account_age_days": account_age_days,
        "previous_orders": previous_orders,
        "previous_chargebacks": previous_chargebacks,
        "previous_refunds": previous_refunds,
        "device_seen_before": device_seen_before,
        "location_consistent": location_consistent,
        "velocity_24h": velocity_24h,
        "delivered": delivered,
        "delivery_confirmed": delivery_confirmed,
        "delivery_age_days": delivery_age_days,
        "customer_contacted": customer_contacted,
        "merchant_response_time_hours": merchant_response_time_hours,
        "refund_requested": refund_requested,
        "dispute_type": dispute_type,
        "dispute_text": dispute_text,
    }
def generate_dispute_text(
    dispute_type: str,
    delivered: bool,
    delivery_confirmed: bool,
    customer_contacted: bool,
    refund_requested: bool,
) -> str:
    """
    Create deliberately varied synthetic dispute descriptions.
    The wording is intentionally not perfectly aligned with the
    structured fields. This prevents the text from becoming a
    trivial label shortcut.
    """
    templates = {
        "ITEM_NOT_RECEIVED": [
            "Customer says the order never arrived.",
            "Customer disputes the transaction claiming the package was not received.",
            "The buyer says they are still waiting for the delivery.",
            "Customer reports that the shipment was not received.",
        ],
        "UNAUTHORIZED_TRANSACTION": [
            "Customer says they did not authorize this payment.",
            "Buyer disputes the transaction as unauthorized.",
            "Customer claims the card payment was not made by them.",
            "The buyer says they do not recognize this transaction.",
        ],
        "PRODUCT_NOT_AS_DESCRIBED": [
            "Customer says the product received was different from the listing.",
            "Buyer claims the item did not match the description.",
            "Customer disputes the purchase because the product was not as expected.",
            "The buyer says the delivered product was materially different.",
        ],
    }
    text = random.choice(templates[dispute_type])
    details = []
    if customer_contacted:
        details.append(
            random.choice(
                [
                    "They mention contacting the merchant.",
                    "The customer says they previously contacted support.",
                    "They report that they tried to resolve this with the seller.",
                ]
            )
        )
    if refund_requested:
        details.append(
            random.choice(
                [
                    "They say they requested a refund.",
                    "The customer mentions asking for their money back.",
                ]
            )
        )
    if delivered and random.random() < 0.5:
        details.append(
            random.choice(
                [
                    "The order appears in their account history.",
                    "They refer to the original order.",
                    "The dispute was filed after the expected delivery period.",
                ]
            )
        )
    if delivery_confirmed and random.random() < 0.5:
        details.append(
            random.choice(
                [
                    "The merchant claims there is delivery confirmation.",
                    "The seller says delivery was confirmed.",
                ]
            )
        )
    random.shuffle(details)
    if details:
        text += " " + " ".join(details)
    return text
def generate_hidden_outcome(
    latent: dict,
    observed: dict,
) -> dict:
    """
    Generate hidden ground-truth outcomes.

    The hidden outcome is generated from latent reality rather than
    directly from the observable features.

    Defensibility and actual win are related but intentionally
    different concepts.
    """

    customer_reliability = latent["customer_reliability"]
    fulfillment_quality = latent["fulfillment_quality"]
    transaction_legitimacy = latent["transaction_legitimacy"]
    merchant_evidence_quality = latent["merchant_evidence_quality"]

    dispute_type = observed["dispute_type"]
    previous_orders = observed["previous_orders"]
    previous_chargebacks = observed["previous_chargebacks"]
    previous_refunds = observed["previous_refunds"]
    device_seen_before = observed["device_seen_before"]
    location_consistent = observed["location_consistent"]
    velocity_24h = observed["velocity_24h"]
    delivered = observed["delivered"]
    delivery_confirmed = observed["delivery_confirmed"]
    customer_contacted = observed["customer_contacted"]
    merchant_response_time_hours = observed[
        "merchant_response_time_hours"
    ]
    refund_requested = observed["refund_requested"]

    # =========================================================
    # 1. UNDERLYING DEFENSIBILITY
    # =========================================================

    # Multiple latent factors contribute to defensibility.
    # No single observable field determines the result.

    defensibility_score = (
        0.65 * customer_reliability
        + 0.85 * fulfillment_quality
        + 0.70 * merchant_evidence_quality
        + 0.60 * transaction_legitimacy
    )

    if dispute_type == "ITEM_NOT_RECEIVED":
        defensibility_score += (
            0.55 * fulfillment_quality
        )

        if observed["delivery_confirmed"]:
            defensibility_score += 0.35

    elif dispute_type == "UNAUTHORIZED_TRANSACTION":
        defensibility_score += (
            0.65 * transaction_legitimacy
        )

        if observed["device_seen_before"]:
            defensibility_score += 0.20

        if observed["location_consistent"]:
            defensibility_score += 0.20

        defensibility_score -= (
            0.08 * observed["velocity_24h"]
        )

    elif dispute_type == "PRODUCT_NOT_AS_DESCRIBED":
        defensibility_score += (
            0.40 * fulfillment_quality
        )

        if observed["refund_requested"]:
            defensibility_score -= 0.20

        if (
            observed["customer_contacted"]
            and observed["merchant_response_time_hours"] > 48
        ):
            defensibility_score -= 0.25

    # Unobserved factors create genuine uncertainty.
    defensibility_score += np.random.normal(0, 0.80)

    # The previous threshold was too permissive and produced an
    # unrealistically high defensibility rate.
    defensibility_probability = sigmoid(
        (defensibility_score - 2.45) * 1.35
    )

    actual_defensible = (
        np.random.random()
        < defensibility_probability
    )

    # =========================================================
    # 2. ACTUAL WIN PROBABILITY
    # =========================================================

    # Winning is related to defensibility, but not identical.
    # Evidence quality, fulfillment and operational behaviour
    # influence the eventual outcome.

    # ---------------------------------------------------------
# Actual win outcome
# ---------------------------------------------------------
# The outcome is generated from observable evidence with
# additional hidden noise. This gives the model real signal
# without making the label deterministic.
#
# IMPORTANT:
# actual_won is NOT exposed to the model.

    win_logit = -1.53

    # Customer history
    win_logit += 0.18 * min(previous_orders / 10.0, 2.0)
    win_logit -= 0.35 * min(previous_chargebacks, 3)
    win_logit -= 0.08 * min(previous_refunds, 5)

    # Transaction legitimacy signals
    win_logit += 0.45 if device_seen_before else -0.45
    win_logit += 0.30 if location_consistent else -0.30

    # Suspicious velocity
    if velocity_24h <= 2:
        win_logit += 0.20
    elif velocity_24h >= 6:
        win_logit -= 0.35

    # Fulfillment evidence
    if delivered:
        win_logit += 0.35
    else:
        win_logit -= 0.35

    if delivery_confirmed:
        win_logit += 0.55
    else:
        win_logit -= 0.25

    # Merchant evidence / responsiveness
    if customer_contacted:
        win_logit += 0.60

    if merchant_response_time_hours <= 12:
        win_logit += 0.30
    elif merchant_response_time_hours >= 48:
        win_logit -= 0.30

    # Refund behavior
    if refund_requested:
        win_logit -= 0.15

    # Dispute-specific effects
    if dispute_type == "ITEM_NOT_RECEIVED":
        if delivered and delivery_confirmed:
            win_logit += 0.45

    elif dispute_type == "UNAUTHORIZED_TRANSACTION":
        if device_seen_before and location_consistent:
            win_logit += 0.50

    elif dispute_type == "PRODUCT_NOT_AS_DESCRIBED":
        if customer_contacted and refund_requested:
            win_logit -= 0.35

    # Hidden noise represents factors unavailable to the model.
    win_logit += np.random.normal(0, 0.85)

    win_probability = 1 / (1 + np.exp(-win_logit))

    actual_won = np.random.random() < win_probability

    # =========================================================
    # 3. RECOVERY AMOUNT
    # =========================================================

    actual_recovery_amount = (
        observed["amount"]
        if actual_won
        else 0.0
    )

    # =========================================================
    # 4. DEFENSE COST
    # =========================================================

    defense_cost = (
        140
        + 0.025 * observed["amount"]
        + 35 * observed["previous_chargebacks"]
        + 20 * int(observed["customer_contacted"])
        + np.random.normal(0, 35)
    )

    defense_cost = max(
        round(defense_cost, 2),
        100.0,
    )

    return {
        "actual_defensible": bool(
            actual_defensible
        ),
        "actual_won": bool(
            actual_won
        ),
        "actual_recovery_amount": round(
            actual_recovery_amount,
            2,
        ),
        "defense_cost": round(
            defense_cost,
            2,
        ),
    }
def generate_dataset(n: int = N_RECORDS) -> pd.DataFrame:
    records = []
    for i in range(n):
        latent = generate_latent_world()
        observed = generate_observable_case(latent)
        hidden = generate_hidden_outcome(
            latent=latent,
            observed=observed,
        )
        record = {
            "case_id": f"CB-{i + 1:05d}",
            **observed,
            **hidden,
        }
        records.append(record)
    return pd.DataFrame(records)
def save_dataset(df: pd.DataFrame) -> None:
    """
    Save:
    - complete dataset for development/debugging
    - observable dataset for model training
    """
    complete_path = OUTPUT_DIR / "chargebacks_complete.csv"
    observable_columns = [
        "case_id",
        "amount",
        "transaction_age_days",
        "payment_method",
        "account_age_days",
        "previous_orders",
        "previous_chargebacks",
        "previous_refunds",
        "device_seen_before",
        "location_consistent",
        "velocity_24h",
        "delivered",
        "delivery_confirmed",
        "delivery_age_days",
        "customer_contacted",
        "merchant_response_time_hours",
        "refund_requested",
        "dispute_type",
        "dispute_text",
    ]
    observable_path = OUTPUT_DIR / "chargebacks_observable.csv"
    df.to_csv(complete_path, index=False)
    df[observable_columns].to_csv(
        observable_path,
        index=False,
    )
    print(f"Generated {len(df)} chargeback cases.")
    print(f"Complete dataset:   {complete_path}")
    print(f"Observable dataset: {observable_path}")
    print("\nDispute distribution:")
    print(df["dispute_type"].value_counts())
    print("\nHidden outcome distribution:")
    print(df["actual_defensible"].value_counts(normalize=True))
    print("\nActual win rate:")
    print(round(df["actual_won"].mean(), 3))
if __name__ == "__main__":
    dataset = generate_dataset()
    save_dataset(dataset)
