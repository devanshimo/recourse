"""A deliberately simple, rules-only chargeback decision baseline."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

DEFEND = "DEFEND"
ACCEPT = "ACCEPT"

# These are the only fields read by the baseline.  Outcome fields are
# intentionally absent: this strategy must be usable before an outcome exists.
REQUIRED_OBSERVABLE_COLUMNS = {
    "case_id",
    "dispute_type",
    "delivered",
    "delivery_confirmed",
    "device_seen_before",
    "location_consistent",
    "velocity_24h",
    "customer_contacted",
    "merchant_response_time_hours",
}


def decide(case: Mapping[str, object]) -> str:
    """Return a transparent DEFEND or ACCEPT decision from observable evidence.

    Rules:
    - ITEM_NOT_RECEIVED: defend only with delivery confirmation.
    - UNAUTHORIZED_TRANSACTION: defend only when the device and location match
      prior customer behaviour and transaction velocity is not high.
    - PRODUCT_NOT_AS_DESCRIBED: defend only when the order was delivered, the
      customer contacted the merchant, and the merchant replied within 24 hours.

    The conservative rules intentionally trade recall for an easy-to-audit
    baseline based on strong, dispute-specific evidence.
    """
    dispute_type = case["dispute_type"]

    if dispute_type == "ITEM_NOT_RECEIVED":
        return DEFEND if bool(case["delivery_confirmed"]) else ACCEPT

    if dispute_type == "UNAUTHORIZED_TRANSACTION":
        has_recognised_transaction_pattern = (
            bool(case["device_seen_before"])
            and bool(case["location_consistent"])
            and float(case["velocity_24h"]) <= 5
        )
        return DEFEND if has_recognised_transaction_pattern else ACCEPT

    if dispute_type == "PRODUCT_NOT_AS_DESCRIBED":
        has_resolved_fulfillment_evidence = (
            bool(case["delivered"])
            and bool(case["customer_contacted"])
            and float(case["merchant_response_time_hours"]) <= 24
        )
        return DEFEND if has_resolved_fulfillment_evidence else ACCEPT

    raise ValueError(f"Unsupported dispute type: {dispute_type!r}")


def predict_decisions(observable_cases: pd.DataFrame) -> pd.Series:
    """Apply the baseline to observable cases without reading hidden outcomes."""
    missing_columns = REQUIRED_OBSERVABLE_COLUMNS - set(observable_cases.columns)
    if missing_columns:
        raise ValueError(
            "Missing required observable columns: "
            f"{', '.join(sorted(missing_columns))}"
        )

    return observable_cases.apply(decide, axis=1).rename("decision")
