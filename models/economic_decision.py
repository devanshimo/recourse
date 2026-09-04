"""Transparent expected-value decision policy for Recourse."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from models.baseline import ACCEPT, DEFEND

REVIEW = "REVIEW"


@dataclass(frozen=True)
class EconomicPolicy:
    """Configurable, explainable parameters for the economic policy."""

    minimum_expected_net_value: float
    review_probability_lower: float = 0.45
    review_probability_upper: float = 0.60


@dataclass(frozen=True)
class EconomicDecision:
    decision: str
    win_probability: float
    expected_recovery: float
    defense_cost: float
    expected_net_value: float
    reason: str


def decide_from_expected_value(
    win_probability: float,
    amount: float,
    defense_cost: float,
    policy: EconomicPolicy,
) -> EconomicDecision:
    """Turn P(win), amount, and an operational cost estimate into a decision."""
    if not 0 <= win_probability <= 1:
        raise ValueError("win_probability must be between 0 and 1.")
    if amount < 0 or defense_cost < 0:
        raise ValueError("amount and defense_cost must be non-negative.")
    if policy.review_probability_lower > policy.review_probability_upper:
        raise ValueError("review probability bounds must be ordered.")

    expected_recovery = win_probability * amount
    expected_net_value = expected_recovery - defense_cost
    in_review_band = (
        policy.review_probability_lower
        <= win_probability
        <= policy.review_probability_upper
    )

    if expected_net_value < policy.minimum_expected_net_value:
        decision = ACCEPT
        reason = "Expected net value is below the policy minimum."
    elif in_review_band:
        decision = REVIEW
        reason = "Expected value clears the minimum, but P(win) is in the review band."
    else:
        decision = DEFEND
        reason = "Expected net value clears the policy minimum outside the review band."

    return EconomicDecision(
        decision=decision,
        win_probability=win_probability,
        expected_recovery=expected_recovery,
        defense_cost=defense_cost,
        expected_net_value=expected_net_value,
        reason=reason,
    )


def estimate_defense_cost(cases: pd.DataFrame) -> pd.Series:
    """Create an observable-only operational defense-cost estimate.

    Actual defense cost is hidden evaluation data.  This estimate is based only
    on case fields known at decision time and is intentionally simple.
    """
    required_columns = {"amount", "previous_chargebacks", "customer_contacted"}
    missing_columns = required_columns - set(cases.columns)
    if missing_columns:
        raise ValueError(f"Missing cost-estimate columns: {', '.join(sorted(missing_columns))}")
    return (
        150.0
        + 0.025 * cases["amount"]
        + 30.0 * cases["previous_chargebacks"]
        + 20.0 * cases["customer_contacted"].astype(int)
    ).rename("estimated_defense_cost")


def decide_cases(
    cases: pd.DataFrame,
    win_probabilities: pd.Series,
    policy: EconomicPolicy,
) -> pd.DataFrame:
    """Apply the economic policy using observables and model probabilities only."""
    if not cases.index.equals(win_probabilities.index):
        raise ValueError("Cases and win probabilities must have matching indices.")
    estimated_costs = estimate_defense_cost(cases)
    records = [
        decide_from_expected_value(
            win_probability=float(probability),
            amount=float(amount),
            defense_cost=float(cost),
            policy=policy,
        )
        for probability, amount, cost in zip(
            win_probabilities,
            cases["amount"],
            estimated_costs,
            strict=True,
        )
    ]
    return pd.DataFrame(
        [record.__dict__ for record in records], index=cases.index
    )
