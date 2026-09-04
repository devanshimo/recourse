"""Auditable evaluation metrics for chargeback-defense decisions."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from models.baseline import ACCEPT, DEFEND


@dataclass(frozen=True)
class EvaluationMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    false_positive_rate: float
    false_negative_rate: float
    amount_recovered: float
    defense_cost: float
    net_economic_value: float
    foregone_recovery: float


def calculate_value_capture(
    policy_net_value: float,
    oracle_net_value: float,
) -> float | None:
    """Return policy value as a share of the positive oracle ceiling.

    An oracle ceiling of zero or less does not define a meaningful percentage,
    so ``None`` is returned instead of dividing by zero or a negative value.
    """
    if oracle_net_value <= 0:
        return None
    return policy_net_value / oracle_net_value


def calculate_metrics(
    decisions: pd.Series,
    outcomes: pd.DataFrame,
) -> EvaluationMetrics:
    """Score DEFEND decisions against hidden outcomes.

    A true positive is a defended case that actually won.  Economic values are
    calculated only for defended cases:

        amount_recovered = sum(actual_recovery_amount where decision == DEFEND)
        defense_cost = sum(defense_cost where decision == DEFEND)
        net_economic_value = amount_recovered - defense_cost
    """
    required_outcome_columns = {
        "actual_won",
        "actual_recovery_amount",
        "defense_cost",
    }
    missing_columns = required_outcome_columns - set(outcomes.columns)
    if missing_columns:
        raise ValueError(
            "Missing required hidden outcome columns: "
            f"{', '.join(sorted(missing_columns))}"
        )
    if len(decisions) != len(outcomes):
        raise ValueError("Decisions and outcomes must contain the same number of rows.")
    if not decisions.isin([DEFEND, ACCEPT, "REVIEW"]).all():
        raise ValueError("Decisions must contain only DEFEND, ACCEPT, or REVIEW.")

    defended = decisions.reset_index(drop=True).eq(DEFEND)
    actual_won = outcomes["actual_won"].reset_index(drop=True).astype(bool)

    true_positives = int((defended & actual_won).sum())
    false_positives = int((defended & ~actual_won).sum())
    false_negatives = int((~defended & actual_won).sum())
    true_negatives = int((~defended & ~actual_won).sum())

    positive_predictions = true_positives + false_positives
    actual_positives = true_positives + false_negatives
    actual_negatives = false_positives + true_negatives

    amount_recovered = float(
        outcomes.loc[defended.to_numpy(), "actual_recovery_amount"].sum()
    )
    defense_cost = float(outcomes.loc[defended.to_numpy(), "defense_cost"].sum())
    foregone_recovery = float(
        outcomes.loc[(~defended & actual_won).to_numpy(), "actual_recovery_amount"].sum()
    )

    return EvaluationMetrics(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        true_negatives=true_negatives,
        precision=true_positives / positive_predictions if positive_predictions else 0.0,
        recall=true_positives / actual_positives if actual_positives else 0.0,
        false_positive_rate=false_positives / actual_negatives if actual_negatives else 0.0,
        false_negative_rate=false_negatives / actual_positives if actual_positives else 0.0,
        amount_recovered=amount_recovered,
        defense_cost=defense_cost,
        net_economic_value=amount_recovered - defense_cost,
        foregone_recovery=foregone_recovery,
    )
