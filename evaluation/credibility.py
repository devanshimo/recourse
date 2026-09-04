"""Held-out test comparison for reference, rules, model, and oracle policies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data.split import DEFAULT_MANIFEST_PATH, load_split_case_ids
from evaluation.evaluate import DEFAULT_COMPLETE_PATH, DEFAULT_OBSERVABLE_PATH
from evaluation.evaluate_model import _cases_for_ids, _decisions_from_probabilities
from evaluation.metrics import EvaluationMetrics, calculate_metrics, calculate_value_capture
from evaluation.policies import always_accept, always_defend, oracle_ceiling
from models.baseline import predict_decisions
from models.logistic_regression import predict_win_probabilities, train_model

# Fixed from validation-only selection.  This evaluation does not tune it on test.
LOGISTIC_VALIDATION_THRESHOLD = 0.25
ORACLE_POLICY_NAME = "Oracle ceiling (evaluation only)"


@dataclass(frozen=True)
class PolicyEvaluation:
    name: str
    decisions: pd.Series
    metrics: EvaluationMetrics
    value_capture: float | None


def run_credibility_evaluation(
    observable_path: Path = DEFAULT_OBSERVABLE_PATH,
    complete_path: Path = DEFAULT_COMPLETE_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> tuple[list[PolicyEvaluation], pd.DataFrame]:
    """Evaluate fixed policies on the held-out test set after training on train."""
    observable = pd.read_csv(observable_path)
    complete = pd.read_csv(complete_path)
    train_ids = load_split_case_ids("train", manifest_path)
    test_ids = load_split_case_ids("test", manifest_path)

    train_cases = _cases_for_ids(observable, train_ids)
    train_labels = complete.set_index("case_id").loc[train_ids, "actual_won"]
    model = train_model(train_cases, train_labels)

    test_cases = _cases_for_ids(observable, test_ids)
    model_probabilities = predict_win_probabilities(model, test_cases)
    decisions_by_policy = {
        "Always accept": always_accept(test_cases),
        "Always defend": always_defend(test_cases),
        "Rules baseline": predict_decisions(test_cases),
        "Logistic regression (threshold 0.25)": _decisions_from_probabilities(
            model_probabilities, LOGISTIC_VALIDATION_THRESHOLD
        ),
    }

    # Test outcomes are read only after fixed observable-only decisions exist.
    test_outcomes = complete.set_index("case_id").loc[
        test_ids,
        [
            "dispute_type",
            "actual_won",
            "actual_recovery_amount",
            "defense_cost",
        ],
    ].reset_index(drop=True)
    decisions_by_policy[ORACLE_POLICY_NAME] = oracle_ceiling(test_outcomes["actual_won"])
    outcomes = test_outcomes.drop(columns="dispute_type")
    raw_metrics = {
        name: calculate_metrics(decisions, outcomes)
        for name, decisions in decisions_by_policy.items()
    }
    oracle_net_value = raw_metrics[ORACLE_POLICY_NAME].net_economic_value
    evaluations = [
        PolicyEvaluation(
            name=name,
            decisions=decisions,
            metrics=metrics,
            value_capture=calculate_value_capture(
                metrics.net_economic_value, oracle_net_value
            ),
        )
        for name, decisions in decisions_by_policy.items()
        for metrics in [raw_metrics[name]]
    ]
    return evaluations, test_outcomes


def comparison_table(evaluations: list[PolicyEvaluation]) -> pd.DataFrame:
    """Build a same-test-set comparison table for all evaluated policies."""
    return pd.DataFrame(
        [
            {
                "policy": evaluation.name,
                "defend_rate": evaluation.decisions.eq("DEFEND").mean(),
                "precision": evaluation.metrics.precision,
                "recall": evaluation.metrics.recall,
                "false_positive_rate": evaluation.metrics.false_positive_rate,
                "recovered_amount": evaluation.metrics.amount_recovered,
                "defense_cost": evaluation.metrics.defense_cost,
                "net_value": evaluation.metrics.net_economic_value,
                "foregone_recovery": evaluation.metrics.foregone_recovery,
                "oracle_value_captured": evaluation.value_capture,
            }
            for evaluation in evaluations
        ]
    )


def dispute_type_breakdown(
    evaluations: list[PolicyEvaluation], test_outcomes: pd.DataFrame
) -> pd.DataFrame:
    """Calculate the requested metrics by policy and dispute type."""
    rows = []
    for evaluation in evaluations:
        for dispute_type, group in test_outcomes.groupby("dispute_type", sort=True):
            indices = group.index
            metrics = calculate_metrics(
                evaluation.decisions.loc[indices],
                group.drop(columns="dispute_type"),
            )
            rows.append(
                {
                    "policy": evaluation.name,
                    "dispute_type": dispute_type,
                    "cases": len(group),
                    "defend_rate": evaluation.decisions.loc[indices].eq("DEFEND").mean(),
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "recovered_amount": metrics.amount_recovered,
                    "defense_cost": metrics.defense_cost,
                    "net_value": metrics.net_economic_value,
                }
            )
    return pd.DataFrame(rows)


def _format_table(table: pd.DataFrame) -> str:
    formatted = table.copy()
    for column in (
        "defend_rate",
        "precision",
        "recall",
        "false_positive_rate",
        "oracle_value_captured",
    ):
        if column in formatted:
            formatted[column] = formatted[column].map(
                lambda value: "N/A" if pd.isna(value) else f"{value:.1%}"
            )
    for column in (
        "recovered_amount",
        "defense_cost",
        "net_value",
        "foregone_recovery",
    ):
        if column in formatted:
            formatted[column] = formatted[column].map(lambda value: f"INR {value:,.2f}")
    return formatted.to_string(index=False)


if __name__ == "__main__":
    evaluations, test_outcomes = run_credibility_evaluation()
    print("TEST comparison")
    print(_format_table(comparison_table(evaluations)))
    print("\nPer-dispute-type TEST results")
    print(_format_table(dispute_type_breakdown(evaluations, test_outcomes)))
