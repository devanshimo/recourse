"""Validation selection and one-time test evaluation of the economic policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data.split import DEFAULT_MANIFEST_PATH, load_split_case_ids
from evaluation.credibility import ORACLE_POLICY_NAME
from evaluation.evaluate import DEFAULT_COMPLETE_PATH, DEFAULT_OBSERVABLE_PATH
from evaluation.evaluate_model import _cases_for_ids, _decisions_from_probabilities
from evaluation.metrics import EvaluationMetrics, calculate_metrics, calculate_value_capture
from evaluation.policies import always_accept, always_defend, oracle_ceiling
from models.baseline import predict_decisions
from models.economic_decision import EconomicPolicy, decide_cases
from models.logistic_regression import predict_win_probabilities, train_model

MARGIN_CANDIDATES = (0.0, 250.0, 500.0, 750.0, 1_000.0, 1_250.0)
REVIEW_PROBABILITY_BOUNDS = (0.45, 0.60)
HISTORICAL_LOGISTIC_THRESHOLD = 0.25


@dataclass(frozen=True)
class EconomicExperimentResult:
    validation_sweep: pd.DataFrame
    selected_policy: EconomicPolicy
    validation_metrics: EvaluationMetrics
    test_metrics: EvaluationMetrics
    comparison: pd.DataFrame


def _outcomes(complete: pd.DataFrame, case_ids: pd.Series) -> pd.DataFrame:
    return complete.set_index("case_id").loc[
        case_ids,
        ["actual_won", "actual_recovery_amount", "defense_cost"],
    ].reset_index(drop=True)


def _evaluate_policy(
    cases: pd.DataFrame,
    probabilities: pd.Series,
    outcomes: pd.DataFrame,
    policy: EconomicPolicy,
) -> tuple[pd.DataFrame, EvaluationMetrics]:
    records = decide_cases(cases, probabilities, policy)
    metrics = calculate_metrics(records["decision"], outcomes)
    return records, metrics


def _sweep_row(policy: EconomicPolicy, records: pd.DataFrame, metrics: EvaluationMetrics) -> dict:
    return {
        "minimum_expected_net_value": policy.minimum_expected_net_value,
        "defend_rate": records["decision"].eq("DEFEND").mean(),
        "accept_rate": records["decision"].eq("ACCEPT").mean(),
        "review_rate": records["decision"].eq("REVIEW").mean(),
        "precision": metrics.precision,
        "recall": metrics.recall,
        "recovered_amount": metrics.amount_recovered,
        "defense_cost": metrics.defense_cost,
        "net_value": metrics.net_economic_value,
        "foregone_recovery": metrics.foregone_recovery,
    }


def run_economic_experiment(
    observable_path: Path = DEFAULT_OBSERVABLE_PATH,
    complete_path: Path = DEFAULT_COMPLETE_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> EconomicExperimentResult:
    """Select policy margin on validation, then evaluate that frozen policy on test."""
    observable = pd.read_csv(observable_path)
    complete = pd.read_csv(complete_path)
    train_ids = load_split_case_ids("train", manifest_path)
    validation_ids = load_split_case_ids("validation", manifest_path)
    test_ids = load_split_case_ids("test", manifest_path)

    train_cases = _cases_for_ids(observable, train_ids)
    train_labels = complete.set_index("case_id").loc[train_ids, "actual_won"]
    model = train_model(train_cases, train_labels)

    validation_cases = _cases_for_ids(observable, validation_ids)
    validation_probabilities = predict_win_probabilities(model, validation_cases)
    validation_outcomes = _outcomes(complete, validation_ids)
    validation_results = []
    validation_by_margin = {}
    for margin in MARGIN_CANDIDATES:
        policy = EconomicPolicy(margin, *REVIEW_PROBABILITY_BOUNDS)
        records, metrics = _evaluate_policy(
            validation_cases, validation_probabilities, validation_outcomes, policy
        )
        validation_results.append(_sweep_row(policy, records, metrics))
        validation_by_margin[margin] = metrics
    validation_sweep = pd.DataFrame(validation_results)
    selected_margin = float(
        validation_sweep.loc[
            validation_sweep["net_value"].idxmax(), "minimum_expected_net_value"
        ]
    )
    selected_policy = EconomicPolicy(selected_margin, *REVIEW_PROBABILITY_BOUNDS)

    # The train fit, review band, margin sweep, and chosen policy are fixed here.
    # Test outcomes are only accessed after the test decisions are made.
    test_cases = _cases_for_ids(observable, test_ids)
    test_probabilities = predict_win_probabilities(model, test_cases)
    economic_records = decide_cases(test_cases, test_probabilities, selected_policy)
    decisions_by_policy = {
        "Always accept": always_accept(test_cases),
        "Always defend": always_defend(test_cases),
        "Rules baseline": predict_decisions(test_cases),
        "Logistic regression (historical threshold 0.25)": _decisions_from_probabilities(
            test_probabilities, HISTORICAL_LOGISTIC_THRESHOLD
        ),
        "Economic policy": economic_records["decision"],
    }
    test_outcomes = _outcomes(complete, test_ids)
    decisions_by_policy[ORACLE_POLICY_NAME] = oracle_ceiling(test_outcomes["actual_won"])
    raw_metrics = {
        name: calculate_metrics(decisions, test_outcomes)
        for name, decisions in decisions_by_policy.items()
    }
    oracle_net_value = raw_metrics[ORACLE_POLICY_NAME].net_economic_value
    comparison = pd.DataFrame(
        [
            {
                "policy": name,
                "defend_rate": decisions.eq("DEFEND").mean(),
                "accept_rate": decisions.eq("ACCEPT").mean(),
                "review_rate": decisions.eq("REVIEW").mean(),
                "precision": metrics.precision,
                "recall": metrics.recall,
                "false_positive_rate": metrics.false_positive_rate,
                "recovered_amount": metrics.amount_recovered,
                "defense_cost": metrics.defense_cost,
                "net_value": metrics.net_economic_value,
                "foregone_recovery": metrics.foregone_recovery,
                "oracle_value_captured": calculate_value_capture(
                    metrics.net_economic_value, oracle_net_value
                ),
            }
            for name, decisions in decisions_by_policy.items()
            for metrics in [raw_metrics[name]]
        ]
    )
    return EconomicExperimentResult(
        validation_sweep=validation_sweep,
        selected_policy=selected_policy,
        validation_metrics=validation_by_margin[selected_margin],
        test_metrics=raw_metrics["Economic policy"],
        comparison=comparison,
    )


def _format_table(table: pd.DataFrame) -> str:
    formatted = table.copy()
    for column in (
        "defend_rate", "accept_rate", "review_rate", "precision", "recall",
        "false_positive_rate", "oracle_value_captured",
    ):
        if column in formatted:
            formatted[column] = formatted[column].map(
                lambda value: "N/A" if pd.isna(value) else f"{value:.1%}"
            )
    for column in ("recovered_amount", "defense_cost", "net_value", "foregone_recovery"):
        if column in formatted:
            formatted[column] = formatted[column].map(lambda value: f"INR {value:,.2f}")
    return formatted.to_string(index=False)


if __name__ == "__main__":
    result = run_economic_experiment()
    print("Validation policy sweep")
    print(_format_table(result.validation_sweep))
    print("\nSelected policy")
    print(result.selected_policy)
    print("\nHeld-out test comparison")
    print(_format_table(result.comparison))
