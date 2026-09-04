"""Evaluate the first logistic-regression Recourse experiment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data.split import DEFAULT_MANIFEST_PATH, load_split_case_ids
from evaluation.evaluate import (
    DEFAULT_COMPLETE_PATH,
    DEFAULT_OBSERVABLE_PATH,
    evaluate_baseline,
    format_report,
)
from evaluation.metrics import EvaluationMetrics, calculate_metrics
from models.baseline import ACCEPT, DEFEND
from models.logistic_regression import predict_win_probabilities, train_model

THRESHOLD_CANDIDATES = tuple(np.round(np.arange(0.20, 0.81, 0.05), 2))


@dataclass(frozen=True)
class ModelExperimentResult:
    validation_threshold: float
    validation_metrics: EvaluationMetrics
    test_metrics: EvaluationMetrics
    test_probabilities: pd.Series
    baseline_test_metrics: EvaluationMetrics


def _cases_for_ids(cases: pd.DataFrame, case_ids: pd.Series) -> pd.DataFrame:
    return cases.set_index("case_id").loc[case_ids].reset_index()


def _outcomes_for_ids(complete: pd.DataFrame, case_ids: pd.Series) -> pd.DataFrame:
    return complete.set_index("case_id").loc[
        case_ids,
        ["actual_won", "actual_recovery_amount", "defense_cost"],
    ].reset_index(drop=True)


def _decisions_from_probabilities(probabilities: pd.Series, threshold: float) -> pd.Series:
    return pd.Series(
        np.where(probabilities >= threshold, DEFEND, ACCEPT),
        index=probabilities.index,
        name="decision",
    )


def select_validation_threshold(
    probabilities: pd.Series,
    validation_outcomes: pd.DataFrame,
) -> tuple[float, EvaluationMetrics]:
    """Pick the validation-only threshold with highest net economic value.

    The small, predeclared grid prevents test-set tuning.  In an exact tie, the
    higher threshold is chosen because it avoids defending extra cases for no
    added validation value.
    """
    candidates = [
        (
            threshold,
            calculate_metrics(
                _decisions_from_probabilities(probabilities, threshold),
                validation_outcomes,
            ),
        )
        for threshold in THRESHOLD_CANDIDATES
    ]
    return max(candidates, key=lambda item: (item[1].net_economic_value, item[0]))


def run_model_experiment(
    observable_path: Path = DEFAULT_OBSERVABLE_PATH,
    complete_path: Path = DEFAULT_COMPLETE_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> ModelExperimentResult:
    """Train on train, select threshold on validation, evaluate once on test."""
    observable = pd.read_csv(observable_path)
    complete = pd.read_csv(complete_path)
    if set(observable["case_id"]) != set(complete["case_id"]):
        raise ValueError("Observable and complete datasets must contain identical case_ids.")

    train_ids = load_split_case_ids("train", manifest_path)
    validation_ids = load_split_case_ids("validation", manifest_path)
    test_ids = load_split_case_ids("test", manifest_path)

    train_cases = _cases_for_ids(observable, train_ids)
    train_outcomes = _outcomes_for_ids(complete, train_ids)
    model = train_model(train_cases, train_outcomes["actual_won"])

    validation_cases = _cases_for_ids(observable, validation_ids)
    validation_outcomes = _outcomes_for_ids(complete, validation_ids)
    validation_probabilities = predict_win_probabilities(model, validation_cases)
    threshold, validation_metrics = select_validation_threshold(
        validation_probabilities, validation_outcomes
    )

    # Test outcomes are accessed only after model fitting and threshold selection.
    test_cases = _cases_for_ids(observable, test_ids)
    test_probabilities = predict_win_probabilities(model, test_cases)
    test_metrics = calculate_metrics(
        _decisions_from_probabilities(test_probabilities, threshold),
        _outcomes_for_ids(complete, test_ids),
    )

    return ModelExperimentResult(
        validation_threshold=threshold,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        test_probabilities=test_probabilities,
        baseline_test_metrics=evaluate_baseline(
            observable_path, complete_path, "test", manifest_path
        ),
    )


def format_model_report(result: ModelExperimentResult) -> str:
    """Report probability output, validation selection, and held-out results."""
    probability_summary = result.test_probabilities.agg(["min", "mean", "max"])
    return "\n".join(
        [
            "Logistic-regression Recourse experiment",
            "",
            "Validation threshold selection:",
            "Selected the predeclared threshold with the greatest validation net economic value.",
            f"Selected threshold: {result.validation_threshold:.2f}",
            f"Validation net economic value: INR {result.validation_metrics.net_economic_value:,.2f}",
            "",
            "Held-out test probability predictions:",
            f"P(win) minimum / mean / maximum: {probability_summary['min']:.3f} / "
            f"{probability_summary['mean']:.3f} / {probability_summary['max']:.3f}",
            "",
            "Validation metrics:",
            format_report(
                result.validation_metrics,
                "Logistic-regression validation evaluation",
            ),
            "",
            "Final held-out test metrics:",
            format_report(
                result.test_metrics,
                "Logistic-regression held-out test evaluation",
            ),
            "",
            "Naive baseline on the same held-out test cases:",
            format_report(result.baseline_test_metrics),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observable-path", type=Path, default=DEFAULT_OBSERVABLE_PATH)
    parser.add_argument("--complete-path", type=Path, default=DEFAULT_COMPLETE_PATH)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    args = parser.parse_args()
    print(format_model_report(run_model_experiment(**vars(args))))


if __name__ == "__main__":
    main()
