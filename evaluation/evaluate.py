"""Run the rules-only baseline against hidden synthetic outcomes."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data.split import DEFAULT_MANIFEST_PATH, SPLIT_NAMES, load_split_case_ids
from evaluation.metrics import EvaluationMetrics, calculate_metrics
from models.baseline import predict_decisions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OBSERVABLE_PATH = PROJECT_ROOT / "data" / "chargebacks_observable.csv"
DEFAULT_COMPLETE_PATH = PROJECT_ROOT / "data" / "chargebacks_complete.csv"


def evaluate_baseline(
    observable_path: Path = DEFAULT_OBSERVABLE_PATH,
    complete_path: Path = DEFAULT_COMPLETE_PATH,
    split: str | None = None,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> EvaluationMetrics:
    """Make decisions from observables, then join hidden fields for scoring."""
    observable = pd.read_csv(observable_path)
    complete = pd.read_csv(complete_path)

    if observable["case_id"].duplicated().any() or complete["case_id"].duplicated().any():
        raise ValueError("case_id must be unique in both datasets.")
    if set(observable["case_id"]) != set(complete["case_id"]):
        raise ValueError("Observable and complete datasets must contain identical case_ids.")
    if split is not None:
        split_case_ids = load_split_case_ids(split, manifest_path)
        observable = observable.set_index("case_id").loc[split_case_ids].reset_index()

    # The baseline receives only `observable`; hidden fields are used below only
    # after its decisions have been generated.
    decisions = predict_decisions(observable)
    outcomes = complete.set_index("case_id").loc[
        observable["case_id"],
        ["actual_won", "actual_recovery_amount", "defense_cost"],
    ].reset_index(drop=True)
    return calculate_metrics(decisions, outcomes)


def format_report(
    metrics: EvaluationMetrics,
    title: str = "Naive rules-only baseline evaluation",
) -> str:
    """Format the baseline result without hiding economic assumptions."""
    return "\n".join(
        [
            title,
            "",
            "Confusion matrix (actual_won as positive):",
            f"                 Actual won  Actual lost",
            f"DEFEND          {metrics.true_positives:>10}  {metrics.false_positives:>11}",
            f"ACCEPT          {metrics.false_negatives:>10}  {metrics.true_negatives:>11}",
            "",
            f"Precision:            {metrics.precision:.3%}",
            f"Recall:               {metrics.recall:.3%}",
            f"False-positive rate:  {metrics.false_positive_rate:.3%}",
            f"False-negative rate:  {metrics.false_negative_rate:.3%}",
            "",
            "Economics (DEFEND cases only):",
            f"Amount recovered:      INR {metrics.amount_recovered:,.2f}",
            f"Defense cost:          INR {metrics.defense_cost:,.2f}",
            f"Net economic value:    INR {metrics.net_economic_value:,.2f}",
            "",
            "Net economic value = amount recovered - defense cost.",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observable-path", type=Path, default=DEFAULT_OBSERVABLE_PATH)
    parser.add_argument("--complete-path", type=Path, default=DEFAULT_COMPLETE_PATH)
    parser.add_argument("--split", choices=SPLIT_NAMES)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    args = parser.parse_args()
    print(
        format_report(
            evaluate_baseline(
                args.observable_path,
                args.complete_path,
                args.split,
                args.manifest_path,
            )
        )
    )


if __name__ == "__main__":
    main()
