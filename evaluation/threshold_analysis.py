"""Validation-only operating-point analysis for the logistic-regression model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.split import DEFAULT_MANIFEST_PATH, load_split_case_ids
from evaluation.evaluate import DEFAULT_COMPLETE_PATH, DEFAULT_OBSERVABLE_PATH
from evaluation.evaluate_model import (
    _cases_for_ids,
    _decisions_from_probabilities,
    _outcomes_for_ids,
)
from evaluation.metrics import calculate_metrics
from models.logistic_regression import predict_win_probabilities, train_model

THRESHOLDS = tuple(np.round(np.arange(0.10, 0.91, 0.05), 2))


def validation_threshold_table(
    observable_path: Path = DEFAULT_OBSERVABLE_PATH,
    complete_path: Path = DEFAULT_COMPLETE_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> pd.DataFrame:
    """Train on train and evaluate every candidate threshold on validation only."""
    observable = pd.read_csv(observable_path)
    complete = pd.read_csv(complete_path)
    train_ids = load_split_case_ids("train", manifest_path)
    validation_ids = load_split_case_ids("validation", manifest_path)

    train_cases = _cases_for_ids(observable, train_ids)
    train_outcomes = _outcomes_for_ids(complete, train_ids)
    model = train_model(train_cases, train_outcomes["actual_won"])

    validation_cases = _cases_for_ids(observable, validation_ids)
    validation_outcomes = _outcomes_for_ids(complete, validation_ids)
    probabilities = predict_win_probabilities(model, validation_cases)

    rows = []
    for threshold in THRESHOLDS:
        decisions = _decisions_from_probabilities(probabilities, threshold)
        metrics = calculate_metrics(decisions, validation_outcomes)
        f1 = (
            2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)
            if metrics.precision + metrics.recall
            else 0.0
        )
        rows.append(
            {
                "threshold": threshold,
                "defend_rate": decisions.eq("DEFEND").mean(),
                "precision": metrics.precision,
                "recall": metrics.recall,
                "false_positive_rate": metrics.false_positive_rate,
                "false_negative_rate": metrics.false_negative_rate,
                "recovered_amount": metrics.amount_recovered,
                "defense_cost": metrics.defense_cost,
                "net_economic_value": metrics.net_economic_value,
                "f1": f1,
            }
        )
    return pd.DataFrame(rows)


def select_operating_points(table: pd.DataFrame) -> dict[str, pd.Series]:
    """Select four transparent validation-only policy operating points."""
    economic = table.loc[table["net_economic_value"].idxmax()]
    f1 = table.loc[table["f1"].idxmax()]

    target_recall = table.loc[table["recall"].between(0.60, 0.75)]
    if target_recall.empty:
        target_recall = table.assign(recall_distance=(table["recall"] - 0.675).abs())
        restrained = target_recall.loc[target_recall["recall_distance"].idxmin()]
    else:
        # Lowest defend rate in the desired recall range reduces unnecessary work.
        restrained = target_recall.sort_values(
            ["defend_rate", "threshold"], ascending=[True, False]
        ).iloc[0]

    recall_floor = table.loc[table["recall"] >= 0.70]
    best_precision = recall_floor.sort_values(
        ["precision", "threshold"], ascending=[False, False]
    ).iloc[0]
    return {
        "max_net_economic_value": economic,
        "max_f1": f1,
        "60_to_75_recall_restrained": restrained,
        "best_precision_recall_at_least_70": best_precision,
    }


def format_threshold_table(table: pd.DataFrame) -> str:
    """Format the complete validation table for terminal review."""
    formatted = table.copy()
    for column in (
        "defend_rate",
        "precision",
        "recall",
        "false_positive_rate",
        "false_negative_rate",
    ):
        formatted[column] = formatted[column].map(lambda value: f"{value:.1%}")
    for column in ("recovered_amount", "defense_cost", "net_economic_value"):
        formatted[column] = formatted[column].map(lambda value: f"INR {value:,.2f}")
    formatted["f1"] = formatted["f1"].map(lambda value: f"{value:.3f}")
    return formatted.to_string(index=False)


if __name__ == "__main__":
    table = validation_threshold_table()
    print(format_threshold_table(table))
    print("\nOperating points")
    for name, operating_point in select_operating_points(table).items():
        print(f"{name}: threshold={operating_point['threshold']:.2f}")
