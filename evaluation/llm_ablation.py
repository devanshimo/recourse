"""Controlled LR-only versus LR-plus-LLM-signals ablation experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data.split import DEFAULT_MANIFEST_PATH, load_split_case_ids
from evaluation.evaluate import DEFAULT_COMPLETE_PATH, DEFAULT_OBSERVABLE_PATH
from evaluation.evaluate_model import (
    _cases_for_ids,
    _decisions_from_probabilities,
    _outcomes_for_ids,
    select_validation_threshold,
)
from evaluation.metrics import EvaluationMetrics, calculate_metrics
from models.llm_evidence import (
    HIDDEN_FIELDS,
    EvidenceProvider,
    EvidenceRequest,
    extract_evidence,
    model_feature_signals,
)
from models.logistic_regression import (
    CATEGORICAL_FEATURE_COLUMNS,
    MODEL_SEED,
    NUMERIC_FEATURE_COLUMNS,
    predict_win_probabilities,
    select_features,
    train_model,
)

LLM_FEATURE_COLUMNS = ("contradiction_confidence", "new_signal_present")


class HeuristicMockEvidenceProvider:
    """Deterministic text-only mock used solely for this offline ablation.

    It identifies the claim from dispute wording.  A contradiction is emitted
    only after comparing that claim with allowlisted merchant evidence.  The
    `new_signal_present` flag comes from wording variants, not a structured
    field extraction.
    """

    def extract(self, request: EvidenceRequest) -> dict[str, object]:
        text = request.dispute_text.lower()
        evidence = request.merchant_evidence
        if any(phrase in text for phrase in ("never arrived", "not received", "still waiting")):
            claim = "item_not_received"
        elif any(phrase in text for phrase in ("did not authorize", "unauthorized", "do not recognize")):
            claim = "unauthorized_transaction"
        elif any(phrase in text for phrase in ("different from", "did not match", "not as expected", "materially different")):
            claim = "product_not_as_described"
        else:
            claim = "other"

        contradiction_token = None
        if claim == "item_not_received" and evidence.get("delivery_confirmed") is True:
            contradiction_token = "delivery_confirmed=true"
        elif (
            claim == "unauthorized_transaction"
            and evidence.get("device_seen_before") is True
            and evidence.get("location_consistent") is True
        ):
            contradiction_token = "device_seen_before=true"

        new_signal = any(
            phrase in text
            for phrase in ("still waiting", "do not recognize", "materially different")
        )
        return {
            "customer_claim": claim,
            "claim_confidence": 0.90 if claim != "other" else 0.50,
            "contradicts_merchant_evidence": contradiction_token is not None,
            "contradiction_confidence": 0.80 if contradiction_token else 0.0,
            "contradiction_detail": (
                f"{contradiction_token} conflicts with the customer claim."
                if contradiction_token
                else "No grounded contradiction identified."
            ),
            "new_signal_present": new_signal,
        }


def llm_feature_frame(
    cases: pd.DataFrame,
    provider: EvidenceProvider,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build exactly two gated model features and separate diagnostics.

    This function refuses hidden outcomes and split labels before any provider
    request is made.  Claim and explanation output remain diagnostics-only.
    """
    leaked_columns = HIDDEN_FIELDS & set(cases.columns)
    if leaked_columns:
        raise ValueError(
            "Hidden columns cannot be supplied to LLM extraction: "
            f"{', '.join(sorted(leaked_columns))}"
        )
    signals = []
    diagnostics = []
    for _, case in cases.iterrows():
        result = extract_evidence(case.to_dict(), provider)
        feature_values = model_feature_signals(result)
        signals.append(feature_values)
        diagnostics.append(
            {
                "customer_claim": result.evidence.customer_claim,
                "claim_confidence": result.evidence.claim_confidence,
                "contradiction_detail": result.evidence.contradiction_detail,
                "grounded_high_confidence_contradiction": (
                    feature_values["contradiction_confidence"] >= 0.60
                ),
                "extraction_failed": result.failed,
            }
        )
    return (
        pd.DataFrame(signals, index=cases.index).loc[:, LLM_FEATURE_COLUMNS],
        pd.DataFrame(diagnostics, index=cases.index),
    )


def _augmented_features(cases: pd.DataFrame, llm_features: pd.DataFrame) -> pd.DataFrame:
    if not cases.index.equals(llm_features.index):
        raise ValueError("Cases and LLM features must have matching indices.")
    if set(llm_features.columns) != set(LLM_FEATURE_COLUMNS):
        raise ValueError("LLM feature frame must contain exactly the gated feature columns.")
    return pd.concat([select_features(cases), llm_features], axis=1)


def train_augmented_model(
    train_cases: pd.DataFrame,
    train_labels: pd.Series,
    train_llm_features: pd.DataFrame,
) -> Pipeline:
    """Train the second logistic regression only on training data and signals."""
    numeric_columns = list(NUMERIC_FEATURE_COLUMNS + LLM_FEATURE_COLUMNS)
    preprocessing = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL_FEATURE_COLUMNS)),
            ("numeric", StandardScaler(), numeric_columns),
        ]
    )
    model = Pipeline(
        [
            ("preprocessing", preprocessing),
            ("classifier", LogisticRegression(max_iter=1_000, random_state=MODEL_SEED)),
        ]
    )
    model.fit(_augmented_features(train_cases, train_llm_features), train_labels.astype(bool))
    return model


def predict_augmented_probabilities(
    model: Pipeline,
    cases: pd.DataFrame,
    llm_features: pd.DataFrame,
) -> pd.Series:
    probabilities = model.predict_proba(_augmented_features(cases, llm_features))[:, 1]
    return pd.Series(probabilities, index=cases.index, name="p_win")


@dataclass(frozen=True)
class AblationResult:
    lr_only_threshold: float
    lr_llm_threshold: float
    lr_only_metrics: EvaluationMetrics
    lr_llm_metrics: EvaluationMetrics
    delta: dict[str, float]
    absolute_delta: dict[str, float]
    decision_flips: int
    decision_flip_rate: float
    flips_with_new_signal: int
    flips_with_high_confidence_contradiction: int
    per_dispute_type: pd.DataFrame


def run_llm_ablation(
    observable_path: Path = DEFAULT_OBSERVABLE_PATH,
    complete_path: Path = DEFAULT_COMPLETE_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    provider: EvidenceProvider | None = None,
) -> AblationResult:
    """Run the controlled split-preserving ablation with no test-set fitting."""
    provider = provider or HeuristicMockEvidenceProvider()
    observable = pd.read_csv(observable_path)
    complete = pd.read_csv(complete_path)
    train_ids = load_split_case_ids("train", manifest_path)
    validation_ids = load_split_case_ids("validation", manifest_path)
    test_ids = load_split_case_ids("test", manifest_path)

    train_cases = _cases_for_ids(observable, train_ids)
    validation_cases = _cases_for_ids(observable, validation_ids)
    test_cases = _cases_for_ids(observable, test_ids)
    train_outcomes = _outcomes_for_ids(complete, train_ids)
    validation_outcomes = _outcomes_for_ids(complete, validation_ids)

    train_llm_features, _ = llm_feature_frame(train_cases, provider)
    validation_llm_features, _ = llm_feature_frame(validation_cases, provider)
    test_llm_features, test_diagnostics = llm_feature_frame(test_cases, provider)

    lr_only_model = train_model(train_cases, train_outcomes["actual_won"])
    lr_llm_model = train_augmented_model(
        train_cases, train_outcomes["actual_won"], train_llm_features
    )
    lr_only_validation_probabilities = predict_win_probabilities(lr_only_model, validation_cases)
    lr_llm_validation_probabilities = predict_augmented_probabilities(
        lr_llm_model, validation_cases, validation_llm_features
    )
    lr_only_threshold, _ = select_validation_threshold(
        lr_only_validation_probabilities, validation_outcomes
    )
    lr_llm_threshold, _ = select_validation_threshold(
        lr_llm_validation_probabilities, validation_outcomes
    )

    # Test labels are accessed only after both models and thresholds are fixed.
    lr_only_test_decisions = _decisions_from_probabilities(
        predict_win_probabilities(lr_only_model, test_cases), lr_only_threshold
    )
    lr_llm_test_decisions = _decisions_from_probabilities(
        predict_augmented_probabilities(lr_llm_model, test_cases, test_llm_features),
        lr_llm_threshold,
    )
    test_outcomes = _outcomes_for_ids(complete, test_ids)
    lr_only_metrics = calculate_metrics(lr_only_test_decisions, test_outcomes)
    lr_llm_metrics = calculate_metrics(lr_llm_test_decisions, test_outcomes)

    metric_names = (
        "precision", "recall", "false_positive_rate", "false_negative_rate",
        "amount_recovered", "defense_cost", "net_economic_value", "foregone_recovery",
    )
    delta = {
        name: float(getattr(lr_llm_metrics, name) - getattr(lr_only_metrics, name))
        for name in metric_names
    }
    delta["defend_rate"] = float(
        lr_llm_test_decisions.eq("DEFEND").mean()
        - lr_only_test_decisions.eq("DEFEND").mean()
    )
    flips = lr_only_test_decisions.ne(lr_llm_test_decisions)
    per_dispute_type = _per_dispute_type_comparison(
        test_cases, test_outcomes, lr_only_test_decisions, lr_llm_test_decisions
    )
    return AblationResult(
        lr_only_threshold=lr_only_threshold,
        lr_llm_threshold=lr_llm_threshold,
        lr_only_metrics=lr_only_metrics,
        lr_llm_metrics=lr_llm_metrics,
        delta=delta,
        absolute_delta={name: abs(value) for name, value in delta.items()},
        decision_flips=int(flips.sum()),
        decision_flip_rate=float(flips.mean()),
        flips_with_new_signal=int(test_llm_features.loc[flips, "new_signal_present"].sum()),
        flips_with_high_confidence_contradiction=int(
            test_diagnostics.loc[flips, "grounded_high_confidence_contradiction"].sum()
        ),
        per_dispute_type=per_dispute_type,
    )


def _per_dispute_type_comparison(
    cases: pd.DataFrame,
    outcomes: pd.DataFrame,
    lr_only_decisions: pd.Series,
    lr_llm_decisions: pd.Series,
) -> pd.DataFrame:
    rows = []
    for dispute_type, group in cases.groupby("dispute_type", sort=True):
        indices = group.index
        for name, decisions in (("LR-only", lr_only_decisions), ("LR + LLM", lr_llm_decisions)):
            metrics = calculate_metrics(decisions.loc[indices], outcomes.loc[indices])
            rows.append(
                {
                    "dispute_type": dispute_type,
                    "model": name,
                    "cases": len(group),
                    "defend_rate": decisions.loc[indices].eq("DEFEND").mean(),
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "net_economic_value": metrics.net_economic_value,
                }
            )
    return pd.DataFrame(rows)


def _summary_row(name: str, metrics: EvaluationMetrics) -> dict[str, object]:
    total_cases = (
        metrics.true_positives
        + metrics.false_positives
        + metrics.false_negatives
        + metrics.true_negatives
    )
    return {
        "model": name,
        "defend_rate": (metrics.true_positives + metrics.false_positives) / total_cases,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "fpr": metrics.false_positive_rate,
        "fnr": metrics.false_negative_rate,
        "recovered_amount": metrics.amount_recovered,
        "defense_cost": metrics.defense_cost,
        "net_economic_value": metrics.net_economic_value,
        "foregone_recovery": metrics.foregone_recovery,
    }


if __name__ == "__main__":
    result = run_llm_ablation()
    print("Validation-selected thresholds:", result.lr_only_threshold, result.lr_llm_threshold)
    print(
        pd.DataFrame(
            [
                _summary_row("LR-only", result.lr_only_metrics),
                _summary_row("LR + LLM", result.lr_llm_metrics),
            ]
        ).to_string(index=False)
    )
    print("Deltas (LR + LLM minus LR-only):", result.delta)
    print("Absolute deltas:", result.absolute_delta)
    print(
        "Decision flips:", result.decision_flips,
        f"({result.decision_flip_rate:.1%}); new signal:", result.flips_with_new_signal,
        "; high-confidence contradiction:", result.flips_with_high_confidence_contradiction,
    )
    print(result.per_dispute_type.to_string(index=False))
