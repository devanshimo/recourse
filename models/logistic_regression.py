"""First transparent probabilistic model experiment for Recourse."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_SEED = 42
HIDDEN_OUTCOME_COLUMNS = {
    "actual_defensible",
    "actual_won",
    "actual_recovery_amount",
    "defense_cost",
}
CATEGORICAL_FEATURE_COLUMNS = ("payment_method", "dispute_type")
NUMERIC_FEATURE_COLUMNS = (
    "amount",
    "transaction_age_days",
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
)
FEATURE_COLUMNS = CATEGORICAL_FEATURE_COLUMNS + NUMERIC_FEATURE_COLUMNS


def select_features(cases: pd.DataFrame) -> pd.DataFrame:
    """Return the structured observable features and reject hidden columns."""
    leaked_columns = HIDDEN_OUTCOME_COLUMNS & set(cases.columns)
    if leaked_columns:
        raise ValueError(
            "Hidden outcome columns cannot be model features: "
            f"{', '.join(sorted(leaked_columns))}"
        )
    missing_columns = set(FEATURE_COLUMNS) - set(cases.columns)
    if missing_columns:
        raise ValueError(
            "Missing model feature columns: "
            f"{', '.join(sorted(missing_columns))}"
        )
    return cases.loc[:, FEATURE_COLUMNS].copy()


def build_model() -> Pipeline:
    """Build the reproducible preprocessing and logistic-regression pipeline."""
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(CATEGORICAL_FEATURE_COLUMNS),
            ),
            ("numeric", StandardScaler(), list(NUMERIC_FEATURE_COLUMNS)),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(max_iter=1_000, random_state=MODEL_SEED),
            ),
        ]
    )


def train_model(
    train_cases: pd.DataFrame,
    train_labels: pd.Series,
) -> Pipeline:
    """Fit only on training observables and their `actual_won` labels."""
    if len(train_cases) != len(train_labels):
        raise ValueError("Training cases and labels must have the same length.")
    model = build_model()
    model.fit(select_features(train_cases), train_labels.astype(bool))
    return model


def predict_win_probabilities(model: Pipeline, cases: pd.DataFrame) -> pd.Series:
    """Return P(actual_won=True) for observable cases."""
    probabilities = model.predict_proba(select_features(cases))[:, 1]
    return pd.Series(probabilities, index=cases.index, name="p_win")
