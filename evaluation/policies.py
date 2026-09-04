"""Evaluation-only reference policies; none are predictive models."""

from __future__ import annotations

import pandas as pd

from models.baseline import ACCEPT, DEFEND


def always_defend(cases: pd.DataFrame) -> pd.Series:
    """Return DEFEND for every supplied case."""
    return pd.Series(DEFEND, index=cases.index, name="decision")


def always_accept(cases: pd.DataFrame) -> pd.Series:
    """Return ACCEPT for every supplied case."""
    return pd.Series(ACCEPT, index=cases.index, name="decision")


def oracle_ceiling(actual_won: pd.Series) -> pd.Series:
    """Evaluation-only oracle: defend exactly the cases known to win.

    This function accepts only the hidden outcome label and must never be used
    as a model feature or production policy.
    """
    return actual_won.astype(bool).map({True: DEFEND, False: ACCEPT}).rename("decision")
