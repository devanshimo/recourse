"""Create and load the fixed train/validation/test split manifest."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

SPLIT_SEED = 42
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_RATIOS = (0.60, 0.20, 0.20)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPLETE_PATH = PROJECT_ROOT / "data" / "chargebacks_complete.csv"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "split_assignments.csv"


def create_split_assignments(
    complete_cases: pd.DataFrame,
    seed: int = SPLIT_SEED,
) -> pd.DataFrame:
    """Create a deterministic split stratified by the hidden `actual_won` label.

    The returned manifest deliberately contains only `case_id` and `split`.
    `actual_won` is used solely while assigning the splits and never appears in
    data that a future model can consume as a feature.
    """
    required_columns = {"case_id", "actual_won"}
    missing_columns = required_columns - set(complete_cases.columns)
    if missing_columns:
        raise ValueError(
            "Missing columns required to create the split: "
            f"{', '.join(sorted(missing_columns))}"
        )
    if complete_cases["case_id"].duplicated().any():
        raise ValueError("case_id must be unique before splitting.")

    assignments: list[dict[str, str]] = []
    randomizer = random.Random(seed)
    for _, outcome_group in complete_cases.groupby("actual_won", sort=True):
        case_ids = outcome_group["case_id"].tolist()
        randomizer.shuffle(case_ids)

        train_end = round(len(case_ids) * SPLIT_RATIOS[0])
        validation_end = train_end + round(len(case_ids) * SPLIT_RATIOS[1])
        split_case_ids = {
            "train": case_ids[:train_end],
            "validation": case_ids[train_end:validation_end],
            "test": case_ids[validation_end:],
        }
        for split, ids in split_case_ids.items():
            assignments.extend({"case_id": case_id, "split": split} for case_id in ids)

    return (
        pd.DataFrame(assignments)
        .sort_values("case_id")
        .reset_index(drop=True)
    )


def save_split_assignments(
    complete_path: Path = DEFAULT_COMPLETE_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> pd.DataFrame:
    """Create and persist the reusable case-ID split manifest."""
    assignments = create_split_assignments(pd.read_csv(complete_path))
    assignments.to_csv(manifest_path, index=False)
    return assignments


def load_split_case_ids(
    split: str,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> pd.Series:
    """Return case IDs for one named split without loading hidden outcomes."""
    if split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of: {', '.join(SPLIT_NAMES)}")
    assignments = pd.read_csv(manifest_path)
    return assignments.loc[assignments["split"] == split, "case_id"].copy()


if __name__ == "__main__":
    saved = save_split_assignments()
    print(saved["split"].value_counts().reindex(SPLIT_NAMES).to_string())
