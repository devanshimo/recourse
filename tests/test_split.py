import unittest
from pathlib import Path

import pandas as pd

from data.split import SPLIT_NAMES, create_split_assignments


class SplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.complete_cases = pd.read_csv(
            Path(__file__).resolve().parents[1] / "data" / "chargebacks_complete.csv"
        )
        cls.assignments = create_split_assignments(cls.complete_cases)

    def test_split_ids_do_not_overlap(self):
        split_ids = [
            set(self.assignments.loc[self.assignments["split"] == split, "case_id"])
            for split in SPLIT_NAMES
        ]
        self.assertFalse(split_ids[0] & split_ids[1])
        self.assertFalse(split_ids[0] & split_ids[2])
        self.assertFalse(split_ids[1] & split_ids[2])

    def test_every_case_appears_once(self):
        self.assertEqual(len(self.assignments), len(self.complete_cases))
        self.assertTrue(self.assignments["case_id"].is_unique)
        self.assertEqual(
            set(self.assignments["case_id"]), set(self.complete_cases["case_id"])
        )

    def test_split_is_reproducible(self):
        repeated_assignments = create_split_assignments(self.complete_cases)
        pd.testing.assert_frame_equal(self.assignments, repeated_assignments)


if __name__ == "__main__":
    unittest.main()
