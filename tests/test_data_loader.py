"""Input validity, immutable loading, and outcome-independent orientation."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.data_loader import load_games, normalize_games, safe_output_directory, validate_raw

FIXTURE = Path(__file__).parent / "fixtures/synthetic_games.csv"


class DataLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = pd.read_csv(FIXTURE, dtype=str).iloc[:4].copy()

    def test_requires_explicit_synthetic_opt_in(self) -> None:
        with self.assertRaisesRegex(ValueError, "Synthetic"):
            load_games(FIXTURE)

    def test_rejects_mixed_provenance(self) -> None:
        self.raw.loc[0, "is_synthetic"] = "false"
        with self.assertRaisesRegex(ValueError, "Mixing"):
            validate_raw(self.raw, allow_synthetic=True)

    def test_does_not_modify_raw_bytes(self) -> None:
        before = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
        games = load_games(FIXTURE, allow_synthetic=True)
        self.assertEqual(len(games), 96)
        self.assertEqual(before, hashlib.sha256(FIXTURE.read_bytes()).hexdigest())

    def test_orientation_is_independent_of_row_order_and_winner(self) -> None:
        first = normalize_games(validate_raw(self.raw, allow_synthetic=True))
        changed = self.raw.iloc[::-1].copy()
        changed["win"] = 1 - pd.to_numeric(changed.win)
        second = normalize_games(validate_raw(changed, allow_synthetic=True))
        pd.testing.assert_series_equal(first.team_A, second.team_A)
        pd.testing.assert_series_equal(first.team_B, second.team_B)
        self.assertTrue((first.team_A_win == 1 - second.team_A_win).all())

    def test_rejects_duplicates_and_partial_games(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_raw(pd.concat([self.raw, self.raw.iloc[:1]]), allow_synthetic=True)
        with self.assertRaisesRegex(ValueError, "exactly two"):
            validate_raw(self.raw.iloc[:3], allow_synthetic=True)

    def test_rejects_invalid_results_and_opponents(self) -> None:
        invalid = self.raw.copy()
        invalid.loc[:1, "win"] = "1"
        with self.assertRaisesRegex(ValueError, "exactly one winner"):
            validate_raw(invalid, allow_synthetic=True)
        invalid = self.raw.copy()
        invalid.loc[0, "opponent"] = "Wrong Team"
        with self.assertRaisesRegex(ValueError, "reciprocal"):
            validate_raw(invalid, allow_synthetic=True)

    def test_requires_timezone_and_result_after_completion(self) -> None:
        invalid = self.raw.copy()
        invalid.loc[0, "start_time"] = "2024-01-01 12:00:00"
        with self.assertRaisesRegex(ValueError, "timezone"):
            validate_raw(invalid, allow_synthetic=True)
        invalid = self.raw.copy()
        invalid["available_at"] = invalid.start_time
        with self.assertRaisesRegex(ValueError, "strictly after"):
            validate_raw(invalid, allow_synthetic=True)
        invalid = self.raw.copy()
        invalid["duration_seconds"] = "1000000"
        with self.assertRaisesRegex(ValueError, "completion"):
            validate_raw(invalid, allow_synthetic=True)

    def test_missing_stats_stay_missing(self) -> None:
        games = normalize_games(validate_raw(self.raw.drop(columns=["gold_diff_10"]), allow_synthetic=True))
        self.assertTrue(games.gold_diff_10_A.isna().all())
        self.assertTrue(games.gold_diff_10_B.isna().all())

    def test_rejects_inconsistent_paired_statistics(self) -> None:
        cases = (
            ("xp_diff_10", (100, -100), "999", "must negate"),
            ("dragon_control", (0.5, 0.5), "0.25", "must complement"),
            ("first_blood", (0, 1), None, "one-sided missing"),
            ("duration_seconds", (1000, 1000), "1234", "inconsistent duration"),
        )
        for column, pair, value, message in cases:
            with self.subTest(column=column):
                invalid = self.raw.copy()
                invalid.loc[0:1, column] = pair
                invalid.loc[0, column] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate_raw(invalid, allow_synthetic=True)

    def test_patch_string_is_preserved(self) -> None:
        self.raw["patch"] = "14.10"
        games = normalize_games(validate_raw(self.raw, allow_synthetic=True))
        self.assertEqual(games.patch.iloc[0], "14.10")

    def test_source_protection_and_missing_file_message(self) -> None:
        with self.assertRaisesRegex(ValueError, "immutable"):
            safe_output_directory(Path(__file__).parents[1] / "data/raw/generated")
        with self.assertRaisesRegex(ValueError, "input CSV"):
            safe_output_directory(FIXTURE.parent, [FIXTURE])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "No historical CSV"):
                load_games(directory)


if __name__ == "__main__":
    unittest.main()
