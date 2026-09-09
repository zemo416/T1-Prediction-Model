"""Critical time-boundary and delayed-result split checks."""

import unittest

import pandas as pd

from src.preprocessing import chronological_split, walk_forward_splits


def temporal_frame(count: int = 30) -> pd.DataFrame:
    times = pd.date_range("2024-01-01", periods=count, freq="D", tz="UTC")
    return pd.DataFrame({
        "game_id": [f"split-{index}" for index in range(count)],
        "start_time": times,
        "available_at": times + pd.Timedelta(hours=1),
    })


class ChronologicalSplittingTests(unittest.TestCase):
    def test_same_timestamp_stays_in_one_partition(self):
        frame = temporal_frame(10)
        duplicates = frame.copy()
        duplicates["game_id"] += "-other"
        shuffled = pd.concat([frame, duplicates]).sample(frac=1, random_state=13)
        split = chronological_split(shuffled)
        self.assertEqual([len(split.train), len(split.validation), len(split.test)], [12, 4, 4])
        groups = [set(part.start_time) for part in (split.train, split.validation, split.test)]
        self.assertFalse(groups[0] & groups[1] or groups[1] & groups[2] or groups[0] & groups[2])
        self.assertLess(split.train.start_time.max(), split.validation.start_time.min())
        self.assertLess(split.validation.start_time.max(), split.test.start_time.min())

    def test_results_available_exactly_at_boundary_are_purged(self):
        frame = temporal_frame(10)
        frame.loc[5, "available_at"] = frame.loc[6, "start_time"]
        frame.loc[7, "available_at"] = frame.loc[8, "start_time"] + pd.Timedelta(days=1)
        split = chronological_split(frame)
        self.assertEqual(split.purged_train, 1)
        self.assertEqual(split.purged_validation, 1)
        self.assertNotIn("split-5", split.train.game_id.tolist())
        self.assertNotIn("split-7", split.validation.game_id.tolist())
        self.assertLess(split.train.available_at.max(), split.validation.start_time.min())
        self.assertLess(split.validation.available_at.max(), split.test.start_time.min())

    def test_explicit_cutoffs_are_exclusive_and_require_timezones(self):
        split = chronological_split(temporal_frame(10), train_end="2024-01-05T00:00:00Z", validation_end="2024-01-08T00:00:00Z")
        self.assertEqual([len(split.train), len(split.validation), len(split.test)], [4, 3, 3])
        self.assertEqual(split.validation.game_id.iloc[0], "split-4")
        self.assertEqual(split.test.game_id.iloc[0], "split-7")
        with self.assertRaisesRegex(ValueError, "timezone"):
            chronological_split(temporal_frame(10), train_end="2024-01-05", validation_end="2024-01-08")

    def test_rejects_duplicate_game_ids_and_invalid_boundaries(self):
        frame = temporal_frame(10)
        with self.assertRaisesRegex(ValueError, "game_id"):
            chronological_split(pd.concat([frame, frame]))
        with self.assertRaisesRegex(ValueError, "both"):
            chronological_split(frame, train_end="2024-01-05")
        with self.assertRaisesRegex(ValueError, "precede"):
            chronological_split(frame, train_end="2024-01-08T00:00:00Z", validation_end="2024-01-05T00:00:00Z")
        with self.assertRaisesRegex(ValueError, "fractions"):
            chronological_split(frame, train_fraction=0.9, validation_fraction=0.2)

    def test_cannot_fit_results_that_are_not_available(self):
        frame = temporal_frame(10)
        frame.loc[:5, "available_at"] = pd.Timestamp("2025-01-01", tz="UTC")
        with self.assertRaisesRegex(ValueError, "purging emptied"):
            chronological_split(frame)

    def test_walk_forward_expands_train_and_never_repeats_test_games(self):
        splits = list(walk_forward_splits(temporal_frame(30), n_splits=3))
        self.assertEqual(len(splits), 3)
        seen = set()
        previous_training = set()
        for split in splits:
            self.assertTrue(previous_training <= set(split.train.game_id))
            self.assertFalse(seen & set(split.test.game_id))
            self.assertLess(split.train.available_at.max(), split.validation.start_time.min())
            self.assertLess(split.validation.available_at.max(), split.test.start_time.min())
            seen.update(split.test.game_id)
            previous_training = set(split.train.game_id)
        self.assertEqual(len(seen), 12)

    def test_small_history_cannot_support_requested_walk_forward_folds(self):
        with self.assertRaisesRegex(ValueError, "Not enough"):
            list(walk_forward_splits(temporal_frame(3), n_splits=3))


if __name__ == "__main__":
    unittest.main()
