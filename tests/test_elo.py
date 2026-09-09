"""Critical Elo math, event timing, and roster regression checks."""

import math
import unittest

import pandas as pd

from src.config import FeatureConfig
from src.elo import EloSystem, expected_score


class EloTests(unittest.TestCase):
    def test_standard_expectation(self):
        self.assertEqual(expected_score(1500, 1500), 0.5)
        self.assertAlmostEqual(expected_score(1900, 1500), 10 / 11)
        self.assertAlmostEqual(expected_score(1900, 1500) + expected_score(1500, 1900), 1)

    def test_result_is_not_visible_at_its_availability_timestamp(self):
        elo = EloSystem()
        start = pd.Timestamp("2024-01-01T12:00:00Z")
        available = start + pd.Timedelta(hours=1)
        snapshot = elo.queue_game(team_A="A", team_B="B", start_time=start, available_at=available, team_A_win=1)
        self.assertEqual(snapshot.expected_A, 0.5)
        elo.advance(available)
        self.assertEqual(elo.rating("A", available), 1500)
        elo.advance(available + pd.Timedelta(nanoseconds=1))
        aged_update = 12 * math.exp2(-(1 / 24) / 365)
        self.assertAlmostEqual(elo.rating("A", available + pd.Timedelta(nanoseconds=1)), 1500 + aged_update)
        self.assertAlmostEqual(elo.rating("B", available + pd.Timedelta(nanoseconds=1)), 1500 - aged_update)

    def test_overlapping_games_freeze_prestart_expectations(self):
        elo = EloSystem(FeatureConfig(elo_half_life_days=1e9))
        start = pd.Timestamp("2024-01-01T00:00:00Z")
        first = elo.queue_game(team_A="A", team_B="B", start_time=start,
                              available_at=start + pd.Timedelta(hours=4), team_A_win=1)
        second = elo.queue_game(team_A="A", team_B="B", start_time=start + pd.Timedelta(hours=1),
                               available_at=start + pd.Timedelta(hours=2), team_A_win=1)
        self.assertEqual(first.expected_A, second.expected_A)
        elo.advance(start + pd.Timedelta(hours=3))
        self.assertAlmostEqual(elo.rating("A", start + pd.Timedelta(hours=3)), 1512, places=6)
        elo.advance(start + pd.Timedelta(hours=5))
        self.assertAlmostEqual(elo.rating("A", start + pd.Timedelta(hours=5)), 1524, places=6)

    def test_time_decay_halves_distance_to_prior(self):
        elo = EloSystem(FeatureConfig(elo_half_life_days=30))
        start = pd.Timestamp("2024-01-01T00:00:00Z")
        available = start + pd.Timedelta(hours=1)
        elo.queue_game(team_A="A", team_B="B", start_time=start, available_at=available, team_A_win=1)
        later = start + pd.Timedelta(days=30)
        elo.advance(later)
        self.assertAlmostEqual(elo.rating("A", later), 1506)
        self.assertAlmostEqual(elo.rating("B", later), 1494)

    def test_delayed_result_update_is_aged_from_game_start(self):
        elo = EloSystem(FeatureConfig(elo_half_life_days=30))
        start = pd.Timestamp("2024-01-01T00:00:00Z")
        available = start + pd.Timedelta(days=30)
        elo.queue_game(team_A="A", team_B="B", start_time=start, available_at=available, team_A_win=1)
        elo.advance(available)
        self.assertEqual(elo.rating("A", available), 1500)
        later = start + pd.Timedelta(days=60)
        elo.advance(later)
        self.assertAlmostEqual(elo.rating("A", later), 1503)

    def test_delayed_old_roster_result_cannot_replace_newer_known_roster(self):
        elo = EloSystem(FeatureConfig(roster_retention=0.5))
        start = pd.Timestamp("2024-01-01T00:00:00Z")
        elo.queue_game(team_A="A", team_B="B", start_time=start,
                       available_at=start + pd.Timedelta(days=10), team_A_win=1, roster_id_A="old")
        elo.queue_game(team_A="A", team_B="B", start_time=start + pd.Timedelta(days=1),
                       available_at=start + pd.Timedelta(days=2), team_A_win=1, roster_id_A="new")
        later = start + pd.Timedelta(days=11)
        elo.advance(later)
        self.assertAlmostEqual(elo.rating("A", later, "new"), elo.rating("A", later))
        self.assertLess(elo.rating("A", later, "old"), elo.rating("A", later, "new"))

    def test_known_roster_change_shrinks_but_unknown_does_not(self):
        elo = EloSystem(FeatureConfig(roster_retention=0.5, elo_half_life_days=1e9))
        start = pd.Timestamp("2024-01-01T00:00:00Z")
        elo.queue_game(team_A="A", team_B="B", start_time=start,
                       available_at=start + pd.Timedelta(hours=1), team_A_win=1, roster_id_A="old")
        later = start + pd.Timedelta(hours=2)
        elo.advance(later)
        self.assertAlmostEqual(elo.rating("A", later, "old"), 1512, places=6)
        self.assertAlmostEqual(elo.rating("A", later, "unknown"), 1512, places=6)
        self.assertAlmostEqual(elo.rating("A", later, "new"), 1506, places=6)
        # Reading a proposed roster cannot change the stored rating.
        self.assertAlmostEqual(elo.rating("A", later, "old"), 1512, places=6)

    def test_invalid_timing_and_backwards_replay_fail(self):
        elo = EloSystem()
        with self.assertRaises(ValueError):
            elo.queue_game(team_A="A", team_B="B", start_time="2024-01-01T00:00:00Z", available_at="2024-01-01T00:00:00Z", team_A_win=1)
        elo.advance("2024-02-01T00:00:00Z")
        with self.assertRaises(ValueError):
            elo.advance("2024-01-01T00:00:00Z")

    def test_naive_timestamps_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "timezone"):
            EloSystem().advance("2024-01-01")


if __name__ == "__main__":
    unittest.main()
