"""Leakage checks use altered outcomes/stats, availability ties, and overlap."""

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.config import METADATA_COLUMNS, FeatureConfig
from src.features import FeatureBuilder, mirror_features


def games_fixture() -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2024-01-01T12:00:00Z")
    for index in range(12):
        at = start + pd.Timedelta(days=index)
        rows.append(dict(
            game_id=f"game-{index:02}", start_time=at, available_at=at + pd.Timedelta(hours=1),
            team_A="Alpha", team_B="Beta", team_A_win=int(index % 3 != 0),
            is_synthetic=True, side_A="blue" if index % 2 == 0 else "red",
            patch="14.1", tournament="Fixture League", stage="regular", best_of=3,
            roster_id_A="alpha-v1", roster_id_B="beta-v1",
            gold_diff_15_A=100.0 * index, gold_diff_15_B=-100.0 * index,
        ))
    return pd.DataFrame(rows)


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.builder = FeatureBuilder()
        self.games = games_fixture()

    def test_current_and_future_results_cannot_affect_current_features(self):
        baseline = self.builder.build_dataset(self.games)
        changed = self.games.copy()
        changed.loc[6:, "team_A_win"] = 1 - changed.loc[6:, "team_A_win"]
        changed.loc[6:, "gold_diff_15_A"] = 999999.0
        changed.loc[6:, "gold_diff_15_B"] = -999999.0
        other = self.builder.build_dataset(changed)
        feature_columns = [name for name in baseline if name not in METADATA_COLUMNS]
        assert_frame_equal(baseline.loc[:6, feature_columns], other.loc[:6, feature_columns])
        self.assertNotEqual(baseline.loc[7, "last5_gold_diff_15_diff"], other.loc[7, "last5_gold_diff_15_diff"])

    def test_rolling_windows_shift_out_current_game(self):
        features = self.builder.build_dataset(self.games)
        self.assertEqual(features.loc[0, "last5_win_rate_diff"], 0)
        self.assertTrue(np.isnan(features.loc[0, "last5_gold_diff_15_diff"]))
        expected = 2 * self.games.loc[1:5, "team_A_win"].mean() - 1
        self.assertAlmostEqual(features.loc[6, "last5_win_rate_diff"], expected)
        self.assertAlmostEqual(features.loc[6, "last5_gold_diff_15_diff"], 600.0)

    def test_simultaneous_and_unfinished_games_are_excluded(self):
        games = self.games.iloc[:4].copy()
        start = games.loc[0, "start_time"]
        games.loc[0, "available_at"] = start + pd.Timedelta(days=2)
        games.loc[1, "start_time"] = start
        games.loc[1, "available_at"] = start + pd.Timedelta(hours=2)
        features = self.builder.build_dataset(games)
        self.assertEqual(features.loc[0, "history_total_games"], 0)
        self.assertEqual(features.loc[1, "history_total_games"], 0)
        # At day 2, game 0 is tied on availability and only game 1 is usable.
        self.assertEqual(features.loc[2, "history_total_games"], 2)
        self.assertEqual(features.loc[2, "last5_win_rate_diff"], 1)
        changed = games.copy()
        changed.loc[0, "team_A_win"] = 1 - changed.loc[0, "team_A_win"]
        changed.loc[0, "gold_diff_15_A"] = 1000000
        other = self.builder.build_dataset(changed)
        columns = [name for name in features if name not in METADATA_COLUMNS]
        assert_frame_equal(features.loc[:2, columns], other.loc[:2, columns])

    def test_delayed_old_result_does_not_become_latest_rolling_game(self):
        games = self.games.copy()
        games.loc[0, "available_at"] = games.loc[9, "start_time"] + pd.Timedelta(hours=2)
        games.loc[0, "gold_diff_15_A"] = 9999999.0
        features = self.builder.build_dataset(games)
        self.assertAlmostEqual(features.loc[10, "last5_gold_diff_15_diff"], 1400.0)

    def test_for_match_matches_historical_dataset_snapshot(self):
        historical = self.builder.build_dataset(self.games)
        row = self.games.iloc[7]
        future = self.builder.for_match(self.games, **row[[
            "team_A", "team_B", "start_time", "side_A", "patch", "tournament", "stage", "best_of", "roster_id_A", "roster_id_B"
        ]].to_dict())
        assert_frame_equal(historical.loc[[7], future.columns].reset_index(drop=True), future)

    def test_swap_orientation_negates_every_directional_feature(self):
        at = self.games["start_time"].max() + pd.Timedelta(days=1)
        forward = self.builder.for_match(self.games, team_A="Alpha", team_B="Beta", start_time=at,
                                         side_A="blue", patch="14.1", best_of=3, roster_id_A="alpha-v1", roster_id_B="beta-v1")
        reverse = self.builder.for_match(self.games, team_A="Beta", team_B="Alpha", start_time=at,
                                         side_A="red", patch="14.1", best_of=3, roster_id_A="beta-v1", roster_id_B="alpha-v1")
        assert_frame_equal(mirror_features(forward), reverse, atol=1e-12, rtol=1e-12)
        assert_frame_equal(mirror_features(mirror_features(forward)), forward)

    def test_missing_statistics_are_not_fabricated(self):
        features = self.builder.build_dataset(self.games)
        self.assertTrue(features["last20_dragon_control_diff"].isna().all())
        cold = self.builder.for_match(pd.DataFrame(), team_A="New A", team_B="New B", start_time="2024-01-01T00:00:00Z")
        self.assertEqual(cold.loc[0, "elo_rating_diff"], 0)
        self.assertEqual(cold.loc[0, "last5_win_rate_diff"], 0)
        self.assertEqual(cold.loc[0, "history_total_games"], 0)

    def test_player_extension_receives_only_safe_history_and_context(self):
        test = self

        class Provider:
            calls = 0

            def for_match(self, history, *, context):
                test.assertNotIn("team_A_win", context)
                test.assertNotIn("gold_diff_15_A", context)
                if len(history):
                    test.assertTrue((history["available_at"] < context["start_time"]).all())
                    test.assertTrue((history["start_time"] < context["start_time"]).all())
                self.calls += 1
                return {"player_example_diff": np.nan}

        provider = Provider()
        features = FeatureBuilder(player_provider=provider).build_dataset(self.games)
        self.assertEqual(provider.calls, len(self.games))
        self.assertTrue(features["player_example_diff"].isna().all())

    def test_time_window_and_configurable_roster_regression(self):
        at = self.games["start_time"].max() + pd.Timedelta(days=91)
        row = self.builder.for_match(self.games, team_A="Alpha", team_B="Beta", start_time=at)
        self.assertEqual(row.loc[0, "recent90d_win_rate_diff"], 0)
        retained = FeatureBuilder(FeatureConfig(roster_retention=1))
        reset = FeatureBuilder(FeatureConfig(roster_retention=0))
        args = dict(team_A="Alpha", team_B="Beta", start_time=at, roster_id_A="new-a", roster_id_B="new-b")
        self.assertNotEqual(retained.for_match(self.games, **args).loc[0, "elo_rating_diff"], 0)
        self.assertEqual(reset.for_match(self.games, **args).loc[0, "elo_rating_diff"], 0)

    def test_historical_team_order_does_not_change_future_features(self):
        at = self.games["start_time"].max() + pd.Timedelta(days=1)
        baseline = self.builder.for_match(self.games, team_A="Alpha", team_B="Beta", start_time=at)
        reoriented = self.games.copy()
        selected = reoriented.index % 2 == 0
        for first, second in (("team_A", "team_B"), ("roster_id_A", "roster_id_B"), ("gold_diff_15_A", "gold_diff_15_B")):
            reoriented.loc[selected, [first, second]] = reoriented.loc[selected, [second, first]].to_numpy()
        reoriented.loc[selected, "team_A_win"] = 1 - reoriented.loc[selected, "team_A_win"]
        reoriented.loc[selected, "side_A"] = reoriented.loc[selected, "side_A"].map({"blue": "red", "red": "blue"})
        other = self.builder.for_match(reoriented, team_A="Alpha", team_B="Beta", start_time=at)
        assert_frame_equal(baseline, other, atol=1e-12, rtol=1e-12)

    def test_for_match_excludes_unavailable_and_future_rows(self):
        games = self.games.copy()
        cutoff = games.loc[5, "start_time"]
        games.loc[0, "available_at"] = cutoff + pd.Timedelta(days=20)
        full = self.builder.for_match(games, team_A="Alpha", team_B="Beta", start_time=cutoff)
        usable = games.loc[(games.start_time < cutoff) & (games.available_at < cutoff)]
        truncated = self.builder.for_match(usable, team_A="Alpha", team_B="Beta", start_time=cutoff)
        assert_frame_equal(full, truncated)

    def test_all_public_cutoffs_require_explicit_timezone(self):
        with self.assertRaisesRegex(ValueError, "timezone"):
            self.builder.for_match(self.games, team_A="Alpha", team_B="Beta", start_time="2024-02-01")
        for column in ("start_time", "available_at"):
            games = self.games.copy()
            games[column] = games[column].dt.tz_localize(None)
            with self.assertRaisesRegex(ValueError, "timezone"):
                self.builder.build_dataset(games)
        utc = self.builder.for_match(self.games, team_A="Alpha", team_B="Beta", start_time="2024-02-01T12:00:00Z")
        offset = self.builder.for_match(self.games, team_A="Alpha", team_B="Beta", start_time="2024-02-01T06:00:00-06:00")
        assert_frame_equal(utc, offset)

    def test_building_features_does_not_modify_input(self):
        original = self.games.copy(deep=True)
        self.builder.build_dataset(self.games)
        assert_frame_equal(self.games, original)


if __name__ == "__main__":
    unittest.main()
