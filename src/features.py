"""Availability-safe pre-draft features, in a team-A minus team-B orientation.

All directional numeric features end in ``_diff``. The invariant numeric columns
are ``best_of``, ``history_total_games``, ``history_min_games`` and ``h2h_games``;
patch, tournament and stage are invariant categorical context. Team names and
outcomes are metadata only. Missing historical statistics stay missing.

Roster IDs enable coarse Elo shrinkage and continuity features, not estimates of
individual player strength. An optional provider can add genuine player features
using the already cutoff-filtered history. Draft features are intentionally absent.
"""

from __future__ import annotations

from bisect import insort_right
from dataclasses import dataclass
import heapq
from typing import Mapping, Protocol

import numpy as np
import pandas as pd

from src.config import CATEGORICAL_FEATURES, HISTORICAL_STATS, METADATA_COLUMNS, FeatureConfig
from src.data_loader import utc_timestamp
from src.elo import EloSystem, _known, _utc


class PlayerFeatureProvider(Protocol):
    """Extension point: return numeric, antisymmetric ``*_diff`` player features.

    ``history`` contains only results available before the cutoff. Providers must
    additionally enforce as-of availability on any external roster/player data.
    No provider or fabricated player statistics are supplied by this project.
    """

    def for_match(self, history: pd.DataFrame, *, context: Mapping[str, object]) -> Mapping[str, float]: ...


def mirror_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Flip directional features only; labels/metadata/context remain untouched."""
    mirrored = frame.copy()
    for column in mirrored.columns:
        if column.endswith("_diff"):
            mirrored[column] = -pd.to_numeric(mirrored[column], errors="raise")
    return mirrored


def _mean(values: list[float], default: float = np.nan) -> float:
    finite = [float(value) for value in values if pd.notna(value) and np.isfinite(value)]
    return float(np.mean(finite)) if finite else default


@dataclass
class _Observation:
    start: pd.Timestamp
    game_id: str
    opponent: str
    win: int
    side: str
    patch: str
    roster: str
    expected: float
    opponent_elo: float
    stats: dict[str, float]

    @property
    def sort_key(self) -> tuple[int, str]:
        return self.start.value, self.game_id


class _History:
    def __init__(self, config: FeatureConfig) -> None:
        self.config = config
        self.elo = EloSystem(config)
        self.teams: dict[str, list[_Observation]] = {}
        self.pending: list[tuple[int, int, dict]] = []
        self.counter = 0
        self.visible: list[dict] = []

    def advance(self, at: pd.Timestamp) -> None:
        self.elo.advance(at)
        while self.pending and self.pending[0][0] < at.value:
            _, _, game = heapq.heappop(self.pending)
            self.visible.append(game)
            snapshot = game["_elo_snapshot"]
            for suffix, other in (("A", "B"), ("B", "A")):
                side = game["side_A"]
                if suffix == "B":
                    side = {"blue": "red", "red": "blue"}.get(side, "unknown")
                observation = _Observation(
                    start=game["start_time"], game_id=str(game["game_id"]),
                    opponent=game[f"team_{other}"],
                    win=int(game["team_A_win"]) if suffix == "A" else 1 - int(game["team_A_win"]),
                    side=side, patch=game["patch"], roster=game[f"roster_id_{suffix}"],
                    expected=snapshot.expected_A if suffix == "A" else 1 - snapshot.expected_A,
                    opponent_elo=snapshot.rating_B if suffix == "A" else snapshot.rating_A,
                    stats={stat: game[f"{stat}_{suffix}"] for stat in HISTORICAL_STATS},
                )
                history = self.teams.setdefault(game[f"team_{suffix}"], [])
                insort_right(history, observation, key=lambda row: row.sort_key)

    def add(self, game: dict) -> None:
        snapshot = self.elo.queue_game(**{name: game[name] for name in (
            "team_A", "team_B", "start_time", "available_at", "team_A_win", "roster_id_A", "roster_id_B"
        )})
        event = dict(game, _elo_snapshot=snapshot)
        heapq.heappush(self.pending, (game["available_at"].value, self.counter, event))
        self.counter += 1


class FeatureBuilder:
    """Replay immutable game history and snapshot features before each game."""

    def __init__(
        self, config: FeatureConfig | None = None,
        player_provider: PlayerFeatureProvider | None = None,
    ) -> None:
        self.config = config or FeatureConfig()
        if min(self.config.rolling_games, self.config.recent_days, self.config.h2h_games) < 1:
            raise ValueError("Rolling windows must be positive.")
        if self.config.form_half_life_days <= 0:
            raise ValueError("Form half-life must be positive.")
        self.player_provider = player_provider

    @staticmethod
    def _prepare(games: pd.DataFrame) -> pd.DataFrame:
        frame = games.copy()
        required = ("game_id", "start_time", "available_at", "team_A", "team_B", "team_A_win")
        if frame.empty:
            for column in required:
                if column not in frame:
                    frame[column] = pd.Series(dtype="object")
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing normalized game columns: {sorted(missing)}")
        for column in ("start_time", "available_at"):
            frame[column] = pd.to_datetime([utc_timestamp(value, column) for value in frame[column]], utc=True)
        if frame[list(required)].isna().any().any():
            raise ValueError("Required game metadata cannot be missing.")
        if frame["game_id"].duplicated().any():
            raise ValueError("Game IDs must be unique.")
        if (frame["available_at"] <= frame["start_time"]).any():
            raise ValueError("Each result must become available strictly after game start.")
        if not frame["team_A_win"].isin([0, 1]).all() or (frame["team_A"] == frame["team_B"]).any():
            raise ValueError("Games require distinct teams and binary outcomes.")
        for column in (*CATEGORICAL_FEATURES, "side_A", "roster_id_A", "roster_id_B"):
            if column not in frame:
                frame[column] = "unknown"
            frame[column] = frame[column].fillna("unknown").astype(str)
        if not frame["side_A"].isin(["blue", "red", "unknown"]).all():
            raise ValueError("side_A must be blue, red, or unknown.")
        if "best_of" not in frame:
            frame["best_of"] = np.nan
        if "is_synthetic" not in frame:
            frame["is_synthetic"] = False
        for stat in HISTORICAL_STATS:
            for suffix in ("A", "B"):
                name = f"{stat}_{suffix}"
                frame[name] = pd.to_numeric(frame[name], errors="raise") if name in frame else np.nan
        return frame.sort_values(["start_time", "game_id"], kind="stable").reset_index(drop=True)

    def _team_features(
        self, observations: list[_Observation], *, at: pd.Timestamp,
        current_side: str, patch: str, roster: str,
    ) -> dict[str, float]:
        config = self.config
        rolling = observations[-config.rolling_games:]
        recent = [row for row in observations if row.start >= at - pd.Timedelta(days=config.recent_days)]
        values: dict[str, float] = {
            "history_games": float(len(observations)),
            f"last{config.rolling_games}_games": float(len(rolling)),
            f"recent{config.recent_days}d_games": float(len(recent)),
            f"recent{config.recent_days}d_win_rate": _mean([row.win for row in recent], 0.5),
            "opponent_adjusted_form": _mean([row.win - row.expected for row in rolling], 0.0),
            "historical_opponent_elo": _mean([row.opponent_elo for row in rolling], config.elo_initial),
            "days_since_last_game": (at - observations[-1].start).total_seconds() / 86400 if observations else np.nan,
            "early_lead_conversion": _mean([row.win for row in rolling if pd.notna(row.stats["gold_diff_15"]) and row.stats["gold_diff_15"] > 0]),
        }
        for window in dict.fromkeys((5, 10, config.rolling_games)):
            rows = observations[-window:]
            values[f"last{window}_win_rate"] = _mean([row.win for row in rows], 0.5)
            for stat in HISTORICAL_STATS:
                values[f"last{window}_{stat}"] = _mean([row.stats[stat] for row in rows])
        for side in ("blue", "red"):
            rows = [row for row in rolling if row.side == side]
            values[f"{side}_win_rate"] = _mean([row.win for row in rows], 0.5)
        values["current_side_win_rate"] = (
            values[f"{current_side}_win_rate"] if current_side in ("blue", "red") else np.nan
        )
        same_patch = [row for row in rolling if patch != "unknown" and row.patch == patch]
        values["same_patch_win_rate"] = _mean([row.win for row in same_patch], 0.5) if patch != "unknown" else np.nan
        values["same_patch_games"] = float(len(same_patch))
        known_rosters = [row for row in rolling if _known(row.roster)]
        values["roster_continuity"] = (
            _mean([float(row.roster == roster) for row in known_rosters]) if _known(roster) else np.nan
        )
        values["current_roster_win_rate"] = (
            _mean([row.win for row in observations if row.roster == roster], 0.5) if _known(roster) else np.nan
        )
        if observations:
            ages = np.array([(at - row.start).total_seconds() / 86400 for row in observations])
            # Subtract the youngest age to avoid underflow after long inactivity.
            weights = np.exp2(-(ages - ages.min()) / config.form_half_life_days)
            values["exp_form_win_rate"] = float(np.average([row.win for row in observations], weights=weights))
        else:
            values["exp_form_win_rate"] = 0.5
        return values

    def _snapshot(self, history: _History, context: Mapping[str, object]) -> dict:
        team_a, team_b, at = context["team_A"], context["team_B"], context["start_time"]
        side_a = context["side_A"]
        side_b = {"blue": "red", "red": "blue"}.get(side_a, "unknown")
        a_rows, b_rows = history.teams.get(team_a, []), history.teams.get(team_b, [])
        a_values = self._team_features(a_rows, at=at, current_side=side_a, patch=context["patch"], roster=context["roster_id_A"])
        b_values = self._team_features(b_rows, at=at, current_side=side_b, patch=context["patch"], roster=context["roster_id_B"])
        result = {name: context[name] for name in CATEGORICAL_FEATURES}
        result.update({f"{name}_diff": value - b_values[name] for name, value in a_values.items()})
        snapshot = history.elo.snapshot(team_a, team_b, at, context["roster_id_A"], context["roster_id_B"])
        h2h = [row for row in a_rows if row.opponent == team_b]
        recent_h2h = h2h[-self.config.h2h_games:]
        result.update(
            elo_rating_diff=snapshot.rating_A - snapshot.rating_B,
            current_side_diff=float({"blue": 1, "red": -1}.get(side_a, 0)),
            h2h_win_rate_diff=2 * _mean([row.win for row in h2h], 0.5) - 1,
            recent_h2h_win_rate_diff=2 * _mean([row.win for row in recent_h2h], 0.5) - 1,
            h2h_elo_residual_diff=2 * _mean([row.win - row.expected for row in recent_h2h], 0.0),
            best_of=float(context["best_of"]) if pd.notna(context["best_of"]) else np.nan,
            history_total_games=float(len(a_rows) + len(b_rows)),
            history_min_games=float(min(len(a_rows), len(b_rows))),
            h2h_games=float(len(h2h)),
        )
        if self.player_provider is not None:
            visible = pd.DataFrame([{key: value for key, value in row.items() if not key.startswith("_")} for row in history.visible])
            prematch_context = {name: context[name] for name in (
                "team_A", "team_B", "start_time", "side_A", "patch", "tournament", "stage",
                "best_of", "roster_id_A", "roster_id_B",
            )}
            extra = dict(self.player_provider.for_match(visible.copy(), context=prematch_context))
            if any(not name.endswith("_diff") or name in result for name in extra):
                raise ValueError("Player features require unique names ending in _diff.")
            result.update({name: float(value) for name, value in extra.items()})
        return result

    def build_dataset(self, games: pd.DataFrame) -> pd.DataFrame:
        """Build training rows, excluding current, tied and unavailable outcomes."""
        frame = self._prepare(games)
        history = _History(self.config)
        results = []
        for game in frame.to_dict("records"):
            history.advance(game["start_time"])
            row = {name: game[name] for name in METADATA_COLUMNS}
            row.update(self._snapshot(history, game))
            results.append(row)
            history.add(game)
        if not results:
            columns = list(METADATA_COLUMNS) + list(self.for_match(
                frame, team_A="__new_a__", team_B="__new_b__", start_time="2000-01-01T00:00:00Z"
            ).columns)
            return pd.DataFrame(columns=list(dict.fromkeys(columns)))
        return pd.DataFrame(results)

    def for_match(
        self, games: pd.DataFrame, *, team_A: str, team_B: str, start_time: object,
        side_A: str = "unknown", patch: str = "unknown", tournament: str = "unknown",
        stage: str = "unknown", best_of: int | None = None,
        roster_id_A: str = "unknown", roster_id_B: str = "unknown",
    ) -> pd.DataFrame:
        """Build one pre-draft feature row for an explicit UTC prediction cutoff."""
        if not team_A or not team_B or team_A == team_B:
            raise ValueError("Prediction requires two distinct, nonempty teams.")
        if side_A not in ("blue", "red", "unknown"):
            raise ValueError("side_A must be blue, red, or unknown.")
        if best_of is not None and pd.notna(best_of) and best_of not in (1, 3, 5):
            raise ValueError("best_of must be 1, 3, 5, or omitted.")
        at = _utc(start_time)
        frame = self._prepare(games)
        history = _History(self.config)
        for game in frame.loc[frame["start_time"] < at].to_dict("records"):
            history.advance(game["start_time"])
            history.add(game)
        history.advance(at)
        context = dict(
            team_A=team_A, team_B=team_B, start_time=at, side_A=side_A,
            patch=patch, tournament=tournament, stage=stage, best_of=best_of,
            roster_id_A=roster_id_A, roster_id_B=roster_id_B,
        )
        return pd.DataFrame([self._snapshot(history, context)])
