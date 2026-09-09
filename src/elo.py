"""Time-decayed Elo with result-availability events and frozen expectations.

A game's update is calculated from its pre-start ratings, but only applied once
its result becomes available. An overlapping game therefore cannot see that
result. Updates age from game start, so delayed publication cannot refresh an
old game's influence. Roster identifiers provide coarse regression toward the population mean;
they are not player ratings and unknown identifiers never imply continuity.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

import pandas as pd

from src.config import FeatureConfig
from src.data_loader import utc_timestamp


def expected_score(rating_a: float, rating_b: float) -> float:
    """Return the standard base-10 Elo win expectation (400-point scale)."""
    exponent = max(-300.0, min(300.0, (rating_b - rating_a) / 400.0))
    return 1.0 / (1.0 + 10.0**exponent)


def _utc(value: object) -> pd.Timestamp:
    return utc_timestamp(value, "Elo timestamp")


def _known(roster: object) -> bool:
    return roster is not None and not pd.isna(roster) and str(roster).strip().lower() not in {"", "unknown"}


@dataclass(frozen=True)
class EloSnapshot:
    rating_A: float
    rating_B: float
    expected_A: float


@dataclass
class _RatingState:
    rating: float
    updated_at: pd.Timestamp
    roster_id: str = "unknown"
    roster_start: pd.Timestamp | None = None


class EloSystem:
    """Replay games in start-time order, releasing results strictly before queries.

    Call ``advance(at)`` before asking for ratings. ``queue_game`` also advances
    to the game's start and returns its frozen pre-match snapshot. Queries and
    queued game starts must be nondecreasing; equal starts are safe.
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()
        if self.config.elo_half_life_days <= 0 or self.config.elo_k <= 0:
            raise ValueError("Elo half-life and K must be positive.")
        if not 0 <= self.config.roster_retention <= 1:
            raise ValueError("Roster retention must be between zero and one.")
        self._ratings: dict[str, _RatingState] = {}
        self._pending: list[tuple[int, int, dict]] = []
        self._counter = 0
        self._time: pd.Timestamp | None = None

    def _decayed(self, state: _RatingState, at: pd.Timestamp) -> float:
        days = max(0.0, (at - state.updated_at).total_seconds() / 86400.0)
        return self.config.elo_initial + (state.rating - self.config.elo_initial) * math.exp2(
            -days / self.config.elo_half_life_days
        )

    def rating(self, team: str, at: object, roster_id: str = "unknown") -> float:
        """Read a decayed rating, optionally shrinking for a known new roster."""
        stamp = _utc(at)
        if self._time is not None and stamp < self._time:
            raise ValueError("Elo cannot query backwards in time; replay history instead.")
        state = self._ratings.get(team)
        if state is None:
            return self.config.elo_initial
        value = self._decayed(state, stamp)
        if _known(roster_id) and _known(state.roster_id) and roster_id != state.roster_id:
            value = self.config.elo_initial + (value - self.config.elo_initial) * self.config.roster_retention
        return value

    def snapshot(
        self, team_A: str, team_B: str, at: object,
        roster_id_A: str = "unknown", roster_id_B: str = "unknown",
    ) -> EloSnapshot:
        """Read both ratings without mutating state or observing queued results."""
        rating_a = self.rating(team_A, at, roster_id_A)
        rating_b = self.rating(team_B, at, roster_id_B)
        return EloSnapshot(rating_a, rating_b, expected_score(rating_a, rating_b))

    def advance(self, at: object) -> None:
        """Apply pending outcomes with availability strictly earlier than ``at``."""
        stamp = _utc(at)
        if self._time is not None and stamp < self._time:
            raise ValueError("Elo games must be replayed chronologically.")
        while self._pending and self._pending[0][0] < stamp.value:
            _, _, event = heapq.heappop(self._pending)
            self._apply(event)
        self._time = stamp

    def queue_game(
        self, *, team_A: str, team_B: str, start_time: object,
        available_at: object, team_A_win: int,
        roster_id_A: str = "unknown", roster_id_B: str = "unknown",
    ) -> EloSnapshot:
        """Schedule one observed game; return its ratings before its own outcome."""
        start, available = _utc(start_time), _utc(available_at)
        if available <= start:
            raise ValueError("Result availability must be strictly after game start.")
        if team_A == team_B or team_A_win not in (0, 1):
            raise ValueError("Elo requires distinct teams and a binary result.")
        self.advance(start)
        snapshot = self.snapshot(team_A, team_B, start, roster_id_A, roster_id_B)
        event = dict(
            team_A=team_A, team_B=team_B, start_time=start, available_at=available,
            roster_id_A=roster_id_A, roster_id_B=roster_id_B,
            delta=self.config.elo_k * (team_A_win - snapshot.expected_A),
        )
        heapq.heappush(self._pending, (available.value, self._counter, event))
        self._counter += 1
        return snapshot

    def _apply(self, event: dict) -> None:
        age_days = (event["available_at"] - event["start_time"]).total_seconds() / 86400.0
        aged_delta = event["delta"] * math.exp2(-age_days / self.config.elo_half_life_days)
        for suffix, sign in (("A", 1), ("B", -1)):
            team, roster = event[f"team_{suffix}"], event[f"roster_id_{suffix}"]
            state = self._ratings.get(team)
            value = self.config.elo_initial if state is None else self._decayed(state, event["available_at"])
            # A delayed old result must not switch a newer known roster back.
            newer_roster = _known(roster) and (
                state is None or state.roster_start is None or event["start_time"] >= state.roster_start
            )
            if newer_roster and state is not None and _known(state.roster_id) and roster != state.roster_id:
                value = self.config.elo_initial + (value - self.config.elo_initial) * self.config.roster_retention
            self._ratings[team] = _RatingState(
                rating=value + sign * aged_delta, updated_at=event["available_at"],
                roster_id=str(roster) if newer_roster else (state.roster_id if state else "unknown"),
                roster_start=event["start_time"] if newer_roster else (state.roster_start if state else None),
            )
