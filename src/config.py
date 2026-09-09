"""Serializable configuration, kept separate from model logic."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureConfig:
    rolling_games: int = 20
    recent_days: int = 90
    form_half_life_days: float = 30.0
    elo_initial: float = 1500.0
    elo_k: float = 24.0
    elo_half_life_days: float = 365.0
    roster_retention: float = 0.75
    h2h_games: int = 10


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    calibration: str = "sigmoid"
    n_estimators: int = 150
    max_depth: int = 3
    learning_rate: float = 0.05


HISTORICAL_STATS = (
    "gold_diff_10", "gold_diff_15", "xp_diff_10", "xp_diff_15",
    "cs_diff_10", "cs_diff_15", "first_blood", "first_tower",
    "dragon_control", "herald_control", "baron_control",
    "first_dragon", "first_herald", "first_baron",
    "duration_seconds", "avg_gold_diff",
)

CATEGORICAL_FEATURES = ("patch", "tournament", "stage")
METADATA_COLUMNS = (
    "game_id", "start_time", "available_at", "team_A", "team_B",
    "team_A_win", "is_synthetic",
)
