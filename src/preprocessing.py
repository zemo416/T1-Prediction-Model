"""Game-level chronological partitions with result-availability purging.

Splitting precedes orientation augmentation. Games sharing a start timestamp
always stay together, and a label must be available strictly before the next
partition begins to be used for fitting.
"""

from dataclasses import dataclass
from collections.abc import Iterator

import pandas as pd

from .data_loader import utc_timestamp


@dataclass
class ChronologicalSplit:
    """Disjoint train, calibration/validation and held-out test games."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    purged_train: int = 0
    purged_validation: int = 0

    def summary(self) -> dict:
        """Return JSON-serializable partition boundaries and purge counts."""
        result = {}
        for name in ("train", "validation", "test"):
            frame = getattr(self, name)
            result[name] = {
                "games": len(frame),
                "start": frame["start_time"].min().isoformat(),
                "end": frame["start_time"].max().isoformat(),
                "last_result_available": frame["available_at"].max().isoformat(),
            }
        result["purged_train"] = self.purged_train
        result["purged_validation"] = self.purged_validation
        return result


def _ordered_games(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"game_id", "start_time", "available_at"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing split metadata: {sorted(missing)}")
    if frame["game_id"].isna().any() or frame["game_id"].duplicated().any():
        raise ValueError("Split at game level: game_id must be unique and nonmissing.")
    ordered = frame.copy()
    for name in ("start_time", "available_at"):
        ordered[name] = pd.to_datetime(ordered[name].map(lambda value: utc_timestamp(value, name)), utc=True)
        if ordered[name].isna().any():
            raise ValueError(f"{name} cannot be missing.")
    if (ordered["available_at"] <= ordered["start_time"]).any():
        raise ValueError("Result available_at must be strictly after game start_time.")
    return ordered.sort_values("start_time", kind="stable").reset_index(drop=True)


def _purged_partition(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame,
) -> ChronologicalSplit:
    if any(part.empty for part in (train, validation, test)):
        raise ValueError("Train, validation and test must each contain games.")
    train_mask = train["available_at"] < validation["start_time"].min()
    validation_mask = validation["available_at"] < test["start_time"].min()
    split = ChronologicalSplit(
        train=train.loc[train_mask].copy(),
        validation=validation.loc[validation_mask].copy(),
        test=test.copy(),
        purged_train=int((~train_mask).sum()),
        purged_validation=int((~validation_mask).sum()),
    )
    if split.train.empty or split.validation.empty:
        raise ValueError("Result-availability purging emptied a fitting partition.")
    return split


def _utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    return utc_timestamp(value, "date boundary")


def chronological_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    train_end: str | pd.Timestamp | None = None,
    validation_end: str | pd.Timestamp | None = None,
) -> ChronologicalSplit:
    """Split ordered unique start times, or use exclusive UTC date cutoffs.

With explicit boundaries, train starts precede ``train_end``, calibration
starts fall in ``[train_end, validation_end)``, and test starts fall on/after
``validation_end``. Cutoffs must include an explicit timezone.
    """
    ordered = _ordered_games(frame)
    if (train_end is None) != (validation_end is None):
        raise ValueError("Provide both train_end and validation_end, or neither.")
    if train_end is not None:
        validation_start = _utc_timestamp(train_end)
        test_start = _utc_timestamp(validation_end)
        if validation_start >= test_start:
            raise ValueError("train_end must precede validation_end.")
    else:
        if not (0 < train_fraction < 1 and 0 < validation_fraction < 1
                and train_fraction + validation_fraction < 1):
            raise ValueError("Split fractions must be positive and sum to less than 1.")
        starts = ordered["start_time"].drop_duplicates().tolist()
        if len(starts) < 3:
            raise ValueError("At least three distinct game start times are required.")
        train_count = max(1, min(int(len(starts) * train_fraction), len(starts) - 2))
        validation_count = max(
            1, min(int(len(starts) * validation_fraction), len(starts) - train_count - 1),
        )
        validation_start = starts[train_count]
        test_start = starts[train_count + validation_count]
    times = ordered["start_time"]
    return _purged_partition(
        ordered.loc[times < validation_start],
        ordered.loc[(times >= validation_start) & (times < test_start)],
        ordered.loc[times >= test_start],
    )


def walk_forward_splits(
    frame: pd.DataFrame,
    n_splits: int = 3,
    min_train_fraction: float = 0.4,
    calibration_fraction: float = 0.2,
) -> Iterator[ChronologicalSplit]:
    """Yield expanding training, later calibration and nonoverlapping test blocks.

Fractions refer to unique start timestamps across the full supplied history.
Calibration length is fixed per fold. Each subsequent training block expands
into the preceding history; each test game is scored only once.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be positive.")
    if not (0 < min_train_fraction < 1 and 0 < calibration_fraction < 1
            and min_train_fraction + calibration_fraction < 1):
        raise ValueError("Training/calibration fractions must be positive and sum to < 1.")
    ordered = _ordered_games(frame)
    starts = ordered["start_time"].drop_duplicates().tolist()
    count = len(starts)
    train_count = max(1, int(count * min_train_fraction))
    calibration_count = max(1, int(count * calibration_fraction))
    first_test = train_count + calibration_count
    remaining = count - first_test
    if remaining < n_splits:
        raise ValueError("Not enough distinct game times for the requested backtest folds.")
    quotient, remainder = divmod(remaining, n_splits)
    test_index = first_test
    times = ordered["start_time"]
    for fold in range(n_splits):
        test_count = quotient + int(fold < remainder)
        stop = test_index + test_count
        validation_start = starts[test_index - calibration_count]
        test_start = starts[test_index]
        test_mask = times >= test_start
        if stop < count:
            test_mask &= times < starts[stop]
        yield _purged_partition(
            ordered.loc[times < validation_start],
            ordered.loc[(times >= validation_start) & (times < test_start)],
            ordered.loc[test_mask],
        )
        test_index = stop
