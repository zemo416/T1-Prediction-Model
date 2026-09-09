"""Read immutable, canonical two-team CSV rows and validate their provenance.

Source-specific adapters should produce this schema in a separate directory;
native vendor exports are deliberately not guessed or silently coerced.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import HISTORICAL_STATS

LOGGER = logging.getLogger(__name__)
REQUIRED_COLUMNS = (
    "game_id", "start_time", "available_at", "team", "opponent", "side",
    "win", "tournament", "is_synthetic",
)
CONTEXT_COLUMNS = ("patch", "stage", "best_of")
RATE_COLUMNS = (
    "first_blood", "first_tower", "dragon_control", "herald_control",
    "baron_control", "first_dragon", "first_herald", "first_baron",
)
SIGNED_PAIR_COLUMNS = (
    "gold_diff_10", "gold_diff_15", "xp_diff_10", "xp_diff_15",
    "cs_diff_10", "cs_diff_15", "avg_gold_diff",
)


def utc_timestamp(value: object, name: str = "timestamp") -> pd.Timestamp:
    """Require an explicit timezone rather than silently assuming local time."""
    try:
        result = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid {name}: {value!r}") from exc
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError(f"{name} must be nonempty and include a timezone: {value!r}")
    return result.tz_convert("UTC")


def discover_csvs(source: str | Path | Iterable[str | Path]) -> list[Path]:
    """Find seasons recursively; sort deterministically and reject missing paths."""
    entries = [source] if isinstance(source, (str, Path)) else list(source)
    paths: list[Path] = []
    for entry in entries:
        path = Path(entry)
        if path.is_dir():
            paths.extend(path.rglob("*.csv"))
        elif path.is_file() and path.suffix.lower() == ".csv":
            paths.append(path)
        else:
            raise FileNotFoundError(f"CSV file or directory not found: {path}")
    paths = sorted(set(p.resolve() for p in paths))
    if not paths:
        raise FileNotFoundError(
            "No historical CSV data found. Put canonical professional games in "
            "data/raw/; see docs/data_schema.md. Synthetic tests: python main.py smoke-test."
        )
    return paths


def validate_raw(frame: pd.DataFrame, *, allow_synthetic: bool = False) -> pd.DataFrame:
    """Validate rows without filling historical statistics with fabricated values."""
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing raw columns: {', '.join(sorted(missing))}. See docs/data_schema.md")
    if frame.empty:
        raise ValueError("The raw dataset is empty.")
    result = frame.copy()
    for column in ("game_id", "team", "opponent", "tournament", "side"):
        if result[column].isna().any():
            raise ValueError(f"{column} cannot be missing.")
        result[column] = result[column].astype(str).str.strip()
        if result[column].eq("").any():
            raise ValueError(f"{column} cannot be blank.")
    for column in ("patch", "stage", "roster_id"):
        if column not in result:
            result[column] = "unknown"
        result[column] = result[column].fillna("unknown").astype(str).str.strip().replace("", "unknown")
    for column in ("start_time", "available_at"):
        result[column] = pd.to_datetime([utc_timestamp(v, column) for v in result[column]], utc=True)
    if (result.available_at <= result.start_time).any():
        raise ValueError("available_at must be strictly after start_time for every completed game.")
    synthetic = result.is_synthetic.astype(str).str.lower().str.strip()
    if not synthetic.isin(["true", "false", "1", "0"]).all():
        raise ValueError("is_synthetic must be explicitly true/false or 1/0.")
    result["is_synthetic"] = synthetic.isin(["true", "1"])
    if result.is_synthetic.nunique() != 1:
        raise ValueError("Mixing real and synthetic data is prohibited.")
    if result.is_synthetic.any() and not allow_synthetic:
        raise ValueError("Synthetic data is test-only. Explicit --allow-synthetic is required.")
    result["win"] = pd.to_numeric(result.win, errors="raise")
    if not result.win.isin([0, 1]).all():
        raise ValueError("win must be 0 or 1.")
    result["win"] = result.win.astype(int)
    result["side"] = result.side.str.lower()
    if not result.side.isin(["blue", "red", "unknown"]).all():
        raise ValueError("side must be blue, red, or unknown.")
    for column in (*HISTORICAL_STATS, "best_of"):
        if column not in result:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="raise")
        if np.isinf(result[column].to_numpy(dtype=float)).any():
            raise ValueError(f"{column} cannot contain infinite values.")
    known_format = result.best_of.dropna()
    if not known_format.isin([1, 3, 5]).all():
        raise ValueError("best_of must be 1, 3, 5, or blank.")
    for column in RATE_COLUMNS:
        if not result[column].dropna().between(0, 1).all():
            raise ValueError(f"{column} must be a fraction between 0 and 1.")
        if column.startswith("first_") and not result[column].dropna().isin([0, 1]).all():
            raise ValueError(f"{column} must be binary when observed.")
    if not result.duration_seconds.dropna().gt(0).all():
        raise ValueError("duration_seconds must be positive when present.")
    known_duration = result.duration_seconds.notna()
    elapsed = (result.available_at - result.start_time).dt.total_seconds()
    if (elapsed[known_duration] < result.loc[known_duration, "duration_seconds"]).any():
        raise ValueError("available_at cannot precede the recorded game's completion.")
    if result.duplicated(["game_id", "team"]).any():
        raise ValueError("Duplicate game_id/team rows found (possibly overlapping source exports).")
    shared = ("start_time", "available_at", "tournament", "patch", "stage", "best_of", "is_synthetic")
    for game_id, group in result.groupby("game_id", sort=False):
        if len(group) != 2:
            raise ValueError(f"Game {game_id!r} must contain exactly two team rows.")
        first, second = (row for _, row in group.iterrows())
        if first.team == second.team or first.opponent != second.team or second.opponent != first.team:
            raise ValueError(f"Game {game_id!r} must have distinct, reciprocal team/opponent rows.")
        if first.win + second.win != 1:
            raise ValueError(f"Game {game_id!r} must have exactly one winner.")
        if set(group.side) not in ({"blue", "red"}, {"unknown"}):
            raise ValueError(f"Game {game_id!r} sides must be blue/red or both unknown.")
        for column in shared:
            if group[column].nunique(dropna=False) != 1:
                raise ValueError(f"Game {game_id!r} has inconsistent shared field {column}.")
        for column in HISTORICAL_STATS:
            observed = group[column].notna()
            if observed.any() and not observed.all():
                raise ValueError(
                    f"Game {game_id!r} has one-sided missing value for {column}; "
                    "paired team statistics must be observed or missing together."
                )
        if group.duration_seconds.notna().all() and not np.isclose(
            group.duration_seconds.iloc[0], group.duration_seconds.iloc[1], rtol=0, atol=1e-9,
        ):
            raise ValueError(f"Game {game_id!r} has inconsistent duration_seconds.")
        for column in SIGNED_PAIR_COLUMNS:
            if group[column].notna().all() and not np.isclose(
                group[column].sum(), 0.0, rtol=0, atol=1e-6,
            ):
                raise ValueError(
                    f"Game {game_id!r} has inconsistent {column}; team perspectives must negate."
                )
        for column in RATE_COLUMNS:
            if group[column].notna().all() and not np.isclose(
                group[column].sum(), 1.0, rtol=0, atol=1e-6,
            ):
                raise ValueError(
                    f"Game {game_id!r} has inconsistent {column}; team perspectives must complement."
                )
    return result.sort_values(["start_time", "game_id", "team"], kind="stable").reset_index(drop=True)


def normalize_games(rows: pd.DataFrame) -> pd.DataFrame:
    """Orient one sample/game using an outcome-independent SHA-256 bit.

    The sorted pair is reversed iff the first digest byte of game_id is odd.
    Team identities and game IDs are metadata, never model inputs.
    """
    records: list[dict] = []
    for game_id, group in rows.groupby("game_id", sort=False):
        pair = group.sort_values("team", kind="stable")
        if hashlib.sha256(str(game_id).encode("utf-8")).digest()[0] & 1:
            pair = pair.iloc[::-1]
        a, b = pair.iloc[0], pair.iloc[1]
        record = {
            "game_id": game_id, "start_time": a.start_time, "available_at": a.available_at,
            "team_A": a.team, "team_B": b.team, "team_A_win": int(a.win),
            "is_synthetic": bool(a.is_synthetic), "side_A": a.side,
            "tournament": a.tournament, "patch": a.patch, "stage": a.stage,
            "best_of": a.best_of, "roster_id_A": a.roster_id, "roster_id_B": b.roster_id,
        }
        for stat in HISTORICAL_STATS:
            record[f"{stat}_A"] = a[stat]
            record[f"{stat}_B"] = b[stat]
        records.append(record)
    return pd.DataFrame(records).sort_values(["start_time", "game_id"], kind="stable").reset_index(drop=True)


def load_games(source: str | Path | Iterable[str | Path] = "data/raw", *, allow_synthetic: bool = False) -> pd.DataFrame:
    """Read any number of canonical season files without modifying them."""
    paths = discover_csvs(source)
    frames = [pd.read_csv(path, dtype=str, keep_default_na=False).replace("", np.nan) for path in paths]
    validated = validate_raw(pd.concat(frames, ignore_index=True), allow_synthetic=allow_synthetic)
    games = normalize_games(validated)
    LOGGER.info("Loaded %d games from %d immutable source file(s)", len(games), len(paths))
    return games


def data_fingerprint(games: pd.DataFrame) -> str:
    """Hash canonical data content for provenance, independent of file paths."""
    payload = games.sort_values(["start_time", "game_id"]).to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def safe_output_directory(output: str | Path, sources: Iterable[Path] = ()) -> Path:
    """Protect immutable source files and the project's conventional raw tree."""
    destination = Path(output).resolve()
    raw_root = (Path(__file__).resolve().parents[1] / "data" / "raw").resolve()
    if destination == raw_root or raw_root in destination.parents:
        raise ValueError("Output must be separate from immutable data/raw.")
    for source in sources:
        source = source.resolve()
        if source.parent == destination or destination in source.parents:
            raise ValueError(f"Output directory cannot contain an input CSV: {source}")
    return destination
