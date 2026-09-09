"""As-of prediction for one future game; model and history cutoffs are enforced."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .data_loader import load_games, utc_timestamp
from .features import FeatureBuilder
from .series import series_win_probability
from .train import load_bundle


def predict_match(
    model_path: str | Path,
    data: str | Path,
    *,
    team1: str,
    team2: str,
    at: str | pd.Timestamp,
    side: str = "unknown",
    patch: str = "unknown",
    tournament: str = "unknown",
    stage: str = "unknown",
    best_of: int | None = None,
    roster1: str = "unknown",
    roster2: str = "unknown",
    allow_synthetic: bool = False,
    mode: str = "pre-draft",
) -> dict[str, Any]:
    """Predict using a fitted artifact whose labels predate the requested instant.

    An unknown side is an explicit equal mixture of blue/red scenarios. This is
    a scenario assumption, not a learned estimate of side-selection probability.
    """
    if mode != "pre-draft":
        raise ValueError("Post-draft prediction requires reliable draft data and is not implemented.")
    if team1 == team2 or not team1.strip() or not team2.strip():
        raise ValueError("Provide two distinct, nonempty team names.")
    if side not in ("blue", "red", "unknown"):
        raise ValueError("side must be blue, red, or unknown.")
    if best_of not in (None, 1, 3, 5):
        raise ValueError("best_of must be 1, 3, or 5.")
    when = utc_timestamp(at, "prediction time")
    bundle = load_bundle(model_path)
    if utc_timestamp(bundle.fitted_available_at) >= when:
        raise ValueError("Model training/calibration includes results unavailable at this prediction time. Use an earlier walk-forward model.")
    if bundle.synthetic and not allow_synthetic:
        raise ValueError("This model is SYNTHETIC / TEST ONLY; --allow-synthetic is required.")
    games = load_games(data, allow_synthetic=allow_synthetic)
    if bool(games.is_synthetic.iloc[0]) != bundle.synthetic:
        raise ValueError("Model and history must have matching real/synthetic provenance.")
    history = games[(games.start_time < when) & (games.available_at < when)]
    known = set(history.team_A) | set(history.team_B)
    missing = {team1, team2} - known
    if missing:
        raise ValueError(f"No available historical games for {', '.join(sorted(missing))}. Check team aliases and prediction time.")
    builder = FeatureBuilder(bundle.feature_config)
    scenarios = []
    sides = ("blue", "red") if side == "unknown" else (side,)
    for current_side in sides:
        row = builder.for_match(
            games, team_A=team1, team_B=team2, start_time=when,
            side_A=current_side, patch=patch, tournament=tournament, stage=stage,
            best_of=best_of, roster_id_A=roster1, roster_id_B=roster2,
        )
        scenarios.append({
            "team1_side": current_side,
            "probability": float(bundle.predict_proba(row, calibrated=True)[0]),
            "raw_probability": float(bundle.predict_proba(row, calibrated=False)[0]),
            "factors": bundle.explain(row, top_n=6),
        })
    probability = sum(item["probability"] for item in scenarios) / len(scenarios)
    warnings = []
    if side == "unknown":
        warnings.append("Side unknown: equal 50/50 mixture of blue-side and red-side scenarios.")
    if "unknown" in (patch, tournament, stage):
        warnings.append("Some pre-match context is unknown; supply only information known at --at.")
    if "unknown" in (roster1, roster2):
        warnings.append("Current roster not supplied; roster change adjustment may be unavailable.")
    latest = history.available_at.max()
    if (when - latest).days > 30:
        warnings.append(f"Historical data is {(when - latest).days} days old at prediction time.")
    if bundle.synthetic:
        warnings.insert(0, "SYNTHETIC / TEST ONLY: no real esports inference or validation.")
    result = {
        "mode": mode, "team1": team1, "team2": team2,
        "as_of": when.isoformat(), "model": bundle.model_name,
        "synthetic": bundle.synthetic,
        "team1_win_probability": probability,
        "team2_win_probability": 1.0 - probability,
        "calibration": bundle.training_config.calibration,
        "scenarios": scenarios, "warnings": warnings,
        "explanation_note": "Model associations, not causal claims. Signed logistic factors describe raw log-odds; XGBoost importance is global and has no local direction.",
    }
    if best_of is not None:
        result["series"] = {
            "best_of": best_of,
            "team1_win_probability": series_win_probability(probability, best_of),
            "assumption": "Independent games with constant win probability; a fresh whole series, not an in-progress series forecast.",
        }
    return result


def format_prediction(result: dict[str, Any]) -> str:
    """Readable report that keeps synthetic status and probability scope visible."""
    lines = []
    if result["synthetic"]:
        lines.append("SYNTHETIC / TEST ONLY")
    lines.extend([
        f"{result['team1']} vs {result['team2']} (one game, {result['mode']})",
        f"As of: {result['as_of']} | Model: {result['model']}",
        f"{result['team1']} win probability: {result['team1_win_probability']:.1%}",
        f"{result['team2']} win probability: {result['team2_win_probability']:.1%}",
        f"Calibration: {result['calibration']}",
    ])
    for scenario in result["scenarios"]:
        lines.append(f"\nModel factors ({result['team1']} on {scenario['team1_side']}, scenario {scenario['probability']:.1%}):")
        for factor in scenario["factors"]:
            name = str(factor["feature"]).removeprefix("numeric__").removeprefix("categorical__")
            if name.endswith("_diff"):
                name = name[:-5] + " (team difference)"
            name = name.replace("_", " ")
            if "contribution_raw_log_odds" in factor:
                value = factor["contribution_raw_log_odds"]
                lines.append(f"  {value:+.3f} raw log-odds: {name}")
            else:
                lines.append(f"  {factor['global_importance']:.3f} global importance (unsigned): {name}")
    lines.append(result["explanation_note"])
    if "series" in result:
        series = result["series"]
        lines.append(f"BO{series['best_of']} approximation: {series['team1_win_probability']:.1%} for {result['team1']}")
        lines.append(series["assumption"])
    lines.extend(f"Note: {warning}" for warning in result["warnings"])
    return "\n".join(lines)
