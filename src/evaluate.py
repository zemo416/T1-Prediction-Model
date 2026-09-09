"""Probability metrics and expanding-window historical backtests."""

from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from .config import FeatureConfig, TrainingConfig
from .data_loader import data_fingerprint as fingerprint_data, safe_output_directory, utc_timestamp
from .preprocessing import walk_forward_splits

if TYPE_CHECKING:
    from .train import ModelBundle


def probability_metrics(y_true, probability) -> dict[str, float | int | None]:
    """Evaluate binary probabilities; AUC is undefined for a single-class block."""
    target = np.asarray(y_true)
    predicted = np.asarray(probability, dtype=float)
    if target.ndim != 1 or predicted.ndim != 1 or len(target) != len(predicted) or len(target) == 0:
        raise ValueError("Nonempty one-dimensional labels and predictions must have equal lengths.")
    if not np.isin(target, [0, 1]).all():
        raise ValueError("Evaluation labels must be binary.")
    if not np.isfinite(predicted).all() or ((predicted < 0) | (predicted > 1)).any():
        raise ValueError("Predicted probabilities must be finite values between 0 and 1.")
    return {
        "games": len(target),
        "log_loss": float(log_loss(target, predicted, labels=[0, 1])),
        "brier_score": float(brier_score_loss(target, predicted)),
        "roc_auc": float(roc_auc_score(target, predicted)) if len(np.unique(target)) == 2 else None,
        "accuracy": float(accuracy_score(target, predicted >= 0.5)),
    }


def baseline_metrics(frame: pd.DataFrame) -> dict[str, dict[str, float | int | None]]:
    """Score reference forecasts without fitting to the held-out outcomes.

    The uniform forecast reflects the orientation-symmetric prior. When the
    leakage-safe Elo rating difference is present, also expose the standard
    400-point Elo expectation as a stronger modeling baseline.
    """
    if "team_A_win" not in frame or frame.empty:
        raise ValueError("Baseline evaluation requires nonempty team_A_win labels.")
    target = frame["team_A_win"].to_numpy()
    result = {
        "uniform_50_percent": probability_metrics(target, np.full(len(frame), 0.5)),
    }
    if "elo_rating_diff" in frame:
        difference = pd.to_numeric(frame["elo_rating_diff"], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(difference).all():
            raise ValueError("Elo baseline requires finite elo_rating_diff values.")
        exponent = np.clip(-difference / 400.0, -300.0, 300.0)
        probability = 1.0 / (1.0 + np.power(10.0, exponent))
        result["elo"] = probability_metrics(target, probability)
    return result


def evaluate_bundle(bundle: "ModelBundle", frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Score strictly later held-out games with both raw and calibrated probabilities."""
    required = {"game_id", "start_time", "available_at", "team_A", "team_B", "team_A_win", "is_synthetic"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Evaluation metadata missing: {sorted(required.difference(frame.columns))}")
    if frame.empty:
        raise ValueError("No held-out games to evaluate.")
    if frame["game_id"].duplicated().any():
        raise ValueError("Evaluate unique games, not mirrored duplicate observations.")
    times = pd.to_datetime(frame["start_time"].map(lambda value: utc_timestamp(value, "game start")), utc=True)
    if times.isna().any() or (times <= pd.Timestamp(bundle.fitted_available_at)).any():
        raise ValueError("Evaluation games must start strictly after all model/calibration results became available.")
    provenance = frame["is_synthetic"]
    if provenance.isna().any() or not provenance.isin([0, 1, True, False]).all():
        raise ValueError("Evaluation requires explicit synthetic provenance.")
    if not (provenance.astype(bool) == bundle.synthetic).all():
        raise ValueError("Evaluation data and model must have matching real/synthetic provenance.")
    raw = bundle.predict_proba(frame, calibrated=False)
    calibrated = bundle.predict_proba(frame, calibrated=True)
    target = frame["team_A_win"].to_numpy()
    metrics = {
        "raw": probability_metrics(target, raw),
        "calibrated": probability_metrics(target, calibrated),
    }
    columns = ["game_id", "start_time", "available_at", "team_A", "team_B", "team_A_win", "is_synthetic"]
    predictions = frame[columns].copy()
    predictions["model"] = bundle.model_name
    predictions["raw_probability"] = raw
    predictions["calibrated_probability"] = calibrated
    return metrics, predictions


def plot_calibration(predictions: pd.DataFrame, output_path: str | Path, synthetic: bool = False) -> None:
    """Write an offscreen reliability diagram for held-out raw/calibrated scores."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if "MPLCONFIGDIR" not in os.environ:
        cache = destination.parent.resolve() / ".matplotlib"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache))
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if predictions.empty:
        raise ValueError("Cannot plot an empty prediction set.")
    figure = Figure(figsize=(8, 6), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.plot([0, 1], [0, 1], color="gray", linestyle=":", label="Perfect calibration")
    for model_name, group in predictions.groupby("model", sort=False):
        for label, column, line_style in (
            ("raw", "raw_probability", "--"),
            ("calibrated", "calibrated_probability", "-"),
        ):
            observed, predicted = calibration_curve(
                group["team_A_win"], group[column], n_bins=10, strategy="uniform",
            )
            axes.plot(predicted, observed, marker="o", linestyle=line_style,
                      label=f"{model_name}: {label}")
    title = "Held-out game probability calibration"
    if synthetic:
        title += "\nSYNTHETIC PIPELINE TEST ONLY"
    axes.set(title=title, xlabel="Mean predicted P(team_A wins)",
             ylabel="Observed team_A win fraction", xlim=(0, 1), ylim=(0, 1))
    axes.grid(alpha=0.2)
    axes.legend(fontsize="small")
    figure.savefig(destination, dpi=150)
    figure.clear()


def backtest(
    features: pd.DataFrame,
    output_dir: str | Path,
    feature_config: FeatureConfig = FeatureConfig(),
    config: TrainingConfig = TrainingConfig(),
    n_splits: int = 3,
    min_train_fraction: float = 0.4,
    calibration_fraction: float = 0.2,
    allow_synthetic: bool = False,
    data_fingerprint: str | None = None,
) -> dict:
    """Refit each fold on strictly historical labels, then pool unique test games.

Feature rows must come from the sequential leakage-safe FeatureBuilder.
Historical features may incorporate earlier completed test games because
this simulates predictions made game by game as results become available.
Each estimator and its calibrator remain fixed throughout their test block.
    """
    from .train import _validate_training_data, fit_bundle

    synthetic = _validate_training_data(features, allow_synthetic)
    data_fingerprint = data_fingerprint or fingerprint_data(features)
    destination = safe_output_directory(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report = {
        "synthetic": synthetic,
        "purpose": "SYNTHETIC PIPELINE TEST ONLY" if synthetic else "chronological walk-forward evaluation",
        "feature_config": asdict(feature_config),
        "training_config": asdict(config),
        "data_fingerprint": data_fingerprint,
        "selection_note": "Prespecified fixed configurations; no test-set model selection.",
        "calibration_note": ("Calibration disabled; each validation block is unused."
                             if config.calibration == "none" else
                             "Each fold uses a later disjoint calibration block before test begins."),
        "folds": [],
        "models": {},
        "baselines": {},
    }
    all_predictions = []
    baseline_frames = []
    for fold_number, split in enumerate(walk_forward_splits(
        features, n_splits, min_train_fraction, calibration_fraction,
    ), start=1):
        fold_report = {
            "fold": fold_number,
            "split": split.summary(),
            "models": {},
            "baselines": baseline_metrics(split.test),
        }
        baseline_frames.append(split.test)
        for model_name in ("logistic_regression", "xgboost"):
            bundle = fit_bundle(split, model_name, feature_config, config, allow_synthetic, data_fingerprint)
            metrics, predictions = evaluate_bundle(bundle, split.test)
            fold_report["models"][model_name] = metrics
            predictions["fold"] = fold_number
            predictions["model_fitted_available_at"] = bundle.fitted_available_at
            all_predictions.append(predictions)
        report["folds"].append(fold_report)
    predictions = pd.concat(all_predictions, ignore_index=True)
    if predictions.duplicated(["model", "game_id"]).any():
        raise ValueError("A game appeared in more than one test fold.")
    baseline_data = pd.concat(baseline_frames, ignore_index=True)
    if baseline_data["game_id"].duplicated().any():
        raise ValueError("A game appeared in more than one baseline test fold.")
    report["baselines"] = baseline_metrics(baseline_data)
    for model_name, group in predictions.groupby("model", sort=False):
        report["models"][model_name] = {
            "raw": probability_metrics(group["team_A_win"], group["raw_probability"]),
            "calibrated": probability_metrics(group["team_A_win"], group["calibrated_probability"]),
        }
    predictions.to_csv(destination / "backtest_predictions.csv", index=False)
    plot_calibration(predictions, destination / "backtest_calibration.png", synthetic)
    with (destination / "backtest_report.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report
