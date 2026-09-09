"""Command-line entry point; run from the project root on Windows or POSIX."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.config import FeatureConfig, TrainingConfig
from src.data_loader import discover_csvs, load_games, safe_output_directory

LOGGER = logging.getLogger(__name__)


def parser() -> argparse.ArgumentParser:
    """Build commands without performing training or loading artifacts on import."""
    root = argparse.ArgumentParser(description="Pre-match LoL game probabilities; historical data required.")
    root.add_argument("--verbose", action="store_true")
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Validate raw CSVs and generate historical features")
    train = commands.add_parser("train", help="Chronological logistic/XGBoost training and calibration")
    backtest = commands.add_parser("backtest", help="Expanding-window chronological backtesting")
    for command in (prepare, train, backtest):
        command.add_argument("--data", type=Path, default=Path("data/raw"))
        command.add_argument("--allow-synthetic", action="store_true", help="Test-only data; never real predictions")
    prepare.add_argument("--output", type=Path, default=Path("data/processed"))
    for command in (train, backtest):
        command.add_argument("--output", type=Path, default=Path("models") if command is train else Path("artifacts/backtest"))
        command.add_argument("--calibration", choices=["sigmoid", "isotonic", "none"], default="sigmoid")
        command.add_argument("--seed", type=int, default=42)
    train.add_argument("--train-end", help="Exclusive train cutoff, timezone-aware; requires --validation-end")
    train.add_argument("--validation-end", help="Exclusive calibration cutoff / test start, timezone-aware")
    backtest.add_argument("--folds", type=int, default=3)
    evaluate = commands.add_parser("evaluate", help="Display the saved held-out evaluation report")
    evaluate.add_argument("--report", type=Path, default=Path("models/report.json"))
    predict = commands.add_parser("predict", help="Predict one game at an explicit pre-match time")
    predict.add_argument("--model", type=Path, default=Path("models/logistic_regression.joblib"))
    predict.add_argument("--data", type=Path, default=Path("data/raw"))
    predict.add_argument("--team1", required=True)
    predict.add_argument("--team2", required=True)
    predict.add_argument("--at", required=True, help="Timezone-aware pre-match timestamp, e.g. 2026-09-10T08:00:00Z")
    predict.add_argument("--side", choices=["blue", "red", "unknown"], default="unknown")
    for context in ("patch", "tournament", "stage", "roster1", "roster2"):
        predict.add_argument(f"--{context}", default="unknown")
    predict.add_argument("--best-of", type=int, choices=[1, 3, 5])
    predict.add_argument("--mode", choices=["pre-draft", "post-draft"], default="pre-draft")
    predict.add_argument("--allow-synthetic", action="store_true")
    predict.add_argument("--json", action="store_true", dest="as_json")
    smoke = commands.add_parser("smoke-test", help="End-to-end test using small fictional fixture only")
    smoke.add_argument("--output", type=Path, default=Path("artifacts/smoke"))
    return root


def main(argv: list[str] | None = None) -> int:
    """Return a nonzero status with an actionable message on invalid input."""
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if args.command == "evaluate":
            report = json.loads(args.report.read_text(encoding="utf-8"))
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0
        if args.command == "predict":
            from src.predict import format_prediction, predict_match
            result = predict_match(
                args.model, args.data, team1=args.team1, team2=args.team2,
                at=args.at, side=args.side, patch=args.patch, tournament=args.tournament,
                stage=args.stage, best_of=args.best_of, roster1=args.roster1, roster2=args.roster2,
                allow_synthetic=args.allow_synthetic, mode=args.mode,
            )
            print(json.dumps(result, indent=2) if args.as_json else format_prediction(result))
            return 0
        from src.features import FeatureBuilder
        from src.train import train_models
        feature_config = FeatureConfig()
        if args.command == "smoke-test":
            from src.predict import format_prediction, predict_match
            fixture = Path(__file__).parent / "tests/fixtures/synthetic_games.csv"
            output = safe_output_directory(args.output, [fixture])
            games = load_games(fixture, allow_synthetic=True)
            features = FeatureBuilder(feature_config).build_dataset(games)
            report = train_models(features, output, feature_config=feature_config, config=TrainingConfig(n_estimators=10), allow_synthetic=True)
            first = games.iloc[0]
            import pandas as pd
            prediction = predict_match(
                output / "logistic_regression.joblib", fixture,
                team1=first.team_A, team2=first.team_B,
                at=games.available_at.max() + pd.Timedelta(days=1), allow_synthetic=True,
                tournament="Fixture League", best_of=3,
            )
            print(format_prediction(prediction))
            print(f"\nSmoke test passed. TEST-ONLY artifacts: {output}")
            return 0
        sources = discover_csvs(args.data)
        output = safe_output_directory(args.output, sources)
        games = load_games(args.data, allow_synthetic=args.allow_synthetic)
        features = FeatureBuilder(feature_config).build_dataset(games)
        if args.command == "prepare":
            output.mkdir(parents=True, exist_ok=True)
            features.to_csv(output / "features.csv", index=False)
            print(f"Prepared {len(features)} game samples: {output / 'features.csv'}")
            return 0
        config = TrainingConfig(seed=args.seed, calibration=args.calibration)
        if args.command == "train":
            report = train_models(features, output, feature_config=feature_config, config=config,
                                  train_end=args.train_end, validation_end=args.validation_end,
                                  allow_synthetic=args.allow_synthetic)
        else:
            from src.evaluate import backtest
            report = backtest(features, output, feature_config=feature_config, config=config,
                              n_splits=args.folds, allow_synthetic=args.allow_synthetic)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    except (ValueError, FileNotFoundError, OSError, ImportError) as exc:
        LOGGER.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
