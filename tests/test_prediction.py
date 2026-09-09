"""Prediction/CLI integration checks using exclusively fictional fixture data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from src.config import TrainingConfig
from src.data_loader import load_games
from src.features import FeatureBuilder
from src.predict import format_prediction, predict_match
from src.train import load_bundle, train_models


class PredictionIntegrationTests(unittest.TestCase):
    """Train the two small models once, then exercise real inference boundaries."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.project = Path(__file__).resolve().parents[1]
        cls.fixture = cls.project / "tests" / "fixtures" / "synthetic_games.csv"
        cls.fixture_hash = hashlib.sha256(cls.fixture.read_bytes()).hexdigest()
        temporary = tempfile.TemporaryDirectory(prefix="t1-prediction-tests-")
        cls.addClassCleanup(temporary.cleanup)
        cls.output = Path(temporary.name)
        cls.games = load_games(cls.fixture, allow_synthetic=True)
        cls.features = FeatureBuilder().build_dataset(cls.games)
        cls.report = train_models(
            cls.features,
            cls.output,
            config=TrainingConfig(n_estimators=5),
            allow_synthetic=True,
        )
        cls.models = {
            name: cls.output / f"{name}.joblib"
            for name in ("logistic_regression", "xgboost")
        }
        cls.at = cls.games.available_at.max() + pd.Timedelta(days=1)
        cls.empty_data = cls.output / "empty-data"
        cls.empty_data.mkdir()

    def prediction(self, model: str = "logistic_regression", **overrides) -> dict:
        arguments = {
            "team1": "Fixture Alpha",
            "team2": "Fixture Beta",
            "at": self.at,
            "side": "blue",
            "patch": "14.3",
            "tournament": "Fixture League",
            "stage": "playoffs",
            "roster1": "fixture-alpha-v1",
            "roster2": "fixture-beta-v1",
            "allow_synthetic": True,
        }
        arguments.update(overrides)
        return predict_match(self.models[model], self.fixture, **arguments)

    def cli(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(self.project / "main.py"), *map(str, arguments)],
            cwd=self.project,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
            check=False,
        )

    def test_artifact_roundtrip_preserves_both_models(self) -> None:
        heldout = self.features.tail(8)
        for model_name, path in self.models.items():
            with self.subTest(model=model_name):
                original = load_bundle(path)
                restored_path = self.output / f"{model_name}-roundtrip.joblib"
                joblib.dump(original, restored_path)
                restored = load_bundle(restored_path)
                self.assertTrue(restored.synthetic)
                self.assertEqual(restored.feature_columns, original.feature_columns)
                self.assertEqual(restored.fitted_available_at, original.fitted_available_at)
                for calibrated in (False, True):
                    np.testing.assert_array_equal(
                        original.predict_proba(heldout, calibrated=calibrated),
                        restored.predict_proba(heldout, calibrated=calibrated),
                    )

    def test_swapping_teams_rosters_and_side_complements_prediction(self) -> None:
        for model in self.models:
            with self.subTest(model=model):
                forward = self.prediction(model, side="blue")
                reversed_game = self.prediction(
                    model,
                    team1="Fixture Beta",
                    team2="Fixture Alpha",
                    side="red",
                    roster1="fixture-beta-v1",
                    roster2="fixture-alpha-v1",
                )
                self.assertAlmostEqual(
                    forward["team1_win_probability"]
                    + reversed_game["team1_win_probability"],
                    1.0,
                    places=10,
                )
                self.assertAlmostEqual(
                    forward["team1_win_probability"] + forward["team2_win_probability"],
                    1.0,
                    places=12,
                )

    def test_unknown_side_is_equal_average_of_two_scenarios(self) -> None:
        for model in self.models:
            with self.subTest(model=model):
                unknown = self.prediction(model, side="unknown")
                blue = self.prediction(model, side="blue")
                red = self.prediction(model, side="red")
                self.assertEqual(
                    {scenario["team1_side"] for scenario in unknown["scenarios"]},
                    {"blue", "red"},
                )
                self.assertAlmostEqual(
                    unknown["team1_win_probability"],
                    (blue["team1_win_probability"] + red["team1_win_probability"]) / 2,
                    places=12,
                )
                self.assertAlmostEqual(
                    unknown["team1_win_probability"],
                    sum(scenario["probability"] for scenario in unknown["scenarios"]) / 2,
                    places=12,
                )
                self.assertTrue(any("50/50" in note for note in unknown["warnings"]))

    def test_model_cannot_predict_at_or_before_its_fitting_cutoff(self) -> None:
        for model, path in self.models.items():
            cutoff = pd.Timestamp(load_bundle(path).fitted_available_at)
            for at in (cutoff, cutoff - pd.Timedelta(days=1)):
                with self.subTest(model=model, at=at):
                    with self.assertRaisesRegex(ValueError, "unavailable|earlier|prediction time"):
                        self.prediction(model, at=at)

    def test_naive_prediction_timestamp_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "timezone"):
            self.prediction(at="2024-04-07T13:00:00")

    def test_synthetic_artifact_requires_explicit_opt_in(self) -> None:
        for model in self.models:
            with self.subTest(model=model):
                with self.assertRaisesRegex(ValueError, "SYNTHETIC|synthetic"):
                    self.prediction(model, allow_synthetic=False)

    def test_synthetic_prediction_and_saved_reports_are_labeled(self) -> None:
        self.assertTrue(self.report["synthetic"])
        saved_report = json.loads((self.output / "report.json").read_text(encoding="utf-8"))
        self.assertTrue(saved_report["synthetic"])
        predictions = pd.read_csv(self.output / "test_predictions.csv")
        self.assertTrue(predictions.is_synthetic.all())
        for model in self.models:
            with self.subTest(model=model):
                prediction = self.prediction(model)
                self.assertTrue(prediction["synthetic"])
                self.assertIn("SYNTHETIC / TEST ONLY", format_prediction(prediction))
                self.assertTrue(any("SYNTHETIC" in note for note in prediction["warnings"]))

    def test_fixture_cannot_supply_a_real_t1_prediction(self) -> None:
        names = set(self.games.team_A) | set(self.games.team_B)
        self.assertEqual(names, {"Fixture Alpha", "Fixture Beta", "Fixture Gamma", "Fixture Delta"})
        self.assertNotIn("T1", names)
        for team1, team2 in (("T1", "Gen.G"), ("Fixture Alpha", "Unseen Team")):
            with self.subTest(team1=team1, team2=team2):
                with self.assertRaisesRegex(ValueError, "No available historical games"):
                    self.prediction(team1=team1, team2=team2)

    def test_postdraft_mode_requires_real_implementation(self) -> None:
        with self.assertRaisesRegex(ValueError, "Post-draft.*not implemented"):
            self.prediction(mode="post-draft")

    def test_cli_json_prediction_runs_end_to_end(self) -> None:
        process = self.cli(
            "predict", "--model", self.models["logistic_regression"],
            "--data", self.fixture,
            "--team1", "Fixture Alpha", "--team2", "Fixture Beta",
            "--at", self.at.isoformat(), "--side", "unknown", "--best-of", "3",
            "--allow-synthetic", "--json",
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertNotIn("Traceback", process.stderr)
        result = json.loads(process.stdout)
        self.assertTrue(result["synthetic"])
        self.assertEqual(result["team1"], "Fixture Alpha")
        self.assertEqual(len(result["scenarios"]), 2)
        self.assertEqual(result["series"]["best_of"], 3)
        self.assertTrue(0 <= result["team1_win_probability"] <= 1)
        self.assertAlmostEqual(
            result["team1_win_probability"] + result["team2_win_probability"], 1.0,
        )

    def test_cli_evaluate_reads_existing_report_without_retraining(self) -> None:
        report_path = self.output / "report.json"
        before = report_path.read_bytes()
        process = self.cli("evaluate", "--report", report_path)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout), json.loads(before))
        self.assertEqual(report_path.read_bytes(), before)

    def test_cli_empty_history_fails_with_data_instructions(self) -> None:
        process = self.cli(
            "prepare", "--data", self.empty_data,
            "--output", self.output / "unused-processed",
        )
        self.assertEqual(process.returncode, 2)
        self.assertIn("No historical CSV data found", process.stderr)
        self.assertIn("docs/data_schema.md", process.stderr)
        self.assertNotIn("Traceback", process.stderr)
        self.assertFalse((self.output / "unused-processed").exists())

    def test_training_and_prediction_leave_raw_fixture_immutable(self) -> None:
        self.prediction()
        self.assertEqual(hashlib.sha256(self.fixture.read_bytes()).hexdigest(), self.fixture_hash)


if __name__ == "__main__":
    unittest.main()
