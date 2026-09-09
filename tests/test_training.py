"""Probability, orientation, train-only fitting and artifact regression checks."""

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.config import TrainingConfig
from src.evaluate import backtest, baseline_metrics, evaluate_bundle, probability_metrics
from src.features import mirror_features
from src.preprocessing import chronological_split
from src.train import fit_bundle, load_bundle, train_models


def synthetic_features(count: int = 60) -> pd.DataFrame:
    """Artificial, clearly labeled feature rows; no invented real team statistics."""
    starts = pd.date_range("2024-01-01", periods=count, freq="D", tz="UTC")
    pairs = [("Fixture A", "Fixture B"), ("Fixture C", "Fixture D"), ("Fixture A", "Fixture C")]
    return pd.DataFrame({
        "game_id": [f"training-fixture-{index}" for index in range(count)],
        "start_time": starts,
        "available_at": starts + pd.Timedelta(hours=2),
        "team_A": [pairs[index % 3][0] for index in range(count)],
        "team_B": [pairs[index % 3][1] for index in range(count)],
        "team_A_win": [index % 2 for index in range(count)],
        "is_synthetic": True,
        "elo_diff": [(-1 if index % 2 == 0 else 1) * (index % 7 + 1) * 20 for index in range(count)],
        "recent_win_rate_diff": [(index % 5 - 2) / 10 for index in range(count)],
        "missing_history_diff": np.nan,
        "best_of": [3 if index < int(count * 0.6) else 5 for index in range(count)],
        "patch": ["training_patch" if index < int(count * 0.6) else "unseen_patch" for index in range(count)],
        "tournament": "Fixture Cup",
        "stage": "regular_season",
    })


class ModelTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.frame = synthetic_features()
        cls.config = TrainingConfig(n_estimators=8, max_depth=2)
        cls.report = train_models(cls.frame, cls.directory.name, config=cls.config, allow_synthetic=True)
        cls.bundles = {
            name: load_bundle(Path(cls.directory.name) / f"{name}.joblib")
            for name in ("logistic_regression", "xgboost")
        }

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_both_models_are_calibrated_and_saved_with_provenance(self):
        self.assertTrue(self.report["synthetic"])
        self.assertIn("uniform_50_percent", self.report["baselines"])
        for name, bundle in self.bundles.items():
            with self.subTest(model=name):
                self.assertTrue(bundle.synthetic)
                self.assertIsNotNone(bundle.calibrator)
                self.assertNotIn("team_A", bundle.feature_columns)
                self.assertNotIn("team_A_win", bundle.feature_columns)
                self.assertIn("scikit-learn", bundle.versions)
                self.assertEqual(self.report["models"][name]["raw"]["games"], 12)
                self.assertEqual(self.report["models"][name]["calibrated"]["games"], 12)
        self.assertTrue((Path(self.directory.name) / "calibration.png").is_file())
        self.assertTrue((Path(self.directory.name) / "test_predictions.csv").is_file())

    def test_mirrored_probabilities_complement_before_and_after_calibration(self):
        frame = self.frame.tail(12)
        for name, bundle in self.bundles.items():
            for calibrated in (False, True):
                with self.subTest(model=name, calibrated=calibrated):
                    first = bundle.predict_proba(frame, calibrated=calibrated)
                    second = bundle.predict_proba(mirror_features(frame), calibrated=calibrated)
                    np.testing.assert_allclose(first + second, 1, atol=1e-12, rtol=0)
                    self.assertTrue(np.isfinite(first).all())

    def test_train_only_transform_excludes_future_categories_and_values(self):
        bundle = self.bundles["logistic_regression"]
        processor = bundle.pipeline.named_steps["preprocessor"]
        numeric = processor.named_transformers_["numeric"]
        best_of_index = bundle.numeric_features.index("best_of")
        self.assertEqual(numeric.named_steps["scaler"].mean_[best_of_index], 3)
        encoder = processor.named_transformers_["categorical"].named_steps["encoder"]
        self.assertNotIn("unseen_patch", set(encoder.categories_[0]))
        values = bundle.predict_proba(self.frame.tail(1))
        self.assertTrue(np.isfinite(values).all())

    def test_test_data_cannot_change_fitted_estimator_or_calibrator(self):
        split = chronological_split(self.frame)
        baseline = fit_bundle(split, "logistic_regression", config=self.config, allow_synthetic=True)
        perturbed = copy.deepcopy(split)
        perturbed.test["team_A_win"] = 1 - perturbed.test["team_A_win"]
        perturbed.test["elo_diff"] = 999999
        perturbed.test["patch"] = "future_category"
        changed = fit_bundle(perturbed, "logistic_regression", config=self.config, allow_synthetic=True)
        np.testing.assert_array_equal(
            baseline.pipeline.named_steps["estimator"].coef_, changed.pipeline.named_steps["estimator"].coef_,
        )
        np.testing.assert_array_equal(baseline.calibrator.coef_, changed.calibrator.coef_)
        np.testing.assert_array_equal(baseline.calibrator.intercept_, changed.calibrator.intercept_)

    def test_logistic_factors_decompose_raw_estimator_log_odds(self):
        bundle = self.bundles["logistic_regression"]
        frame = self.frame.tail(1)
        factors = bundle.explain(frame, top_n=1000)
        estimator = bundle.pipeline.named_steps["estimator"]
        explained = sum(factor["contribution_raw_log_odds"] for factor in factors) + estimator.intercept_[0]
        actual = bundle.pipeline.decision_function(bundle._input(frame))[0]
        self.assertAlmostEqual(explained, actual, places=10)
        self.assertTrue(all("not final probability" in factor["scope"] for factor in factors))

    def test_tree_importance_is_not_given_false_local_direction(self):
        factors = self.bundles["xgboost"].explain(self.frame.tail(1))
        self.assertTrue(all(factor["association"] == "unsigned" for factor in factors))
        self.assertTrue(all("global_importance" in factor for factor in factors))

    def test_evaluation_rejects_training_history(self):
        with self.assertRaisesRegex(ValueError, "strictly after"):
            evaluate_bundle(self.bundles["logistic_regression"], self.frame.head(2))

    def test_direct_prediction_rejects_rows_before_artifact_available(self):
        with self.assertRaisesRegex(ValueError, "strictly after"):
            self.bundles["logistic_regression"].predict_proba(self.frame.head(1))

    def test_isotonic_and_disabled_calibration_modes(self):
        split = chronological_split(self.frame)
        for method in ("isotonic", "none"):
            with self.subTest(calibration=method):
                bundle = fit_bundle(split, "logistic_regression", config=TrainingConfig(calibration=method),
                                    allow_synthetic=True)
                predicted = bundle.predict_proba(split.test)
                reversed_probability = bundle.predict_proba(mirror_features(split.test))
                np.testing.assert_allclose(predicted + reversed_probability, 1, atol=1e-12, rtol=0)
                if method == "none":
                    self.assertIsNone(bundle.calibrator)
                    np.testing.assert_array_equal(predicted, bundle.predict_proba(split.test, calibrated=False))

    def test_output_cannot_overwrite_raw_data_tree(self):
        raw_path = Path(__file__).resolve().parents[1] / "data" / "raw"
        with self.assertRaisesRegex(ValueError, "immutable"):
            train_models(self.frame, raw_path, config=self.config, allow_synthetic=True)

    def test_default_training_rejects_synthetic_data(self):
        with self.assertRaisesRegex(ValueError, "Synthetic fixtures"):
            train_models(self.frame, self.directory.name, config=self.config)

    def test_real_and_synthetic_data_cannot_be_mixed(self):
        mixed = self.frame.copy()
        mixed.loc[0, "is_synthetic"] = False
        with self.assertRaisesRegex(ValueError, "mix real and synthetic"):
            train_models(mixed, self.directory.name, config=self.config, allow_synthetic=True)

    def test_single_team_only_history_is_not_accepted(self):
        team_only = self.frame.copy()
        team_only["team_A"] = "Fixture Target"
        with self.assertRaisesRegex(ValueError, "same team"):
            train_models(team_only, self.directory.name, config=self.config, allow_synthetic=True)

    def test_walk_forward_backtest_reports_unique_later_games(self):
        with tempfile.TemporaryDirectory() as destination:
            report = backtest(self.frame, destination, config=self.config, n_splits=2, allow_synthetic=True)
            scored = pd.read_csv(Path(destination) / "backtest_predictions.csv")
            self.assertEqual(len(report["folds"]), 2)
            self.assertFalse(scored.duplicated(["model", "game_id"]).any())
            self.assertTrue((pd.to_datetime(scored.start_time, utc=True) >
                             pd.to_datetime(scored.model_fitted_available_at, utc=True)).all())
            self.assertEqual(report["models"]["logistic_regression"]["raw"]["games"], 24)
            self.assertEqual(report["baselines"]["uniform_50_percent"]["games"], 24)
            self.assertIn("uniform_50_percent", report["folds"][0]["baselines"])


class MetricTests(unittest.TestCase):
    def test_uniform_and_elo_baselines(self):
        frame = pd.DataFrame({
            "team_A_win": [0, 1, 1],
            "elo_rating_diff": [-400.0, 0.0, 400.0],
        })
        metrics = baseline_metrics(frame)
        self.assertEqual(metrics["uniform_50_percent"]["accuracy"], 2 / 3)
        self.assertEqual(metrics["elo"]["accuracy"], 1.0)
        self.assertLess(metrics["elo"]["log_loss"], metrics["uniform_50_percent"]["log_loss"])

    def test_known_probability_metrics(self):
        metrics = probability_metrics([0, 1], [0.2, 0.8])
        self.assertAlmostEqual(metrics["log_loss"], -np.log(0.8))
        self.assertAlmostEqual(metrics["brier_score"], 0.04)
        self.assertEqual(metrics["roc_auc"], 1)
        self.assertEqual(metrics["accuracy"], 1)

    def test_single_class_auc_is_explicitly_undefined(self):
        metrics = probability_metrics([1, 1], [0.6, 0.8])
        self.assertIsNone(metrics["roc_auc"])
        self.assertTrue(np.isfinite(metrics["log_loss"]))

    def test_rejects_invalid_probabilities(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            probability_metrics([0, 1], [np.nan, 0.8])


if __name__ == "__main__":
    unittest.main()
