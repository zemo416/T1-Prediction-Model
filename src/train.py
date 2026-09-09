"""Train-only preprocessing, chronological calibration and portable model bundles."""

from dataclasses import asdict, dataclass, field
from importlib.metadata import version
import json
import logging
from pathlib import Path
import platform
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .config import CATEGORICAL_FEATURES, METADATA_COLUMNS, FeatureConfig, TrainingConfig
from .data_loader import data_fingerprint as fingerprint_data, safe_output_directory, utc_timestamp
from .preprocessing import ChronologicalSplit, chronological_split

LOGGER = logging.getLogger(__name__)


def _mirror(frame: pd.DataFrame) -> pd.DataFrame:
    # Keep the orientation contract in the feature module as the single source.
    from .features import mirror_features

    return mirror_features(frame)


def _log_odds(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-8, 1 - 1e-8)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


@dataclass
class ModelBundle:
    """Trusted-local joblib artifact with fitted transforms and provenance.

``predict_proba`` returns P(team_A wins), independent of team-name ordering.
Never load joblib artifacts from untrusted sources: pickle can execute code.
    """

    model_name: str
    feature_config: FeatureConfig
    training_config: TrainingConfig
    feature_columns: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    pipeline: Pipeline
    calibrator: LogisticRegression | IsotonicRegression | None
    training_available_at: str
    calibration_available_at: str | None
    fitted_available_at: str
    synthetic: bool
    versions: dict[str, str]
    data_fingerprint: str | None = None
    schema_version: int = 1
    split_summary: dict = field(default_factory=dict)

    def _input(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.feature_columns).difference(frame.columns)
        if missing:
            raise ValueError(f"Prediction is missing fitted features: {sorted(missing)}")
        selected = frame[self.feature_columns].copy()
        for name in self.numeric_features:
            selected[name] = pd.to_numeric(selected[name], errors="raise").astype(float)
            selected[name] = selected[name].replace([np.inf, -np.inf], np.nan)
        for name in self.categorical_features:
            selected[name] = selected[name].map(
                lambda value: str(value) if pd.notna(value) else np.nan,
            ).astype(object)
        return selected

    def predict_proba(self, frame: pd.DataFrame, calibrated: bool = True) -> np.ndarray:
        """Return symmetrized probabilities; swapping teams complements them.

When game metadata is supplied, enforce the model's availability cutoff.
Feature-only frames require an as-of check by the caller (the prediction CLI
performs this check before building a future game's feature row).
        """
        if "start_time" in frame:
            times = frame["start_time"].map(lambda value: utc_timestamp(value, "prediction time"))
            if (times <= utc_timestamp(self.fitted_available_at)).any():
                raise ValueError("Prediction games must start strictly after fitted results became available.")
        return self._predict_proba_unchecked(frame, calibrated=calibrated)

    def _predict_proba_unchecked(self, frame: pd.DataFrame, calibrated: bool = True) -> np.ndarray:
        """Internal scoring also used to fit the later calibration block."""
        original = self._input(frame)
        reverse = self._input(_mirror(frame))
        forward_probability = np.asarray(self.pipeline.predict_proba(original)[:, 1], dtype=float)
        reverse_probability = np.asarray(self.pipeline.predict_proba(reverse)[:, 1], dtype=float)
        raw = 0.5 + 0.5 * (forward_probability - reverse_probability)
        if not calibrated or self.calibrator is None:
            return raw
        if isinstance(self.calibrator, IsotonicRegression):
            calibrated_forward = self.calibrator.predict(raw)
            calibrated_reverse = self.calibrator.predict(1 - raw)
        else:
            calibrated_forward = self.calibrator.predict_proba(_log_odds(raw))[:, 1]
            calibrated_reverse = self.calibrator.predict_proba(_log_odds(1 - raw))[:, 1]
        return 0.5 + 0.5 * (calibrated_forward - calibrated_reverse)

    def explain(self, frame: pd.DataFrame, top_n: int = 8) -> list[dict[str, Any]]:
        """Explain one game with honest raw-score or global-importance labels.

Logistic contributions sum with the intercept to the original estimator's
raw log odds. They do not exactly decompose the final symmetrized/calibrated
probability. XGBoost importances are global and have no signed direction.
        """
        if len(frame) != 1:
            raise ValueError("explain expects exactly one prediction row.")
        if top_n < 1:
            raise ValueError("top_n must be positive.")
        transform = self.pipeline.named_steps["preprocessor"]
        estimator = self.pipeline.named_steps["estimator"]
        names = transform.get_feature_names_out()
        transformed = transform.transform(self._input(frame))
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        if self.model_name == "logistic_regression":
            contributions = transformed[0] * estimator.coef_[0]
            order = np.argsort(-np.abs(contributions), kind="stable")[:top_n]
            return [
                {
                    "feature": str(names[index]),
                    "contribution_raw_log_odds": float(contributions[index]),
                    "transformed_value": float(transformed[0, index]),
                    "association": "toward_team_A" if contributions[index] > 0 else (
                        "toward_team_B" if contributions[index] < 0 else "neutral"
                    ),
                    "scope": "raw estimator log odds; not final probability or causality",
                }
                for index in order
            ]
        importance = estimator.feature_importances_
        order = np.argsort(-importance, kind="stable")[:top_n]
        return [
            {
                "feature": str(names[index]),
                "global_importance": float(importance[index]),
                "association": "unsigned",
                "scope": "global model importance; not a local contribution or causality",
            }
            for index in order
        ]


def _validate_training_data(features: pd.DataFrame, allow_synthetic: bool) -> bool:
    missing = set(METADATA_COLUMNS).difference(features.columns)
    if missing:
        raise ValueError(f"Missing feature metadata: {sorted(missing)}")
    if features.empty:
        raise ValueError("No games to train on.")
    if features["team_A_win"].isna().any() or not features["team_A_win"].isin([0, 1]).all():
        raise ValueError("team_A_win must contain observed binary results.")
    provenance = features["is_synthetic"]
    if provenance.isna().any() or not provenance.isin([True, False, 0, 1]).all():
        raise ValueError("is_synthetic must be explicit booleans.")
    if provenance.astype(bool).nunique() != 1:
        raise ValueError("Never mix real and synthetic games for training or evaluation.")
    synthetic = bool(provenance.astype(bool).iloc[0])
    if synthetic and not allow_synthetic:
        raise ValueError("Synthetic fixtures are for testing only; use explicit allow_synthetic.")
    teams = set(features["team_A"]) | set(features["team_B"])
    if len(teams) < 3:
        raise ValueError("Train on broader professional history containing at least three teams.")
    common = set(features.iloc[0][["team_A", "team_B"]])
    for team_a, team_b in features[["team_A", "team_B"]].itertuples(index=False, name=None):
        common.intersection_update((team_a, team_b))
        if not common:
            break
    if common:
        raise ValueError("Every game features the same team; provide broader, non-team-only history.")
    return synthetic


def _estimator(model_name: str, config: TrainingConfig):
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=2000, random_state=config.seed)
    if model_name == "xgboost":
        return XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            n_estimators=config.n_estimators, max_depth=config.max_depth,
            learning_rate=config.learning_rate, random_state=config.seed,
            n_jobs=1, subsample=1.0, colsample_bytree=1.0,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def fit_bundle(
    split: ChronologicalSplit,
    model_name: str,
    feature_config: FeatureConfig = FeatureConfig(),
    config: TrainingConfig = TrainingConfig(),
    allow_synthetic: bool = False,
    data_fingerprint: str | None = None,
) -> ModelBundle:
    """Fit one fixed model on train, then optional calibration on later validation."""
    synthetic = _validate_training_data(split.train, allow_synthetic)
    _validate_training_data(pd.concat([split.train, split.validation, split.test]), allow_synthetic)
    if config.calibration not in ("sigmoid", "isotonic", "none"):
        raise ValueError("calibration must be sigmoid, isotonic, or none.")
    if split.train["team_A_win"].nunique() < 2:
        raise ValueError("Training must contain both observed result classes.")
    if config.calibration != "none" and split.validation["team_A_win"].nunique() < 2:
        raise ValueError("Calibration requires both observed result classes; use a larger block.")
    if pd.to_datetime(split.train["available_at"], utc=True).max() >= pd.to_datetime(
        split.validation["start_time"], utc=True,
    ).min():
        raise ValueError("Training results must be available before calibration starts.")
    if pd.to_datetime(split.validation["available_at"], utc=True).max() >= pd.to_datetime(
        split.test["start_time"], utc=True,
    ).min():
        raise ValueError("Calibration results must be available before testing starts.")
    columns = [name for name in split.train.columns if name not in METADATA_COLUMNS]
    categorical = [name for name in columns if name in CATEGORICAL_FEATURES]
    numeric = [name for name in columns if name not in categorical]
    if not numeric:
        raise ValueError("No numeric pre-match features were supplied.")
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown", keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ], remainder="drop")
    training_available = pd.to_datetime(split.train["available_at"], utc=True).max()
    calibration_available = pd.to_datetime(split.validation["available_at"], utc=True).max()
    bundle = ModelBundle(
        model_name=model_name, feature_config=feature_config, training_config=config,
        feature_columns=columns, numeric_features=numeric, categorical_features=categorical,
        pipeline=Pipeline([("preprocessor", preprocessor), ("estimator", _estimator(model_name, config))]),
        calibrator=None, training_available_at=training_available.isoformat(),
        calibration_available_at=calibration_available.isoformat() if config.calibration != "none" else None,
        fitted_available_at=(max(training_available, calibration_available)
                             if config.calibration != "none" else training_available).isoformat(),
        synthetic=synthetic,
        versions={"python": platform.python_version(),
                  **{name: version(name) for name in ("numpy", "pandas", "scikit-learn", "xgboost", "joblib")}},
        data_fingerprint=data_fingerprint, split_summary=split.summary(),
    )
    # Mirror only after the game-level time split. Do not add the duplicated
    # game labels into rolling history or any reported evaluation sample count.
    training_frame = pd.concat([split.train, _mirror(split.train)], ignore_index=True)
    labels = split.train["team_A_win"].astype(int).to_numpy()
    bundle.pipeline.fit(bundle._input(training_frame), np.concatenate([labels, 1 - labels]))
    if config.calibration != "none":
        validation_raw = bundle._predict_proba_unchecked(split.validation, calibrated=False)
        calibration_p = np.concatenate([validation_raw, 1 - validation_raw])
        validation_y = split.validation["team_A_win"].astype(int).to_numpy()
        calibration_y = np.concatenate([validation_y, 1 - validation_y])
        if config.calibration == "sigmoid":
            bundle.calibrator = LogisticRegression(C=1e6, max_iter=2000, random_state=config.seed)
            bundle.calibrator.fit(_log_odds(calibration_p), calibration_y)
        else:
            bundle.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            bundle.calibrator.fit(calibration_p, calibration_y)
    LOGGER.info("Fitted %s with %d train games and %d calibration games%s", model_name,
                len(split.train), len(split.validation), " (SYNTHETIC TEST ONLY)" if synthetic else "")
    return bundle


def train_models(
    features: pd.DataFrame,
    output_dir: str | Path,
    feature_config: FeatureConfig = FeatureConfig(),
    config: TrainingConfig = TrainingConfig(),
    train_end: str | pd.Timestamp | None = None,
    validation_end: str | pd.Timestamp | None = None,
    allow_synthetic: bool = False,
    data_fingerprint: str | None = None,
) -> dict:
    """Train both prespecified models and compare calibration on untouched test games.

No test-set model selection or refit occurs. Artifacts carry synthetic
provenance when explicit fixture mode is requested.
    """
    from .evaluate import baseline_metrics, evaluate_bundle, plot_calibration

    synthetic = _validate_training_data(features, allow_synthetic)
    split = chronological_split(
        features, config.train_fraction, config.validation_fraction, train_end, validation_end,
    )
    data_fingerprint = data_fingerprint or fingerprint_data(features)
    destination = safe_output_directory(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report = {
        "synthetic": synthetic,
        "purpose": "SYNTHETIC PIPELINE TEST ONLY" if synthetic else "historical held-out evaluation",
        "feature_config": asdict(feature_config),
        "training_config": asdict(config),
        "split": split.summary(),
        "calibration_note": ("Calibration disabled; validation block is unused."
                             if config.calibration == "none" else
                             "Validation fits calibration and is not an unbiased evaluation set."),
        "selection_note": "Both prespecified models reported; no model chosen using the test set.",
        "data_fingerprint": data_fingerprint,
        "models": {},
        "baselines": baseline_metrics(split.test),
    }
    all_predictions = []
    for model_name in ("logistic_regression", "xgboost"):
        bundle = fit_bundle(split, model_name, feature_config, config, allow_synthetic, data_fingerprint)
        metrics, predictions = evaluate_bundle(bundle, split.test)
        report["models"][model_name] = metrics
        all_predictions.append(predictions)
        joblib.dump(bundle, destination / f"{model_name}.joblib")
    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(destination / "test_predictions.csv", index=False)
    plot_calibration(predictions, destination / "calibration.png", synthetic=synthetic)
    with (destination / "report.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def load_bundle(path: str | Path) -> ModelBundle:
    """Load a trusted local model artifact and check its schema type."""
    bundle = joblib.load(path)
    if not isinstance(bundle, ModelBundle) or bundle.schema_version != 1:
        raise ValueError("Unsupported model bundle; retrain with this project version.")
    return bundle
