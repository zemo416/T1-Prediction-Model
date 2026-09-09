# T1 Match Predictor

A Python 3.12 backend for estimating the probability of winning an individual
professional League of Legends map before play begins. Train on broad
professional match history, then request a prediction for T1 or any covered
team. There is no frontend and no fabricated T1 history.

The initial implementation includes canonical CSV validation, chronological
historical features, decaying Elo, logistic regression and XGBoost,
probability calibration, holdout evaluation, walk-forward backtesting,
explanations, a prediction CLI, and constant-probability BO3/BO5 utilities.
Real-world accuracy and calibration cannot be established without real data.
The only committed dataset is a small, explicitly synthetic test fixture.

## Installation

Use Python **3.12**. The dependency set is pandas, NumPy, scikit-learn, XGBoost,
matplotlib, and joblib; the tests use Python's standard `unittest` module.

PowerShell with an existing Python 3.12 installation:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe main.py smoke-test
```

Alternatively, if `uv` is installed:

```powershell
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe main.py smoke-test
```

On macOS/Linux, use `python3.12 -m venv .venv`, then substitute
`.venv/bin/python` for `.venv\Scripts\python.exe`. Run commands from the project
root. The remaining examples use `python` assuming the environment is active;
the explicit interpreter path works without activating it.

`smoke-test` uses 96 fictional maps, writes its outputs under
`artifacts/smoke/` by default, and identifies every result as synthetic. It is a pipeline
check, not a measure of esports prediction quality. Normal data loading rejects
synthetic rows unless `--allow-synthetic` is explicitly supplied; real and
synthetic observations cannot be mixed.

## Project layout

```text
data/
  raw/                  Immutable canonical historical CSV files
  processed/            Generated one-row-per-map feature tables
models/                 Fitted models and evaluation artifacts
src/
  config.py             Feature and training configuration
  data_loader.py        Strict schema validation and map normalization
  elo.py                Ratings, inactivity decay, roster regression
  features.py           Historical feature generation and extension points
  preprocessing.py      Chronological partitions and availability purging
  train.py              Train-only transforms, models, calibration, bundles
  evaluate.py           Metrics, calibration plots, walk-forward evaluation
  predict.py            Timestamped map inference and explanations
  series.py             BO1/BO3/BO5 probability conversion
tests/
  fixtures/             Clearly labeled fictional test data
docs/data_schema.md     Full source-data contract and conversion guidance
main.py                 Command-line entry point
```

## Real data needed before meaningful training

Provide several recent seasons of map-level professional history covering
**all participating teams** in LCK, MSI, and Worlds, preferably with other
regional leagues as well. International opponents need enough prior history
to support useful strength estimates. Training only on T1 would provide too
few observations and weak opponent coverage.

Place canonical CSV files in `data/raw/`; subdirectories are read recursively,
so additional seasons can be added as `data/raw/2025/lck.csv`, for example.
Each completed map must have exactly two rows, one for each team, with these
required columns:

```text
game_id,start_time,available_at,team,opponent,side,win,tournament,is_synthetic
```

The teams must be distinct reciprocal opponents with opposite binary `win`
values. `side` is `blue`, `red`, or `unknown`, and real rows must explicitly
declare `is_synthetic=false`. Both timestamps must include a timezone;
`available_at` is strictly after `start_time` and must be at least the time
the map finished and its included statistics became available. The same
map's two rows must agree on shared context. Optional columns include patch,
stage, best-of format, a pre-match-known roster identifier, and historical
early-game/objective statistics. Read the complete
[data contract](docs/data_schema.md) before conversion.

[Oracle's Elixir downloads](https://oracleselixir.com/tools/downloads) is a
candidate source of historical professional data. Its native CSV format is
**not directly supported**: a reviewed adapter must establish the required
schema, timestamps, alias mapping, statistic definitions, and pre-match
availability of roster/side context. Keep original downloads separately and
retain conversion notes. Do not fabricate availability timestamps to make a
file pass validation.

No real-data download or T1 prediction is bundled. To proceed beyond the
smoke test, supply those real canonical files, or provide the original source
exports plus timestamp/provenance information needed to build their adapter.

## Features and orientation

One normalized sample is one map, with `team_A`, `team_B`, and `team_A_win`.
The two team names are sorted and a stable SHA-256 bit of `game_id` chooses
whether to reverse the pair. This rule is deterministic and independent of
the result. Team names, raw IDs, and raw timestamps are metadata, not direct
model inputs.

Training augments each map with its mirrored orientation and flipped label.
Mirroring preserves context while swapping team features and negating signed
differences/side. Both orientations stay in the same chronological partition.
Inference symmetrizes predictions, so reversing the teams and their side
produces complementary probabilities. Calibration is also applied with
orientation symmetry preserved. This guards against arbitrary team ordering
becoming a learned advantage.

| Feature family | Definition |
| --- | --- |
| Elo | Pre-map rating difference; standard opponent-dependent updates, inactivity decay toward the initial rating, and optional roster-change regression. |
| Form | Historical win rate in recent windows, last-five and last-ten map win rates, and exponentially time-weighted form. |
| Early game | Historical averages of gold, XP, and CS differentials at 10/15 minutes; first-blood and first-tower rates. |
| Objectives | Historical dragon, herald, and baron control fractions and first-objective rates. |
| Game control | Historical duration, average gold difference, and early-lead conversion where observed data supports it. |
| Side | Known current side and each team's prior blue/red-side performance. At inference an unknown side is an equal mixture of blue/red scenarios. |
| Context | Patch, tournament, stage, and best-of format when known at the cutoff. |
| Head-to-head | Prior meetings, recent head-to-head results, and rating/context information derived from eligible history. |
| Coverage | Prior observation counts and missing-history values expose sparse history. The prediction CLI rejects team names with no available historical maps. |

Defaults in `src/config.py` use a 20-map rolling window, a 90-day recent
window, a 30-day exponentially weighted form half-life, an Elo initial rating
of 1500, Elo K of 24, and an Elo inactivity half-life of 365 days. An observed
roster-ID change retains 75% of the existing deviation from initial Elo. These
are configurable modeling choices, not claims that the values are optimal;
tune them using training/validation history only.

Unknown historical statistics remain missing in the feature table. Numeric
imputation/scaling and categorical encoding are learned from the training
partition; unavailable metrics do not become fabricated source observations.
Missing-category handling supports unseen patches and competitions.

## Leakage prevention and forecast scope

Feature generation follows the historical timeline. A previous map is eligible
only when both `start_time < target_time` and `available_at < target_time`.
The target's result and statistics never update its own features. Games
starting simultaneously cannot see one another's results. Delayed publication
is respected even when a map has already finished. Shifting is implemented by
updating state only after historical availability, rather than by relying on
row order alone.

The same policy applies to Elo, form, head-to-head, and optional historical
statistics. Preprocessing is fitted on the training partition only. Training
labels must have become available before later forecast partitions begin.
Calibration uses a later, disjoint validation period. The final test period
is reserved for evaluation and must not select a model, calibration method,
hyperparameters, or features. Walk-forward folds refit using only information
available before each fold's forecast interval. No random train/test splitting
is used, and timestamp groups are not split across partitions.

This is an **individual-map** model. Before a later map in a series it may use
earlier maps if their results/statistics were already available at that map's
cutoff. That evaluation is not a forecast of all maps made before the series
began. For a before-series forecast, use one fixed pre-series information
cutoff and account explicitly for later unknown side/draft choices.

Historical training uses each map's start as its information cutoff. Calling
the resulting model pre-draft also requires verifying that the context
supplied at that cutoff was known before champion select. Supply `unknown`
for sides or omit roster/context values lacking that provenance. To backtest
predictions hours or days before a game, add archived decision timestamps and
as-of context snapshots; map-start data alone does not validate that setting.

## Commands

Get all supported options with `python main.py --help` or a subcommand's
`--help`.

Validate canonical real data and generate historical features:

```powershell
python main.py prepare --data data/raw
```

Train logistic regression and XGBoost with chronological train/validation/test
partitions; default output is `models/`:

```powershell
python main.py train --data data/raw --output models --calibration sigmoid
```

By default, chronological timestamp groups are divided approximately 60% for
training, 20% for calibration/validation, and 20% for heldout testing. For
explicit season boundaries, provide timezone-aware cutoffs:

```powershell
python main.py train --data data/raw --output models --train-end "2024-01-01T00:00:00Z" --validation-end "2025-01-01T00:00:00Z" --calibration sigmoid
```

Choose `sigmoid` (Platt-style), `isotonic`, or `none`. Isotonic needs a
sufficiently large validation sample to avoid overfitting. Compare calibrated
and uncalibrated results; calibration is not a guarantee of reliability under
future roster, patch, or tournament shifts. Random seeds and configuration
are recorded with the fitted artifacts for reproducibility. Training saves
`logistic_regression.joblib`, `xgboost.joblib`, `report.json`,
`test_predictions.csv`, and `calibration.png` in the output directory. Load
only your own trusted model artifacts; joblib uses Python serialization.

View the saved heldout evaluation report:

```powershell
python main.py evaluate
```

This reads `models/report.json`; it does not silently retrain or treat the
training set as a test set. Reports compare log loss, Brier score, ROC-AUC,
accuracy, calibration curves, a uniform 50% forecast, and—when available—the
standalone decaying-Elo forecast. The baselines are references, not models
selected on the test set. Lower log loss and Brier score are better.
Accuracy uses a 0.5 decision threshold; probability quality is the main goal.
ROC-AUC is undefined for a test slice containing only one class and is reported
as unavailable rather than invented.

Run expanding-history walk-forward evaluation:

```powershell
python main.py backtest --data data/raw --output artifacts/backtest --folds 3
```

Each fold has an expanding training window, a later calibration block, and
nonoverlapping test maps. The model is fixed within its test block, while
historical features update as earlier results become available. Output files
are `backtest_report.json`, `backtest_predictions.csv`, and
`backtest_calibration.png`.

Request a map prediction at an explicit information cutoff, using only context
that was known at that time:

```powershell
python main.py predict --model models/logistic_regression.joblib --data data/raw --team1 T1 --team2 "Gen.G" --at "2026-10-01T09:00:00Z" --side unknown --tournament LCK --stage playoffs --best-of 5
```

The timestamp above is illustrative, not a claimed scheduled match. Supply
the real decision time; include `--patch` when known, and use `--side blue` or
`--side red` only when team1's side was established by that time. A later
trained artifact is not valid for predicting an earlier historical cutoff.
Both requested team names must match the canonical history and have available
past maps; unknown aliases fail with an actionable error. Optional `--roster1`
and `--roster2` supply pre-match-known roster IDs. `--json` returns structured
output. `--mode post-draft` currently fails explicitly because genuine draft
features have not been implemented.

With `--side unknown`, the reported map probability is the arithmetic mean of
separate blue-side and red-side predictions, and both scenarios are included.
The 50/50 mixture is an explicit assumption about unknown side selection, not
a learned estimate. Its calibration is not separately established by the
known-side heldout evaluation.

The CLI returns both teams' complementary map probabilities and the model's
largest contributing factors/associations. Logistic coefficients can support
signed contributions; tree feature importance is global and must not be
presented as a signed local cause. Explanations describe model associations,
not causal effects or the complete arithmetic of a calibrated probability.
SHAP can be added later without changing the historical data contract.

## Series conversion

```python
from src.series import series_win_probability

series_win_probability(0.7, best_of=3)  # 0.784
series_win_probability(0.7, best_of=5)  # 0.83692
```

For odd `n = best_of`, the utility sums
`C(n, k) * p**k * (1-p)**(n-k)` over `k >= (n+1)/2`. This is equivalent to
stopping the series when either team wins enough maps; it does not multiply
the same probability by the number of maps. It assumes independent maps with
constant win probability. Real side selection, drafts, adaptation, and shared
team form can violate that assumption. These values are approximations, not
separately calibrated series forecasts.

## Limitations and next work

The fixture verifies program behavior only. A production model still needs
audited real history, source-specific conversion, sufficient heldout samples,
and sustained out-of-time validation across seasons and tournaments. Missing
or inaccurate publication times can defeat as-of correctness even when code
is chronological. A current source snapshot can contain retrospectively
corrected statistics; preserve timestamped versions before claiming strict
historical reproducibility.

Roster IDs trigger a simple Elo shrink when an identified roster changes.
This does not measure role quality, individual player skill, substitute
effects, or how many players changed. Team aliases must be resolved upstream.
Patch context and time decay do not automatically estimate a special
patch-change effect. Competition effects come from data rather than
hand-authored narratives about T1 or Worlds.

The feature pipeline is modular for adding role-level rolling player
statistics, player ratings, roster continuity, and champion proficiency when
reliable timestamped data exists. Planned post-draft features require
separate draft provenance and a distinct prediction cutoff. No player stats,
draft picks, bans, composition scores, or narrative buffs are fabricated in
this version. Future work can add source adapters, versioned as-of snapshots,
player/draft modules, SHAP explanations, and series models with varying map
probabilities. Any model-selection decisions belong in historical validation,
with untouched future periods kept for final evaluation.
