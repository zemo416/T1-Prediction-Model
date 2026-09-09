# Canonical historical game data

The first version accepts canonical CSV files, not arbitrary source exports.
Place real files under `data/raw/`, optionally in subdirectories such as
`data/raw/2024/lck.csv`. Every `*.csv` below the selected data directory is read
recursively. Preserve original downloads separately and convert copies into
this schema; the pipeline does not modify raw input files.

Each map has exactly two rows, one from each team's perspective. A series of
three maps therefore has six rows and three distinct `game_id` values. Do not
put a series-level win/loss label in this table.

## Required columns

| Column | Type | Meaning |
| --- | --- | --- |
| `game_id` | Nonempty string | Globally unique map ID, stable across files and refreshes. Exactly two rows share it. |
| `start_time` | Timezone-aware ISO 8601 timestamp | Actual map start, for example `2024-03-01T09:00:00Z`. UTC is recommended; explicit offsets are accepted and normalized. |
| `available_at` | Timezone-aware ISO 8601 timestamp | The later of completion time and the time the result/statistics used in the row became available. Must be strictly after `start_time`. |
| `team` | Nonempty string | Canonical team name for this row. Resolve historical aliases during source conversion. |
| `opponent` | Nonempty string | Canonical opposing team name. Must match the other row's `team`. |
| `side` | `blue`, `red`, or `unknown` | This team's side. A pair must be blue/red or unknown/unknown. |
| `win` | Integer `0` or `1` | This team's map result. The two rows must have opposite labels. |
| `tournament` | Nonempty string | Competition identifier, consistently defined across seasons. |
| `is_synthetic` | Boolean | Explicit `false` for verified real data; `true` only for test fixtures. |

Both rows must agree on `start_time`, `available_at`, `tournament`,
`is_synthetic`, and any supplied shared context (`patch`, `stage`, `best_of`).
Their teams must be distinct and reciprocal. Duplicate map/team rows and
conflicting map identifiers across files are errors, not independent samples.

Do not infer `available_at` from `start_time`, a season CSV download time, or a
fixed game duration. Historical publication/ingestion records are ideal. If
only an estimated time can be established, document the conservative delay and
validate it against the intended prediction cutoff. If no defensible
availability timestamp is available, that source is not ready for this
pipeline. A single row timestamp covers every historical statistic in that
row: a late correction must not silently appear in an earlier snapshot. The
v1 format does not implement bitemporal correction/version history.

## Optional context known before the target map

| Column | Type | Meaning |
| --- | --- | --- |
| `patch` | String | Patch version, such as `14.1`. Unknown values may be blank. |
| `stage` | String | Consistent event stage, such as `regular_season`, `playoffs`, or `international`. |
| `best_of` | Integer `1`, `3`, or `5` | Series format; each row still represents one map. |
| `roster_id` | String | Stable identifier for this team's announced roster, if established before the prediction cutoff. |

An absent or blank optional value is missing, not zero. A roster inferred from
the players who ultimately appeared in the completed map is not evidence that
the roster was known before champion select. Retain provenance and omit
`roster_id` when its pre-match availability cannot be established. The same
rule applies to side choice and other context: use `unknown` or leave the field
blank if it was not known at the prediction time being evaluated.

The prediction CLI's `--at` is the feature cutoff for the requested map. The
historical training table uses each map's start as its cutoff and therefore
assumes included context was already available at the earlier pre-draft
decision point. Backtests at a much earlier horizon, or before a whole series,
need explicit archived decision-time snapshots and a corresponding adapter;
this table alone cannot prove that forecast setting.

## Optional statistics from completed historical maps

These columns describe the map on their row. They become usable only for later
target maps whose cutoff is strictly greater than that map's `available_at`.
No column below is used directly from the target map.

| Columns | Unit and interpretation |
| --- | --- |
| `gold_diff_10`, `gold_diff_15` | Team gold minus opponent gold at minute 10/15. |
| `xp_diff_10`, `xp_diff_15` | Team XP minus opponent XP at minute 10/15. |
| `cs_diff_10`, `cs_diff_15` | Team CS minus opponent CS at minute 10/15. |
| `first_blood`, `first_tower` | Binary indicator that this team secured the first event. |
| `dragon_control`, `herald_control`, `baron_control` | Fraction of that objective type secured by this team, in `[0, 1]`. |
| `first_dragon`, `first_herald`, `first_baron` | Binary indicator that this team secured the first objective. |
| `duration_seconds` | Completed map duration in seconds; positive when supplied. |
| `avg_gold_diff` | Time-average team gold minus opponent gold over the completed map. |

Use signed team perspective for differentials; paired values must negate
each other. Paired rate/first-event values must complement to one, and duration
must be the same for both rows. A paired statistic must be observed on both
team rows or missing on both. These invariants are validated. Use empty fields for absent or
inapplicable measurements, including an objective type that never spawned or
was never taken. Do not replace missing measurements with invented averages or
impute entire unobserved columns in the source file. A converter must resolve
source definitions, objective denominators, early-ended games, remake policy,
and patch-specific metric availability consistently and document those choices.

The exact optional statistic names are centralized in `src/config.py`.
Extra player or draft columns do not automatically become model features.
Adding them requires a documented schema, time-availability policy, feature
implementation, and leakage tests.

## Example and source preparation

The committed `tests/fixtures/synthetic_games.csv` is a 96-map, 192-row fixture
containing only fictional teams named `Fixture Alpha`, `Fixture Beta`,
`Fixture Gamma`, and `Fixture Delta`. Every row is explicitly synthetic. Its
values and timestamps are for pipeline verification only and must not be used
as real esports observations or performance evidence.

For real use, provide broad professional map history across recent seasons:
all teams in LCK, MSI, and Worlds, and other regions where possible to estimate
the strength of international opponents. At minimum, provide every required
column above. The optional statistics improve coverage of early-game and
objective features; missing values remain visible in model preprocessing.

[Oracle's Elixir downloads](https://oracleselixir.com/tools/downloads) is a
candidate historical-data source. Its native exports require a separately
reviewed conversion adapter: v1 does not promise direct Oracle's Elixir format
compatibility. In particular, establish true map identifiers, team-level rows,
team aliases, timestamp meaning, pre-match context provenance, and the time
the result/statistics became available. Do not rename its date column to
`available_at` without evidence. Keep original exports and conversion notes so
that the resulting canonical files can be audited.

After conversion, use `python main.py prepare --data data/raw` to validate the
data and create the derived feature table in `data/processed/`. Any rejected
input must be corrected at the conversion layer without silently rewriting
the original download.
