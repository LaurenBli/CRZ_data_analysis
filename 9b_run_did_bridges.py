import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from regression_helpers import extract_terms

INPUT = "data/processed/analysis_panel_with_event_flags.parquet"
OUT_PANEL = "data/processed/bridge_did_panel.parquet"
OUT_RESULTS = "outputs/models/9b_did_bridges_results.txt"
OUT_PRETRENDS = "outputs/models/9b_bridge_pretrend_summary.csv"
OUT_MAIN_RESULTS = "outputs/models/9b_bridge_key_results.csv"

os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)

# ------------------------------------------------------------
# Load hourly master panel
# ------------------------------------------------------------

df = pd.read_parquet(INPUT)

df["transit_timestamp"] = pd.to_datetime(df["transit_timestamp"])
df["date"] = pd.to_datetime(df["date"])
df["year_month"] = df["transit_timestamp"].dt.to_period("M").astype(str)
df["date_cluster"] = df["transit_timestamp"].dt.date.astype(str)

# Ensure numeric / int flags
for col in ["post_congestion_pricing", "holiday_flag", "severe_weather_flag"]:
    if col not in df.columns:
        raise ValueError(f"Missing required flag column: {col}")

    df[col] = df[col].fillna(0).astype(int)

required_bridge_cols = [
    "bridge_traffic_treated",
    "bridge_traffic_control",
]

missing_bridge_cols = [
    col for col in required_bridge_cols
    if col not in df.columns
]

if missing_bridge_cols:
    raise ValueError(
        f"Missing required bridge columns: {missing_bridge_cols}"
    )

# ------------------------------------------------------------
# Build long treated/control bridge panel
# ------------------------------------------------------------

base_cols = [
    "transit_timestamp",
    "date",
    "hour",
    "day_of_week",
    "year_month",
    "date_cluster",
    "post_congestion_pricing",
    "holiday_flag",
    "severe_weather_flag",
]

treated = df[
    base_cols
    + [
        "bridge_traffic_treated",
    ]
].copy()

treated = treated.rename(
    columns={
        "bridge_traffic_treated": "bridge_traffic",
    }
)

treated["treated_group"] = 1
treated["bridge_group"] = "treated"

control = df[
    base_cols
    + [
        "bridge_traffic_control",
    ]
].copy()

control = control.rename(
    columns={
        "bridge_traffic_control": "bridge_traffic",
    }
)

control["treated_group"] = 0
control["bridge_group"] = "control"

did = pd.concat(
    [
        treated,
        control,
    ],
    ignore_index=True,
)

did["bridge_traffic"] = pd.to_numeric(
    did["bridge_traffic"],
    errors="coerce",
)

did = did.dropna(
    subset=[
        "bridge_traffic",
        "treated_group",
        "bridge_group",
    ]
).copy()

did = did[did["bridge_traffic"] >= 0].copy()

if did["bridge_group"].nunique() != 2:
    raise ValueError(
        "Expected exactly two bridge groups: treated and control"
    )

if did["treated_group"].isna().any():
    raise ValueError("Missing treated_group values")

if did["transit_timestamp"].isna().any():
    raise ValueError("Missing transit_timestamp values")

did["bridge_group"] = pd.Categorical(
    did["bridge_group"],
    categories=[
        "control",
        "treated",
    ],
    ordered=True,
)

did["log_bridge_traffic"] = np.log1p(
    did["bridge_traffic"]
)

# Time index for robustness trend model.
# One value per hour, shared by treated/control rows.
hour_index = (
    did[["transit_timestamp"]]
    .drop_duplicates()
    .sort_values("transit_timestamp")
    .reset_index(drop=True)
)

hour_index["time_index"] = np.arange(
    len(hour_index)
)

did = did.merge(
    hour_index,
    on="transit_timestamp",
    how="left",
)

did.to_parquet(
    OUT_PANEL,
    index=False,
)

# ------------------------------------------------------------
# Pre-trend summaries
# ------------------------------------------------------------

pre_summary = (
    did[did["post_congestion_pricing"] == 0]
    .groupby(
        [
            "year_month",
            "bridge_group",
        ],
        observed=True,
    )["bridge_traffic"]
    .mean()
    .reset_index()
)

pre_summary_pivot = (
    pre_summary
    .pivot(
        index="year_month",
        columns="bridge_group",
        values="bridge_traffic",
    )
    .reset_index()
)

if {"treated", "control"}.issubset(pre_summary_pivot.columns):
    pre_summary_pivot["treated_minus_control"] = (
        pre_summary_pivot["treated"]
        - pre_summary_pivot["control"]
    )

    pre_summary_pivot["treated_over_control"] = (
        pre_summary_pivot["treated"]
        / pre_summary_pivot["control"].replace(0, np.nan)
    )

pre_summary_pivot.to_csv(
    OUT_PRETRENDS,
    index=False,
)

# ------------------------------------------------------------
# DiD models
# ------------------------------------------------------------

level_formula = """
bridge_traffic ~ post_congestion_pricing * treated_group
+ C(bridge_group)
+ C(hour)
+ C(day_of_week)
+ C(year_month)
+ holiday_flag
+ severe_weather_flag
"""

log_formula = """
log_bridge_traffic ~ post_congestion_pricing * treated_group
+ C(bridge_group)
+ C(hour)
+ C(day_of_week)
+ C(year_month)
+ holiday_flag
+ severe_weather_flag
"""


level_trend_formula = """
bridge_traffic ~ post_congestion_pricing * treated_group
+ C(bridge_group)
+ treated_group:time_index
+ C(hour)
+ C(day_of_week)
+ C(year_month)
+ holiday_flag
+ severe_weather_flag
"""

log_trend_formula = """
log_bridge_traffic ~ post_congestion_pricing * treated_group
+ C(bridge_group)
+ treated_group:time_index
+ C(hour)
+ C(day_of_week)
+ C(year_month)
+ holiday_flag
+ severe_weather_flag
"""

level_model = smf.ols(
    level_formula,
    data=did,
).fit(
    cov_type="cluster",
    cov_kwds={
        "groups": did["date_cluster"],
    },
)

log_model = smf.ols(
    log_formula,
    data=did,
).fit(
    cov_type="cluster",
    cov_kwds={
        "groups": did["date_cluster"],
    },
)

level_trend_model = smf.ols(
    level_trend_formula,
    data=did,
).fit(
    cov_type="cluster",
    cov_kwds={
        "groups": did["date_cluster"],
    },
)

log_trend_model = smf.ols(
    log_trend_formula,
    data=did,
).fit(
    cov_type="cluster",
    cov_kwds={
        "groups": did["date_cluster"],
    },
)

# ------------------------------------------------------------
# Extract key coefficients only
# ------------------------------------------------------------

term = "post_congestion_pricing:treated_group"

main_results = pd.concat(
    [
        extract_terms(level_model, [term], "Bridge DiD - level"),
        extract_terms(log_model, [term], "Bridge DiD - log"),
        extract_terms(level_trend_model, [term], "Bridge DiD - level trend"),
        extract_terms(log_trend_model, [term], "Bridge DiD - log trend"),
    ],
    ignore_index=True,
)

log_rows = main_results["model"].str.contains(
    "log",
    case=False,
    na=False,
)

main_results.loc[log_rows, "percent_effect"] = (
    100 * (np.exp(main_results.loc[log_rows, "coef"]) - 1)
)

main_results.to_csv(OUT_MAIN_RESULTS, index=False)

# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

with open(OUT_RESULTS, "w", encoding="utf-8") as f:
    f.write("=" * 90 + "\n")
    f.write("9b Bridge DiD Results\n")
    f.write("=" * 90 + "\n\n")

    f.write("Input file:\n")
    f.write(f"{INPUT}\n\n")

    f.write("Output panel:\n")
    f.write(f"{OUT_PANEL}\n\n")

    f.write("Pre-trend summary CSV:\n")
    f.write(f"{OUT_PRETRENDS}\n\n")

    f.write("Specification:\n")
    f.write("Outcome ~ post_congestion_pricing * treated_group\n")
    f.write("+ bridge-group FE + hour FE + day-of-week FE + year-month FE\n")
    f.write("+ holiday_flag + severe_weather_flag\n\n")

    f.write("Robustness specification:\n")
    f.write("Baseline specification + treated_group:time_index\n\n")

    f.write("Standard errors:\n")
    f.write("Clustered by calendar date\n\n")

    f.write("Sample:\n")
    f.write(
        f"{did['transit_timestamp'].min()} "
        f"to {did['transit_timestamp'].max()}\n"
    )
    f.write(f"Rows: {len(did):,}\n\n")

    f.write("-" * 90 + "\n")
    f.write("Pre-period monthly treated/control means\n")
    f.write("-" * 90 + "\n\n")
    f.write(pre_summary_pivot.to_string(index=False))
    f.write("\n\n")

    # --------------------------------------------------------
    # Main Results
    # --------------------------------------------------------

    display_results = main_results.copy()

    for col in ["coef", "std_err", "ci_low", "ci_high"]:
        display_results[col] = display_results[col].round(4)

    display_results["p_value"] = display_results["p_value"].round(4)
    display_results["r_squared"] = display_results["r_squared"].round(4)

    if "percent_effect" in display_results.columns:
        display_results["percent_effect"] = (
            display_results["percent_effect"].round(2)
        )

    display_results = display_results.fillna("")

    baseline_results = display_results[
        display_results["model"].isin(
            [
                "Bridge DiD - level",
                "Bridge DiD - log",
            ]
        )
    ]

    trend_results = display_results[
        display_results["model"].isin(
            [
                "Bridge DiD - level trend",
                "Bridge DiD - log trend",
            ]
        )
    ]

    f.write("-" * 90 + "\n")
    f.write("Main Results\n")
    f.write("-" * 90 + "\n\n")

    f.write("Target coefficient:\n")
    f.write(f"{term}\n\n")

    f.write("Baseline Models\n")
    f.write("-" * 15 + "\n")
    f.write(baseline_results.to_string(index=False))
    f.write("\n\n")

    f.write("Trend Robustness Models\n")
    f.write("-" * 23 + "\n")
    f.write(trend_results.to_string(index=False))
    f.write("\n\n")