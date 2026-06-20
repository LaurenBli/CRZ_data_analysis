# 9gd_run_did_forhire.py

import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from regression_helpers import extract_terms

# ---------------------------------------------------------------------
# 9gd For-Hire DiD
#
# TRUE for-hire DiD only.
#
# Expects:
#   data/processed/forhire_did_panel.parquet
#
# Required columns:
#   transit_timestamp
#   forhire_trips
#   treated_group
#   forhire_zone_group
#
# Treatment:
#   treated_group = 1 -> pickup starts in core CRZ
#   treated_group = 0 -> pickup starts outside CRZ
#
# Border pickup zones are excluded upstream in 9gc.
# ---------------------------------------------------------------------

FORHIRE_DID_PANEL = "data/processed/forhire_did_panel.parquet"

OUT_PANEL = "data/processed/forhire_did_panel_checked.parquet"
OUT_RESULTS = "outputs/models/9g_did_forhire_results.txt"
OUT_PRETRENDS = "outputs/models/9g_forhire_pretrend_summary.csv"
OUT_MAIN_RESULTS = "outputs/models/9g_forhire_key_results.csv"

os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)


def add_common_controls(df):
    df["transit_timestamp"] = pd.to_datetime(df["transit_timestamp"])
    df["date"] = pd.to_datetime(df["transit_timestamp"].dt.date)
    df["hour"] = df["transit_timestamp"].dt.hour
    df["day_of_week"] = df["transit_timestamp"].dt.day_name()
    df["year_month"] = df["transit_timestamp"].dt.to_period("M").astype(str)
    df["date_cluster"] = df["transit_timestamp"].dt.date.astype(str)

    policy_start = pd.Timestamp("2025-01-05 00:00:00")

    df["post_congestion_pricing"] = (
        df["transit_timestamp"] >= policy_start
    ).astype(int)

    for col in ["holiday_flag", "severe_weather_flag", "major_event_flag"]:
        if col not in df.columns:
            df[col] = 0

        df[col] = df[col].fillna(0).astype(int)

    return df


# ---------------------------------------------------------------------
# Load for-hire DiD data
# ---------------------------------------------------------------------

if not os.path.exists(FORHIRE_DID_PANEL):
    raise FileNotFoundError(
        f"Missing {FORHIRE_DID_PANEL}. "
        "Run 9gc_build_forhire_did_panel.py first."
    )

did = pd.read_parquet(FORHIRE_DID_PANEL)
did = add_common_controls(did)

required = {
    "transit_timestamp",
    "forhire_trips",
    "treated_group",
    "forhire_zone_group",
    "post_congestion_pricing",
    "hour",
    "day_of_week",
    "year_month",
    "date_cluster",
    "holiday_flag",
    "severe_weather_flag",
}

missing = required - set(did.columns)

if missing:
    raise ValueError(
        f"{FORHIRE_DID_PANEL} exists but is missing required columns: {missing}"
    )


# ---------------------------------------------------------------------
# Clean model data
# ---------------------------------------------------------------------

did["forhire_trips"] = pd.to_numeric(
    did["forhire_trips"],
    errors="coerce",
)

did["treated_group"] = did["treated_group"].astype(int)
did["post_congestion_pricing"] = did["post_congestion_pricing"].astype(int)

did = did.dropna(
    subset=[
        "forhire_trips",
        "treated_group",
        "forhire_zone_group",
    ]
).copy()

did = did[did["forhire_trips"] >= 0].copy()

if did["forhire_zone_group"].nunique() != 2:
    raise ValueError(
        "Expected exactly two forhire_zone_group values."
    )

if did["treated_group"].isna().any():
    raise ValueError("Missing treated_group values.")

if did["transit_timestamp"].isna().any():
    raise ValueError("Missing transit_timestamp values.")

did["forhire_zone_group"] = pd.Categorical(
    did["forhire_zone_group"],
    categories=[
        "outside_crz_pickup",
        "core_crz_pickup",
    ],
    ordered=True,
)

did["log_forhire_trips"] = np.log1p(
    did["forhire_trips"]
)

# One time index per hour, shared across treated/control groups.
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


# ---------------------------------------------------------------------
# Pre-trend summary
# ---------------------------------------------------------------------

pre_summary = (
    did[did["post_congestion_pricing"] == 0]
    .groupby(
        [
            "year_month",
            "forhire_zone_group",
        ],
        observed=True,
    )["forhire_trips"]
    .mean()
    .reset_index()
)

pre_summary_pivot = (
    pre_summary
    .pivot(
        index="year_month",
        columns="forhire_zone_group",
        values="forhire_trips",
    )
    .reset_index()
)

if {"core_crz_pickup", "outside_crz_pickup"}.issubset(pre_summary_pivot.columns):
    pre_summary_pivot["treated_minus_control"] = (
        pre_summary_pivot["core_crz_pickup"]
        - pre_summary_pivot["outside_crz_pickup"]
    )

    pre_summary_pivot["treated_over_control"] = (
        pre_summary_pivot["core_crz_pickup"]
        / pre_summary_pivot["outside_crz_pickup"].replace(0, np.nan)
    )

pre_summary_pivot.to_csv(
    OUT_PRETRENDS,
    index=False,
)


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

level_formula = """
forhire_trips ~ post_congestion_pricing * treated_group
+ C(forhire_zone_group)
+ C(hour)
+ C(day_of_week)
+ C(year_month)
+ holiday_flag
+ severe_weather_flag
"""

log_formula = """
log_forhire_trips ~ post_congestion_pricing * treated_group
+ C(forhire_zone_group)
+ C(hour)
+ C(day_of_week)
+ C(year_month)
+ holiday_flag
+ severe_weather_flag
"""

level_trend_formula = """
forhire_trips ~ post_congestion_pricing * treated_group
+ C(forhire_zone_group)
+ treated_group:time_index
+ C(hour)
+ C(day_of_week)
+ C(year_month)
+ holiday_flag
+ severe_weather_flag
"""

log_trend_formula = """
log_forhire_trips ~ post_congestion_pricing * treated_group
+ C(forhire_zone_group)
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


# ---------------------------------------------------------------------
# Extract key coefficients only
# ---------------------------------------------------------------------

term = "post_congestion_pricing:treated_group"

main_results = pd.concat(
    [
        extract_terms(level_model, [term], "For-Hire DiD - level"),
        extract_terms(log_model, [term], "For-Hire DiD - log"),
        extract_terms(level_trend_model, [term], "For-Hire DiD - level trend"),
        extract_terms(log_trend_model, [term], "For-Hire DiD - log trend"),
    ],
    ignore_index=True,
)

log_rows = main_results["model"].str.contains(
    "log",
    case=False,
    na=False,
)

main_results.loc[log_rows, "approx_percent_effect"] = (
    100 * main_results.loc[log_rows, "coef"]
)

main_results.loc[log_rows, "exact_percent_effect"] = (
    100 * (np.exp(main_results.loc[log_rows, "coef"]) - 1)
)

main_results.to_csv(
    OUT_MAIN_RESULTS,
    index=False,
)


# ---------------------------------------------------------------------
# Optional provider-specific descriptive summaries
# ---------------------------------------------------------------------

provider_cols = [
    "uber_trips",
    "lyft_trips",
]

provider_prepost = []

for col in provider_cols:
    if col in did.columns:
        tmp = (
            did.groupby(
                [
                    "post_congestion_pricing",
                    "forhire_zone_group",
                ],
                observed=True,
            )[col]
            .mean()
            .reset_index()
        )

        tmp["provider_metric"] = col
        provider_prepost.append(tmp)

if provider_prepost:
    provider_prepost = pd.concat(
        provider_prepost,
        ignore_index=True,
    )
else:
    provider_prepost = pd.DataFrame()


# ---------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------

with open(OUT_RESULTS, "w", encoding="utf-8") as f:
    f.write("=" * 90 + "\n")
    f.write("9g For-Hire DiD Results\n")
    f.write("=" * 90 + "\n\n")

    f.write("Input source:\n")
    f.write(f"{FORHIRE_DID_PANEL}\n\n")

    f.write("Output checked panel:\n")
    f.write(f"{OUT_PANEL}\n\n")

    f.write("Pre-trend summary CSV:\n")
    f.write(f"{OUT_PRETRENDS}\n\n")

    f.write("Key results CSV:\n")
    f.write(f"{OUT_MAIN_RESULTS}\n\n")

    f.write("Specification:\n")
    f.write("forhire_trips ~ post_congestion_pricing * treated_group\n")
    f.write("+ forhire-zone-group FE + hour FE + day-of-week FE + year-month FE\n")
    f.write("+ holiday_flag + severe_weather_flag\n\n")

    f.write("Robustness specification:\n")
    f.write("Baseline specification + treated_group:time_index\n\n")

    f.write("Standard errors:\n")
    f.write("Clustered by calendar date\n\n")

    f.write("Identification:\n")
    f.write(
        "Treatment group = FHV pickups inside the CRZ.\n"
    )
    f.write(
        "Control group = FHV pickups outside the CRZ.\n"
    )
    f.write(
        "Pickup-border zones are excluded upstream from the main DiD sample.\n\n"
    )

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

    if not provider_prepost.empty:
        f.write("-" * 90 + "\n")
        f.write("Provider-specific pre/post hourly means\n")
        f.write("-" * 90 + "\n\n")
        f.write(provider_prepost.to_string(index=False))
        f.write("\n\n")

    display_results = main_results.copy()

    for col in ["coef", "std_err", "ci_low", "ci_high"]:
        display_results[col] = display_results[col].round(4)

    display_results["p_value"] = display_results["p_value"].round(4)
    display_results["r_squared"] = display_results["r_squared"].round(4)

    for col in ["approx_percent_effect", "exact_percent_effect"]:
        if col in display_results.columns:
            display_results[col] = display_results[col].round(2)

    display_results = display_results.drop(
        columns=[
            "term",
        ],
        errors="ignore",
    )

    display_results = display_results.fillna("")

    baseline_results = display_results[
        display_results["model"].isin(
            [
                "For-Hire DiD - level",
                "For-Hire DiD - log",
            ]
        )
    ]

    trend_results = display_results[
        display_results["model"].isin(
            [
                "For-Hire DiD - level trend",
                "For-Hire DiD - log trend",
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


print("=" * 90)
print("9g For-Hire DiD complete")
print("=" * 90)
print(f"Saved checked panel to: {OUT_PANEL}")
print(f"Saved pre-trend summary to: {OUT_PRETRENDS}")
print(f"Saved key results to: {OUT_MAIN_RESULTS}")
print(f"Saved results to: {OUT_RESULTS}")
print()
print("Main Results")
print(main_results.to_string(index=False))
