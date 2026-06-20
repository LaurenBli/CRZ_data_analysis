# 9ee_bus_exposure_robustness.py
"""
Bus CBD exposure robustness check.

This script tests whether the binary bus treatment definition
(any CBD stop vs no CBD stops) is too coarse.

Instead of treating every CBD-serving route equally, it uses the already
constructed share_cbd_stops variable and creates exposure bins:

    0%       = control routes
    0-10%    = low CBD exposure
    10-30%   = medium CBD exposure
    30%+     = high CBD exposure

For each nonzero exposure group, the script estimates a separate DiD
against the 0% control group.

Important interpretation
------------------------
This is NOT a stop-level ridership analysis. The underlying ridership data
remain route-hour level. This is a route-level exposure-intensity robustness
check based on the share of a route's stops classified as CBD in the MTA Bus
Stops reference file.

Input
-----
data/processed/bus_master_with_crz_groups.parquet

Required columns
----------------
transit_timestamp
bus_route
ridership
share_cbd_stops

Outputs
-------
data/processed/bus_exposure_did_panel.parquet
outputs/models/9ee_bus_exposure_robustness_results.txt
outputs/models/9ee_bus_exposure_robustness_key_results.csv
outputs/models/9ee_bus_exposure_route_summary.csv
outputs/models/9ee_bus_exposure_pretrend_summary.csv
"""

import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from regression_helpers import extract_terms


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

INPUT = "data/processed/bus_master_with_crz_groups.parquet"

OUT_PANEL = "data/processed/bus_exposure_did_panel.parquet"
OUT_RESULTS = "outputs/models/9ee_bus_exposure_robustness_results.txt"
OUT_KEY_RESULTS = "outputs/models/9ee_bus_exposure_robustness_key_results.csv"
OUT_ROUTE_SUMMARY = "outputs/models/9ee_bus_exposure_route_summary.csv"
OUT_PRETRENDS = "outputs/models/9ee_bus_exposure_pretrend_summary.csv"

os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def add_common_controls(df):
    df = df.copy()
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


def safe_reindex_summary(summary, expected_groups):
    """Reindex group summaries without failing when some bins are empty."""
    summary = summary.set_index("cbd_exposure_group")
    summary = summary.reindex(expected_groups)
    return summary.reset_index()


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

print("=" * 90)
print("Loading bus master with CBD exposure")
print("=" * 90)

if not os.path.exists(INPUT):
    raise FileNotFoundError(f"Could not find input file: {INPUT}")

df = pd.read_parquet(INPUT)

required = {
    "transit_timestamp",
    "bus_route",
    "ridership",
    "share_cbd_stops",
}

missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in {INPUT}: {missing}")

df = df.copy()

df["transit_timestamp"] = pd.to_datetime(df["transit_timestamp"])
df["bus_route"] = df["bus_route"].astype(str).str.strip().str.upper()
df["ridership"] = pd.to_numeric(df["ridership"], errors="coerce").fillna(0)
df["share_cbd_stops"] = pd.to_numeric(
    df["share_cbd_stops"],
    errors="coerce",
).fillna(0)

df = df[df["ridership"] >= 0].copy()

if "main_did_sample" in df.columns:
    df = df[df["main_did_sample"] == 1].copy()


# ---------------------------------------------------------------------
# Create exposure bins
# ---------------------------------------------------------------------

expected_groups = ["0%", "0-30%", "30%+"]

df["cbd_exposure_group"] = pd.cut(
    df["share_cbd_stops"],
    bins=[-0.000001, 0, 0.30, 1.0],
    labels=["0%", "0-30%", "30%+"],
    include_lowest=True,
)

df["cbd_exposure_group"] = df["cbd_exposure_group"].astype(str)

present_groups = [
    g for g in expected_groups
    if g in set(df["cbd_exposure_group"].dropna())
]

if "0%" not in present_groups:
    raise ValueError(
        "No 0% CBD exposure routes found. Cannot use 0% group as control."
    )


# ---------------------------------------------------------------------
# Route-level diagnostics
# ---------------------------------------------------------------------

route_cols = ["bus_route", "share_cbd_stops", "cbd_exposure_group"]
if "n_unique_stops" in df.columns:
    route_cols.append("n_unique_stops")

route_summary_base = df[route_cols].drop_duplicates().copy()

agg_spec = {
    "n_routes": ("bus_route", "nunique"),
    "mean_share_cbd_stops": ("share_cbd_stops", "mean"),
    "min_share_cbd_stops": ("share_cbd_stops", "min"),
    "max_share_cbd_stops": ("share_cbd_stops", "max"),
}

if "n_unique_stops" in route_summary_base.columns:
    agg_spec["mean_n_unique_stops"] = ("n_unique_stops", "mean")
    agg_spec["min_n_unique_stops"] = ("n_unique_stops", "min")
    agg_spec["max_n_unique_stops"] = ("n_unique_stops", "max")

route_summary = (
    route_summary_base
    .groupby("cbd_exposure_group", observed=True)
    .agg(**agg_spec)
    .reset_index()
)

route_summary = safe_reindex_summary(route_summary, expected_groups)
route_summary.to_csv(OUT_ROUTE_SUMMARY, index=False)


# ---------------------------------------------------------------------
# Build exposure-hour panel
# ---------------------------------------------------------------------

did = add_common_controls(df)

panel = (
    did.groupby(
        [
            "transit_timestamp",
            "cbd_exposure_group",
        ],
        as_index=False,
    )
    .agg(
        bus_ridership=("ridership", "sum"),
        n_routes=("bus_route", "nunique"),
        mean_share_cbd_stops=("share_cbd_stops", "mean"),
    )
)

panel = add_common_controls(panel)

panel["cbd_exposure_group"] = pd.Categorical(
    panel["cbd_exposure_group"],
    categories=expected_groups,
    ordered=True,
)

panel = (
    panel.sort_values(["transit_timestamp", "cbd_exposure_group"])
    .reset_index(drop=True)
)

panel.to_parquet(OUT_PANEL, index=False)


# ---------------------------------------------------------------------
# Pre-trend summary
# ---------------------------------------------------------------------

pre_summary = (
    panel[panel["post_congestion_pricing"] == 0]
    .groupby(
        [
            "year_month",
            "cbd_exposure_group",
        ],
        observed=True,
    )["bus_ridership"]
    .mean()
    .reset_index()
)

pre_pivot = (
    pre_summary
    .pivot(
        index="year_month",
        columns="cbd_exposure_group",
        values="bus_ridership",
    )
    .reset_index()
)

for g in ["0-30%", "30%+"]:
    if g in pre_pivot.columns and "0%" in pre_pivot.columns:
        pre_pivot[f"{g}_minus_0pct"] = pre_pivot[g] - pre_pivot["0%"]
        pre_pivot[f"{g}_over_0pct"] = (
            pre_pivot[g] / pre_pivot["0%"].replace(0, np.nan)
        )

pre_pivot.to_csv(OUT_PRETRENDS, index=False)


# ---------------------------------------------------------------------
# Run separate DiDs: each exposure bin vs 0%
# ---------------------------------------------------------------------

all_results = []

nonzero_groups = [g for g in ["0-30%", "30%+"] if g in present_groups]

if not nonzero_groups:
    raise ValueError("No nonzero CBD exposure groups found.")

for exposure_group in nonzero_groups:
    sub = panel[
        panel["cbd_exposure_group"].astype(str).isin(["0%", exposure_group])
    ].copy()

    sub["treated_group"] = (
        sub["cbd_exposure_group"].astype(str) == exposure_group
    ).astype(int)

    sub["comparison_group"] = f"{exposure_group} vs 0%"

    # One time index per hour, shared across both groups in the comparison.
    hour_index = (
        sub[["transit_timestamp"]]
        .drop_duplicates()
        .sort_values("transit_timestamp")
        .reset_index(drop=True)
    )

    hour_index["time_index"] = np.arange(len(hour_index))

    sub = sub.merge(
        hour_index,
        on="transit_timestamp",
        how="left",
    )

    sub["log_bus_ridership"] = np.log1p(sub["bus_ridership"])

    if sub["treated_group"].nunique() != 2:
        print(f"Skipping {exposure_group}: comparison does not contain both groups.")
        continue

    if sub["post_congestion_pricing"].nunique() != 2:
        print(f"Skipping {exposure_group}: comparison does not contain pre and post periods.")
        continue

    level_formula = """
    bus_ridership ~ post_congestion_pricing * treated_group
    + C(cbd_exposure_group)
    + C(hour)
    + C(day_of_week)
    + C(year_month)
    + holiday_flag
    + severe_weather_flag
    """

    log_formula = """
    log_bus_ridership ~ post_congestion_pricing * treated_group
    + C(cbd_exposure_group)
    + C(hour)
    + C(day_of_week)
    + C(year_month)
    + holiday_flag
    + severe_weather_flag
    """

    level_trend_formula = """
    bus_ridership ~ post_congestion_pricing * treated_group
    + C(cbd_exposure_group)
    + treated_group:time_index
    + C(hour)
    + C(day_of_week)
    + C(year_month)
    + holiday_flag
    + severe_weather_flag
    """

    log_trend_formula = """
    log_bus_ridership ~ post_congestion_pricing * treated_group
    + C(cbd_exposure_group)
    + treated_group:time_index
    + C(hour)
    + C(day_of_week)
    + C(year_month)
    + holiday_flag
    + severe_weather_flag
    """

    cov_kwds = {"groups": sub["date_cluster"]}

    level_model = smf.ols(level_formula, data=sub).fit(
        cov_type="cluster",
        cov_kwds=cov_kwds,
    )

    log_model = smf.ols(log_formula, data=sub).fit(
        cov_type="cluster",
        cov_kwds=cov_kwds,
    )

    level_trend_model = smf.ols(level_trend_formula, data=sub).fit(
        cov_type="cluster",
        cov_kwds=cov_kwds,
    )

    log_trend_model = smf.ols(log_trend_formula, data=sub).fit(
        cov_type="cluster",
        cov_kwds=cov_kwds,
    )

    term = "post_congestion_pricing:treated_group"

    results = pd.concat(
        [
            extract_terms(
                level_model,
                [term],
                f"Bus exposure DiD - level - {exposure_group} vs 0%",
            ),
            extract_terms(
                log_model,
                [term],
                f"Bus exposure DiD - log - {exposure_group} vs 0%",
            ),
            extract_terms(
                level_trend_model,
                [term],
                f"Bus exposure DiD - level trend - {exposure_group} vs 0%",
            ),
            extract_terms(
                log_trend_model,
                [term],
                f"Bus exposure DiD - log trend - {exposure_group} vs 0%",
            ),
        ],
        ignore_index=True,
    )

    results["exposure_group"] = exposure_group
    results["control_group"] = "0%"
    results["comparison"] = f"{exposure_group} vs 0%"
    results["n_rows"] = len(sub)
    results["n_hours"] = sub["transit_timestamp"].nunique()
    results["treated_routes_max"] = (
        sub.loc[sub["treated_group"] == 1, "n_routes"].max()
    )
    results["control_routes_max"] = (
        sub.loc[sub["treated_group"] == 0, "n_routes"].max()
    )

    all_results.append(results)


if not all_results:
    raise ValueError("No exposure DiD models were estimated.")

main_results = pd.concat(all_results, ignore_index=True)

log_rows = main_results["model"].str.contains("log", case=False, na=False)
main_results.loc[log_rows, "percent_effect"] = (
    100 * (np.exp(main_results.loc[log_rows, "coef"]) - 1)
)

main_results.to_csv(OUT_KEY_RESULTS, index=False)


# ---------------------------------------------------------------------
# Save readable output
# ---------------------------------------------------------------------

display_results = main_results.copy()

for col in ["coef", "std_err", "ci_low", "ci_high"]:
    if col in display_results.columns:
        display_results[col] = display_results[col].round(4)

if "p_value" in display_results.columns:
    display_results["p_value"] = display_results["p_value"].round(4)

if "r_squared" in display_results.columns:
    display_results["r_squared"] = display_results["r_squared"].round(4)

if "percent_effect" in display_results.columns:
    display_results["percent_effect"] = display_results["percent_effect"].round(2)

display_results = display_results.drop(columns=["term"], errors="ignore")
display_results = display_results.fillna("")

baseline_results = display_results[
    display_results["model"].str.contains(" - level - | - log - ", regex=True)
    & ~display_results["model"].str.contains("trend", case=False, na=False)
]

trend_results = display_results[
    display_results["model"].str.contains("trend", case=False, na=False)
]

with open(OUT_RESULTS, "w", encoding="utf-8") as f:
    f.write("=" * 90 + "\n")
    f.write("9ee Bus CBD exposure robustness results\n")
    f.write("=" * 90 + "\n\n")

    f.write("Input:\n")
    f.write(f"{INPUT}\n\n")

    f.write("Interpretation:\n")
    f.write(
        "This is a route-level exposure-intensity robustness check. "
        "Routes are grouped by the share of their stops classified as CBD in "
        "the MTA Bus Stops reference file. The outcome remains route-level "
        "bus ridership aggregated by hour and exposure group. This is not a "
        "stop-level ridership analysis.\n\n"
    )

    f.write("Exposure bins:\n")
    f.write("0%       = no CBD stops; control group\n")
    f.write("0-30%    = low/moderate CBD exposure\n")
    f.write("30%+     = high CBD exposure\n\n")

    f.write("Outputs:\n")
    f.write(f"Panel:          {OUT_PANEL}\n")
    f.write(f"Key results:    {OUT_KEY_RESULTS}\n")
    f.write(f"Route summary:  {OUT_ROUTE_SUMMARY}\n")
    f.write(f"Pretrends:      {OUT_PRETRENDS}\n\n")

    f.write("Specification:\n")
    f.write("bus_ridership ~ post_congestion_pricing * treated_group\n")
    f.write("+ exposure-group FE + hour FE + day-of-week FE + year-month FE\n")
    f.write("+ holiday_flag + severe_weather_flag\n\n")

    f.write("Robustness specification:\n")
    f.write("Baseline specification + treated_group:time_index\n\n")

    f.write("Standard errors:\n")
    f.write("Clustered by calendar date\n\n")

    f.write("-" * 90 + "\n")
    f.write("Route exposure summary\n")
    f.write("-" * 90 + "\n\n")
    f.write(route_summary.to_string(index=False))
    f.write("\n\n")

    f.write("-" * 90 + "\n")
    f.write("Pre-period monthly means by exposure group\n")
    f.write("-" * 90 + "\n\n")
    f.write(pre_pivot.to_string(index=False))
    f.write("\n\n")

    f.write("-" * 90 + "\n")
    f.write("Baseline DiD results\n")
    f.write("-" * 90 + "\n\n")
    f.write(baseline_results.to_string(index=False))
    f.write("\n\n")

    f.write("-" * 90 + "\n")
    f.write("Trend robustness DiD results\n")
    f.write("-" * 90 + "\n\n")
    f.write(trend_results.to_string(index=False))
    f.write("\n\n")


print("=" * 90)
print("9ee Bus CBD exposure robustness complete")
print("=" * 90)
print(f"Saved exposure panel to:   {OUT_PANEL}")
print(f"Saved key results to:      {OUT_KEY_RESULTS}")
print(f"Saved route summary to:    {OUT_ROUTE_SUMMARY}")
print(f"Saved pretrend summary to: {OUT_PRETRENDS}")
print(f"Saved readable results to: {OUT_RESULTS}")
print()
print("Route exposure summary:")
print(route_summary.to_string(index=False))
print()
print("Main results:")
print(main_results.to_string(index=False))
