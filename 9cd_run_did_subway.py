import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from regression_helpers import extract_terms

HOURLY_PANEL = "data/processed/analysis_panel_with_event_flags.parquet"
SUBWAY_DID_PANEL = "data/processed/subway_did_panel.parquet"

OUT_PANEL = "data/processed/subway_did_panel_checked.parquet"
OUT_RESULTS = "outputs/models/9c_did_subway_results.txt"
OUT_PRETRENDS = "outputs/models/9c_subway_pretrend_summary.csv"
OUT_MAIN_RESULTS = "outputs/models/9c_subway_key_results.csv"

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


if os.path.exists(SUBWAY_DID_PANEL):
    did = pd.read_parquet(SUBWAY_DID_PANEL)
    did = add_common_controls(did)

    required = {
        "transit_timestamp",
        "subway_ridership",
        "treated_group",
        "subway_station_group",
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
            f"{SUBWAY_DID_PANEL} exists but is missing required columns: {missing}"
        )

elif os.path.exists(HOURLY_PANEL):
    df = pd.read_parquet(HOURLY_PANEL)
    df = add_common_controls(df)

    wide_required = {
        "subway_ridership_treated",
        "subway_ridership_control",
    }

    if not wide_required.issubset(df.columns):
        raise ValueError(
            "\nCannot run true subway DiD yet.\n\n"
            "Missing subway treated/control geography.\n\n"
            "Create either:\n"
            "  data/processed/subway_did_panel.parquet\n"
            "with columns:\n"
            "  transit_timestamp, subway_ridership, treated_group, subway_station_group\n\n"
            "or add wide columns to the hourly panel:\n"
            "  subway_ridership_treated, subway_ridership_control\n\n"
            "Do NOT use aggregate subway_ridership for DiD."
        )

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

    treated = df[base_cols + ["subway_ridership_treated"]].copy()
    treated = treated.rename(
        columns={"subway_ridership_treated": "subway_ridership"}
    )
    treated["treated_group"] = 1
    treated["subway_station_group"] = "core_crz"

    control = df[base_cols + ["subway_ridership_control"]].copy()
    control = control.rename(
        columns={"subway_ridership_control": "subway_ridership"}
    )
    control["treated_group"] = 0
    control["subway_station_group"] = "outside_crz"

    did = pd.concat([treated, control], ignore_index=True)

else:
    raise FileNotFoundError(
        f"Neither {SUBWAY_DID_PANEL} nor {HOURLY_PANEL} was found."
    )


# ---------------------------------------------------------------------
# Clean model data
# ---------------------------------------------------------------------

did["subway_ridership"] = pd.to_numeric(
    did["subway_ridership"],
    errors="coerce",
)

did["treated_group"] = did["treated_group"].astype(int)
did["post_congestion_pricing"] = did["post_congestion_pricing"].astype(int)

did = did.dropna(
    subset=[
        "subway_ridership",
        "treated_group",
        "subway_station_group",
    ]
).copy()

did = did[did["subway_ridership"] >= 0].copy()

if did["treated_group"].isna().any():
    raise ValueError("Missing treated_group values.")

if did["transit_timestamp"].isna().any():
    raise ValueError("Missing transit_timestamp values.")

did["subway_station_group"] = pd.Categorical(
    did["subway_station_group"],
    categories=["outside_crz", "core_crz"],
    ordered=True,
)

did["log_subway_ridership"] = np.log1p(did["subway_ridership"])

hour_index = (
    did[["transit_timestamp"]]
    .drop_duplicates()
    .sort_values("transit_timestamp")
    .reset_index(drop=True)
)

hour_index["time_index"] = np.arange(len(hour_index))

did = did.merge(hour_index, on="transit_timestamp", how="left")

did.to_parquet(OUT_PANEL, index=False)


# ---------------------------------------------------------------------
# Pre-trend summary
# ---------------------------------------------------------------------

pre_summary = (
    did[did["post_congestion_pricing"] == 0]
    .groupby(["year_month", "subway_station_group"], observed=True)["subway_ridership"]
    .mean()
    .reset_index()
)

pre_summary_pivot = (
    pre_summary
    .pivot(
        index="year_month",
        columns="subway_station_group",
        values="subway_ridership",
    )
    .reset_index()
)

if {"core_crz", "outside_crz"}.issubset(pre_summary_pivot.columns):
    pre_summary_pivot["treated_minus_control"] = (
        pre_summary_pivot["core_crz"] - pre_summary_pivot["outside_crz"]
    )

    pre_summary_pivot["treated_over_control"] = (
        pre_summary_pivot["core_crz"]
        / pre_summary_pivot["outside_crz"].replace(0, np.nan)
    )

pre_summary_pivot.to_csv(OUT_PRETRENDS, index=False)


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

level_formula = """
subway_ridership ~ post_congestion_pricing * treated_group
+ C(subway_station_group)
+ C(hour)
+ C(day_of_week)
+ C(year_month)
+ holiday_flag
+ severe_weather_flag
"""

log_formula = """
log_subway_ridership ~ post_congestion_pricing * treated_group
+ C(subway_station_group)
+ C(hour)
+ C(day_of_week)
+ C(year_month)
+ holiday_flag
+ severe_weather_flag
"""

level_trend_formula = """
subway_ridership ~ post_congestion_pricing * treated_group
+ C(subway_station_group)
+ treated_group:time_index
+ C(hour)
+ C(day_of_week)
+ C(year_month)
+ holiday_flag
+ severe_weather_flag
"""

log_trend_formula = """
log_subway_ridership ~ post_congestion_pricing * treated_group
+ C(subway_station_group)
+ treated_group:time_index
+ C(hour)
+ C(day_of_week)
+ C(year_month)
+ holiday_flag
+ severe_weather_flag
"""

level_model = smf.ols(level_formula, data=did).fit(
    cov_type="cluster",
    cov_kwds={"groups": did["date_cluster"]},
)

log_model = smf.ols(log_formula, data=did).fit(
    cov_type="cluster",
    cov_kwds={"groups": did["date_cluster"]},
)

level_trend_model = smf.ols(level_trend_formula, data=did).fit(
    cov_type="cluster",
    cov_kwds={"groups": did["date_cluster"]},
)

log_trend_model = smf.ols(log_trend_formula, data=did).fit(
    cov_type="cluster",
    cov_kwds={"groups": did["date_cluster"]},
)


# ---------------------------------------------------------------------
# Extract key coefficients only
# ---------------------------------------------------------------------

term = "post_congestion_pricing:treated_group"

main_results = pd.concat(
    [
        extract_terms(level_model, [term], "Subway DiD - level"),
        extract_terms(log_model, [term], "Subway DiD - log"),
        extract_terms(level_trend_model, [term], "Subway DiD - level trend"),
        extract_terms(log_trend_model, [term], "Subway DiD - log trend"),
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


# ---------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------

with open(OUT_RESULTS, "w", encoding="utf-8") as f:
    f.write("=" * 90 + "\n")
    f.write("9c Subway DiD Results\n")
    f.write("=" * 90 + "\n\n")

    f.write("Input source:\n")
    if os.path.exists(SUBWAY_DID_PANEL):
        f.write(f"{SUBWAY_DID_PANEL}\n\n")
    else:
        f.write(f"{HOURLY_PANEL} with subway_ridership_treated/control\n\n")

    f.write("Output checked panel:\n")
    f.write(f"{OUT_PANEL}\n\n")

    f.write("Pre-trend summary CSV:\n")
    f.write(f"{OUT_PRETRENDS}\n\n")

    f.write("Key results CSV:\n")
    f.write(f"{OUT_MAIN_RESULTS}\n\n")

    f.write("Specification:\n")
    f.write("subway_ridership ~ post_congestion_pricing * treated_group\n")
    f.write("+ subway-station-group FE + hour FE + day-of-week FE + year-month FE\n")
    f.write("+ holiday_flag + severe_weather_flag\n\n")

    f.write("Robustness specification:\n")
    f.write("Baseline specification + treated_group:time_index\n\n")

    f.write("Standard errors:\n")
    f.write("Clustered by calendar date\n\n")

    f.write("Identification note:\n")
    f.write(
        "This is valid as DiD only if treated_group reflects a real "
        "station/geographic treatment-control mapping, not aggregate subway ridership.\n\n"
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

    display_results = main_results.copy()

    for col in ["coef", "std_err", "ci_low", "ci_high"]:
        display_results[col] = display_results[col].round(4)

    display_results["p_value"] = display_results["p_value"].round(4)
    display_results["r_squared"] = display_results["r_squared"].round(4)

    if "percent_effect" in display_results.columns:
        display_results["percent_effect"] = (
            display_results["percent_effect"].round(2)
        )

    display_results = display_results.drop(columns=["term"], errors="ignore")
    display_results = display_results.fillna("")

    baseline_results = display_results[
        display_results["model"].isin(
            ["Subway DiD - level", "Subway DiD - log"]
        )
    ]

    trend_results = display_results[
        display_results["model"].isin(
            ["Subway DiD - level trend", "Subway DiD - log trend"]
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
print("9c Subway DiD complete")
print("=" * 90)
print(f"Saved checked panel to: {OUT_PANEL}")
print(f"Saved pre-trend summary to: {OUT_PRETRENDS}")
print(f"Saved key results to: {OUT_MAIN_RESULTS}")
print(f"Saved results to: {OUT_RESULTS}")
print()
print("Main Results")
print(main_results.to_string(index=False))

