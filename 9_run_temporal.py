import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# ---------------------------------------------------------------------
# 9 CRZ temporal-effects model
#
# Purpose:
#   Estimate interpretable temporal structure in post-policy CRZ entries.
#
# This is NOT a causal DiD.
# It explains how CRZ entries vary by:
#   - hour of day
#   - weekday type
#   - hour-by-weekday-type structure
#   - season period
#   - holiday
#   - severe weather
#
# Input:
#   data/processed/analysis_panel_with_event_flags.parquet
#
# Output:
#   outputs/models/9_crz_temporal_results.txt
# ---------------------------------------------------------------------

INPUT = "data/processed/analysis_panel_with_event_flags.parquet"
OUT_RESULTS = "outputs/models/9_crz_temporal_results.txt"

os.makedirs("outputs/models", exist_ok=True)

df = pd.read_parquet(INPUT)

# ---------------------------------------------------------------------
# Time setup
# ---------------------------------------------------------------------

df["transit_timestamp"] = pd.to_datetime(df["transit_timestamp"])
df["date"] = pd.to_datetime(df["transit_timestamp"].dt.date)
df["hour"] = df["transit_timestamp"].dt.hour
df["day_of_week"] = df["transit_timestamp"].dt.day_name()
df["date_cluster"] = df["transit_timestamp"].dt.date.astype(str)

policy_start = pd.Timestamp("2025-01-05 00:00:00")

df["post_congestion_pricing"] = (
    df["transit_timestamp"] >= policy_start
).astype(int)

# Keep only post-policy CRZ period
df = df[df["post_congestion_pricing"] == 1].copy()

# ---------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------

for col in ["holiday_flag", "severe_weather_flag"]:
    if col not in df.columns:
        df[col] = 0

    df[col] = df[col].fillna(0).astype(int)

# ---------------------------------------------------------------------
# Interpretable temporal variables
# ---------------------------------------------------------------------

# Peak period is retained for descriptive tables only.
if "peak_period" not in df.columns:
    def assign_peak_period(hour):
        if 6 <= hour <= 9:
            return "am_peak"
        elif 16 <= hour <= 19:
            return "pm_peak"
        elif 0 <= hour <= 5:
            return "overnight"
        else:
            return "off_peak"

    df["peak_period"] = df["hour"].apply(assign_peak_period)

df["peak_period"] = pd.Categorical(
    df["peak_period"],
    categories=["overnight", "off_peak", "am_peak", "pm_peak"],
    ordered=True,
)

# Week structure
def assign_day_type(day):
    if day in ["Saturday", "Sunday"]:
        return "weekend"
    elif day == "Friday":
        return "friday"
    else:
        return "mon_thu"

df["day_type"] = df["day_of_week"].apply(assign_day_type)

df["day_type"] = pd.Categorical(
    df["day_type"],
    categories=["mon_thu", "friday", "weekend"],
    ordered=True,
)

# Season / period buckets
def assign_period(ts):
    ym = ts.strftime("%Y-%m")

    if ym in ["2025-01", "2025-02"]:
        return "rollout_winter_2025"
    elif ym in ["2025-03", "2025-04", "2025-05", "2025-06"]:
        return "spring_2025"
    elif ym in ["2025-07", "2025-08"]:
        return "summer_2025"
    elif ym in ["2025-09", "2025-10", "2025-11", "2025-12"]:
        return "fall_winter_2025"
    elif ym in ["2026-01", "2026-02", "2026-03"]:
        return "winter_2026"
    else:
        return "other"

df["period_bucket"] = df["transit_timestamp"].apply(assign_period)

df["period_bucket"] = pd.Categorical(
    df["period_bucket"],
    categories=[
        "rollout_winter_2025",
        "spring_2025",
        "summer_2025",
        "fall_winter_2025",
        "winter_2026",
        "other",
    ],
    ordered=True,
)

# Remove impossible / unused period bucket if present
df = df[df["period_bucket"] != "other"].copy()

# ---------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------

outcomes = [
    "crz_entries",
    "crz_excluded_roadway_entries",
    "crz_entries_1___cars_pickups_and_vans",
    "crz_entries_2___single_unit_trucks",
    "crz_entries_3___multi_unit_trucks",
    "crz_entries_4___buses",
    "crz_entries_5___motorcycles",
    "crz_entries_tlc_taxi_fhv",
    "crz_entries_region_brooklyn",
    "crz_entries_region_east_60th_st",
    "crz_entries_region_fdr_drive",
    "crz_entries_region_new_jersey",
    "crz_entries_region_queens",
    "crz_entries_region_west_60th_st",
    "crz_entries_region_west_side_highway",
]

outcomes = [c for c in outcomes if c in df.columns]

# ---------------------------------------------------------------------
# Model specification
# ---------------------------------------------------------------------

formula_template = """
{outcome} ~ C(hour)
+ C(day_type)
+ C(hour):C(day_type)
+ C(period_bucket)
+ holiday_flag
+ severe_weather_flag
"""

results = {}

# ---------------------------------------------------------------------
# Estimate models
# ---------------------------------------------------------------------

for outcome in outcomes:
    model_df = df.copy()

    model_df[outcome] = pd.to_numeric(model_df[outcome], errors="coerce")
    model_df = model_df.dropna(subset=[outcome])
    model_df = model_df[model_df[outcome] >= 0].copy()

    if model_df.empty:
        continue

    model_df[f"log_{outcome}"] = np.log1p(model_df[outcome])

    level_formula = formula_template.format(outcome=outcome)
    log_formula = formula_template.format(outcome=f"log_{outcome}")

    level_model = smf.ols(level_formula, data=model_df).fit(
        cov_type="cluster",
        cov_kwds={"groups": model_df["date_cluster"]},
    )

    log_model = smf.ols(log_formula, data=model_df).fit(
        cov_type="cluster",
        cov_kwds={"groups": model_df["date_cluster"]},
    )

    results[outcome] = {
        "level_model": level_model,
        "log_model": log_model,
        "mean": model_df[outcome].mean(),
        "median": model_df[outcome].median(),
        "total": model_df[outcome].sum(),
        "n": int(model_df.shape[0]),
    }

# ---------------------------------------------------------------------
# Descriptive summaries
# ---------------------------------------------------------------------

summary_rows = []

for outcome, r in results.items():
    summary_rows.append(
        {
            "outcome": outcome,
            "total": r["total"],
            "hourly_mean": r["mean"],
            "hourly_median": r["median"],
            "n_hours": r["n"],
            "level_r2": r["level_model"].rsquared,
            "log_r2": r["log_model"].rsquared,
        }
    )

summary = pd.DataFrame(summary_rows)

peak_summary = (
    df.groupby("peak_period", observed=True)[outcomes]
    .mean(numeric_only=True)
    .round(2)
)

hour_summary = (
    df.groupby("hour")[outcomes]
    .mean(numeric_only=True)
    .round(2)
)

day_type_summary = (
    df.groupby("day_type", observed=True)[outcomes]
    .mean(numeric_only=True)
    .round(2)
)

period_summary = (
    df.groupby("period_bucket", observed=True)[outcomes]
    .mean(numeric_only=True)
    .round(2)
)

daily = (
    df.groupby("date")[outcomes]
    .sum(numeric_only=True)
    .reset_index()
)

daily_summary = daily[outcomes].describe().T

# ---------------------------------------------------------------------
# Helper: compact coefficient table
# ---------------------------------------------------------------------

def compact_terms(model):
    rows = []

    for term, coef in model.params.items():
        if term == "Intercept":
            continue

        rows.append(
            {
                "term": term,
                "coef": coef,
                "std_err": model.bse.get(term, np.nan),
                "p_value": model.pvalues.get(term, np.nan),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------

with open(OUT_RESULTS, "w", encoding="utf-8") as f:
    f.write("=" * 90 + "\n")
    f.write("9 CRZ Temporal Effects Results\n")
    f.write("=" * 90 + "\n\n")

    f.write("NOTE:\n")
    f.write(
        "This is not a treated/control DiD. "
        "CRZ data begin at implementation, so this model estimates "
        "interpretable post-policy temporal patterns.\n\n"
    )

    f.write("Input file:\n")
    f.write(f"{INPUT}\n\n")

    f.write("Post-policy sample period:\n")
    f.write(
        f"{df['transit_timestamp'].min()} "
        f"to {df['transit_timestamp'].max()}\n\n"
    )

    f.write("Model specification:\n")
    f.write(
        "Outcome ~ hour FE + day_type + hour-by-day_type interaction "
        "+ period_bucket + holiday_flag + severe_weather_flag\n\n"
    )

    f.write("Baseline categories:\n")
    f.write("hour: 0\n")
    f.write("day_type: mon_thu\n")
    f.write("period_bucket: rollout_winter_2025\n\n")

    f.write("Standard errors:\n")
    f.write("Clustered by calendar date\n\n")

    f.write("=" * 90 + "\n")
    f.write("Outcome totals and model fit\n")
    f.write("=" * 90 + "\n\n")
    f.write(summary.to_string(index=False))
    f.write("\n\n")

    f.write("=" * 90 + "\n")
    f.write("Mean hourly entries by peak_period\n")
    f.write("=" * 90 + "\n\n")
    f.write(peak_summary.to_string())
    f.write("\n\n")

    f.write("=" * 90 + "\n")
    f.write("Mean hourly entries by hour\n")
    f.write("=" * 90 + "\n\n")
    f.write(hour_summary.to_string())
    f.write("\n\n")

    f.write("=" * 90 + "\n")
    f.write("Mean hourly entries by day_type\n")
    f.write("=" * 90 + "\n\n")
    f.write(day_type_summary.to_string())
    f.write("\n\n")

    f.write("=" * 90 + "\n")
    f.write("Mean hourly entries by period_bucket\n")
    f.write("=" * 90 + "\n\n")
    f.write(period_summary.to_string())
    f.write("\n\n")

    f.write("=" * 90 + "\n")
    f.write("Daily total distribution\n")
    f.write("=" * 90 + "\n\n")
    f.write(daily_summary.to_string())
    f.write("\n\n")

    for outcome, r in results.items():
        f.write("=" * 90 + "\n")
        f.write(f"Outcome: {outcome}\n")
        f.write("=" * 90 + "\n\n")

        f.write("LEVEL MODEL: compact coefficient table\n")
        f.write("-" * 90 + "\n")
        f.write(compact_terms(r["level_model"]).to_string(index=False))
        f.write("\n\n")

        f.write("LOG MODEL: compact coefficient table\n")
        f.write("-" * 90 + "\n")
        f.write(compact_terms(r["log_model"]).to_string(index=False))
        f.write("\n\n")

        f.write("Model fit:\n")
        f.write(f"Level R2: {r['level_model'].rsquared:.4f}\n")
        f.write(f"Log R2:   {r['log_model'].rsquared:.4f}\n")
        f.write(f"N:        {int(r['level_model'].nobs):,}\n\n")

# ---------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------

print("=" * 90)
print("9 CRZ temporal-effects models complete")
print("=" * 90)
print(f"Saved results to: {OUT_RESULTS}")
print()
print("Modeled outcomes:")
for outcome in outcomes:
    print(f"  - {outcome}")
