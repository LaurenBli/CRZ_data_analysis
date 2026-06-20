import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# ---------------------------------------------------------------------
# 9a Multimodal substitution baseline models
#
# Purpose:
#   Estimate whether post-policy CRZ entry intensity is associated with
#   substitution toward taxi, subway, bus, Citi Bike, and bridge traffic
#   after accounting for interpretable temporal structure.
#
# This is NOT a true geography-based DiD.
# It is a post-policy association / substitution-intensity model.
#
# Uses temporal controls:
#   - hour fixed effects
#   - day_type
#   - period_bucket
#   - linear time trend
#   - holiday_flag
#   - severe_weather_flag
#
# Input:
#   data/processed/analysis_panel_with_event_flags.parquet
#
# Output:
#   outputs/models/9a_multimodal_substitution_results.txt
# ---------------------------------------------------------------------

INPUT = "data/processed/analysis_panel_with_event_flags.parquet"
OUT_RESULTS = "outputs/models/9a_multimodal_substitution_results.txt"

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

# Keep post-policy only because CRZ intensity exists post-policy
df = df[df["post_congestion_pricing"] == 1].copy()
df = df.sort_values("transit_timestamp").reset_index(drop=True)

# Linear time trend in days since post-policy start
df["time_index_days"] = (
    df["transit_timestamp"] - df["transit_timestamp"].min()
).dt.total_seconds() / 86400

# ---------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------

for col in ["holiday_flag", "severe_weather_flag"]:
    if col not in df.columns:
        df[col] = 0

    df[col] = df[col].fillna(0).astype(int)

# Peak period retained only for descriptive readability if needed.
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

# Day type
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

# Period buckets
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
    ],
    ordered=True,
)

df = df[df["period_bucket"].notna()].copy()

# ---------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------

outcomes = [
    "taxi_trips",
    "subway_ridership",
    "bus_ridership",
    "citibike_rides",
    "bridge_traffic_total",
    "bridge_traffic_treated",
]

outcomes = [c for c in outcomes if c in df.columns]

# ---------------------------------------------------------------------
# Main treatment intensity variables
# ---------------------------------------------------------------------

intensity_vars = [
    "crz_entries",
    "crz_entries_1___cars_pickups_and_vans",
    "crz_entries_tlc_taxi_fhv",
]

intensity_vars = [c for c in intensity_vars if c in df.columns]

# Scale CRZ entries so coefficients are per 10,000 entries
for col in intensity_vars:
    df[f"{col}_per_10k"] = pd.to_numeric(df[col], errors="coerce") / 10000

intensity_scaled = [f"{c}_per_10k" for c in intensity_vars]

# ---------------------------------------------------------------------
# Model specs
# ---------------------------------------------------------------------

baseline_controls = """
C(hour)
+ C(day_type)
+ C(period_bucket)
+ time_index_days
+ holiday_flag
+ severe_weather_flag
"""

results = {}

for outcome in outcomes:
    model_df = df.copy()

    model_df[outcome] = pd.to_numeric(model_df[outcome], errors="coerce")
    model_df = model_df.dropna(subset=[outcome] + intensity_scaled)
    model_df = model_df[model_df[outcome] >= 0].copy()

    if model_df.empty:
        continue

    model_df[f"log_{outcome}"] = np.log1p(model_df[outcome])

    # Model A: baseline temporal-only model
    baseline_level_formula = f"""
    {outcome} ~ {baseline_controls}
    """

    baseline_log_formula = f"""
    log_{outcome} ~ {baseline_controls}
    """

    # Model B: CRZ total intensity
    total_level_formula = f"""
    {outcome} ~ crz_entries_per_10k
    + {baseline_controls}
    """

    total_log_formula = f"""
    log_{outcome} ~ crz_entries_per_10k
    + {baseline_controls}
    """

    models = {
        "baseline_level": smf.ols(
            baseline_level_formula,
            data=model_df,
        ).fit(
            cov_type="cluster",
            cov_kwds={"groups": model_df["date_cluster"]},
        ),
        "baseline_log": smf.ols(
            baseline_log_formula,
            data=model_df,
        ).fit(
            cov_type="cluster",
            cov_kwds={"groups": model_df["date_cluster"]},
        ),
        "total_intensity_level": smf.ols(
            total_level_formula,
            data=model_df,
        ).fit(
            cov_type="cluster",
            cov_kwds={"groups": model_df["date_cluster"]},
        ),
        "total_intensity_log": smf.ols(
            total_log_formula,
            data=model_df,
        ).fit(
            cov_type="cluster",
            cov_kwds={"groups": model_df["date_cluster"]},
        ),
    }

    # Model C: CRZ class decomposition
    # For taxi_trips, exclude TLC/FHV CRZ intensity to avoid partially
    # regressing taxi activity on a taxi/FHV component of CRZ entries.
    if outcome == "taxi_trips":
        decomposition_terms = [
            "crz_entries_1___cars_pickups_and_vans_per_10k",
        ]
    else:
        decomposition_terms = [
            "crz_entries_1___cars_pickups_and_vans_per_10k",
            "crz_entries_tlc_taxi_fhv_per_10k",
        ]

    decomposition_terms = [
        term for term in decomposition_terms
        if term in model_df.columns
    ]

    if decomposition_terms:
        decomposition_rhs = "\n+ ".join(decomposition_terms)

        decomposition_level_formula = f"""
        {outcome} ~ {decomposition_rhs}
        + {baseline_controls}
        """

        decomposition_log_formula = f"""
        log_{outcome} ~ {decomposition_rhs}
        + {baseline_controls}
        """

        models["class_decomposition_level"] = smf.ols(
            decomposition_level_formula,
            data=model_df,
        ).fit(
            cov_type="cluster",
            cov_kwds={"groups": model_df["date_cluster"]},
        )

        models["class_decomposition_log"] = smf.ols(
            decomposition_log_formula,
            data=model_df,
        ).fit(
            cov_type="cluster",
            cov_kwds={"groups": model_df["date_cluster"]},
        )

    results[outcome] = {
        "models": models,
        "mean": model_df[outcome].mean(),
        "median": model_df[outcome].median(),
        "n": int(model_df.shape[0]),
    }

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def extract_key_terms(model, terms):
    rows = []

    for term in terms:
        if term in model.params.index:
            rows.append(
                {
                    "term": term,
                    "coef": model.params[term],
                    "std_err": model.bse[term],
                    "p_value": model.pvalues[term],
                }
            )

    return pd.DataFrame(rows)


def model_fit_table(models):
    rows = []

    for name, model in models.items():
        rows.append(
            {
                "model": name,
                "r2": model.rsquared,
                "adj_r2": model.rsquared_adj,
                "n": int(model.nobs),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------

key_terms = [
    "crz_entries_per_10k",
    "crz_entries_1___cars_pickups_and_vans_per_10k",
    "crz_entries_tlc_taxi_fhv_per_10k",
]

with open(OUT_RESULTS, "w", encoding="utf-8") as f:
    f.write("=" * 90 + "\n")
    f.write("9a Multimodal Substitution Baseline Results\n")
    f.write("=" * 90 + "\n\n")

    f.write("NOTE:\n")
    f.write(
        "This is not a true geography-based DiD. "
        "It estimates post-policy associations between CRZ entry intensity "
        "and multimodal outcomes, conditional on interpretable temporal controls.\n\n"
    )

    f.write("Input file:\n")
    f.write(f"{INPUT}\n\n")

    f.write("Sample period:\n")
    f.write(
        f"{df['transit_timestamp'].min()} "
        f"to {df['transit_timestamp'].max()}\n\n"
    )

    f.write("Temporal controls:\n")
    f.write(
        "hour FE + day_type + period_bucket + linear time trend "
        "+ holiday_flag + severe_weather_flag\n\n"
    )

    f.write("Intensity variables scaled per 10,000 CRZ entries.\n\n")

    f.write("Baseline categories:\n")
    f.write("hour: 0\n")
    f.write("day_type: mon_thu\n")
    f.write("period_bucket: rollout_winter_2025\n\n")

    f.write("Standard errors:\n")
    f.write("Clustered by calendar date\n\n")

    f.write("Special handling:\n")
    f.write(
        "For taxi_trips decomposition models, TLC/FHV CRZ intensity is excluded "
        "to avoid partly regressing taxi activity on a taxi/FHV component of CRZ entries.\n\n"
    )

    for outcome, r in results.items():
        f.write("=" * 90 + "\n")
        f.write(f"Outcome: {outcome}\n")
        f.write("=" * 90 + "\n\n")

        f.write("Outcome summary:\n")
        f.write(f"Mean:   {r['mean']:,.2f}\n")
        f.write(f"Median: {r['median']:,.2f}\n")
        f.write(f"N:      {r['n']:,}\n\n")

        f.write("Model fit:\n")
        f.write(model_fit_table(r["models"]).to_string(index=False))
        f.write("\n\n")

        f.write("Key CRZ intensity coefficients\n")
        f.write("-" * 90 + "\n\n")

        for model_name, model in r["models"].items():
            if "baseline" in model_name:
                continue

            f.write(f"{model_name}\n")
            f.write("-" * len(model_name) + "\n")

            table = extract_key_terms(model, key_terms)

            if table.empty:
                f.write("No key intensity terms found.\n\n")
            else:
                f.write(table.to_string(index=False))
                f.write("\n\n")

        f.write("Full model summaries\n")
        f.write("-" * 90 + "\n\n")

        for model_name, model in r["models"].items():
            f.write(f"{model_name}\n")
            f.write("-" * len(model_name) + "\n")
            f.write(str(model.summary()))
            f.write("\n\n")

# ---------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------

print("=" * 90)
print("9a multimodal substitution baseline complete")
print("=" * 90)
print(f"Saved results to: {OUT_RESULTS}")
print()
print("Modeled outcomes:")
for outcome in results.keys():
    print(f"  - {outcome}")
