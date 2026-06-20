# 10c_event_study_taxi.py

import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

INPUT = "data/processed/taxi_did_panel.parquet"

OUT_RESULTS = "outputs/models/10c_event_study_taxi_results.txt"
OUT_FIGURE = "outputs/figures/10c_event_study_taxi.png"
OUT_COEFS = "outputs/models/10c_event_study_taxi_coefficients.csv"
OUT_COV = "outputs/models/10c_event_study_taxi_covariance.csv"

os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

print("=" * 90)
print("Loading Taxi DiD panel...")
print("=" * 90)

df = pd.read_parquet(INPUT)

required = {
    "transit_timestamp",
    "taxi_trips",
    "treated_group",
    "taxi_zone_group",
}

missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["transit_timestamp"] = pd.to_datetime(df["transit_timestamp"])
df["taxi_trips"] = pd.to_numeric(df["taxi_trips"], errors="coerce")
df["treated_group"] = df["treated_group"].astype(int)

df = df.dropna(
    subset=[
        "taxi_trips",
        "treated_group",
        "taxi_zone_group",
    ]
).copy()

df = df[df["taxi_trips"] >= 0].copy()
df["log_taxi_trips"] = np.log1p(df["taxi_trips"])

df["hour"] = df["transit_timestamp"].dt.hour
df["day_of_week"] = df["transit_timestamp"].dt.day_name()
df["date_cluster"] = df["transit_timestamp"].dt.date.astype(str)

for col in ["holiday_flag", "severe_weather_flag"]:
    if col not in df.columns:
        df[col] = 0

    df[col] = df[col].fillna(0).astype(int)

# Month relative to Jan 2025.
# December 2024 is event_time = -1 and is the omitted reference month.
df["event_time"] = (
    (df["transit_timestamp"].dt.year - 2025) * 12
    + (df["transit_timestamp"].dt.month - 1)
)

df = df[
    (df["event_time"] >= -12)
    & (df["event_time"] <= 12)
].copy()

df["event_time_cat"] = df["event_time"].astype(str)

if "-1" not in df["event_time_cat"].unique():
    raise ValueError("Reference month (-1 = Dec 2024) not found in data.")

# Group label is perfectly determined by treated_group in this two-group panel.
# Do not add C(group): it is collinear and destabilizes clustered covariance.
formula = """
log_taxi_trips ~
C(event_time_cat, Treatment(reference='-1')) * treated_group
+ C(hour)
+ C(day_of_week)
+ holiday_flag
+ severe_weather_flag
"""

print("=" * 90)
print("Running Taxi event study regression...")
print("=" * 90)

model = smf.ols(
    formula,
    data=df,
).fit(
    cov_type="cluster",
    cov_kwds={"groups": df["date_cluster"]},
)

results = []

for k in range(-12, 13):
    if k == -1:
        continue

    term = (
        "C(event_time_cat, Treatment(reference='-1'))"
        f"[T.{k}]:treated_group"
    )

    beta = model.params.get(term, np.nan)
    se = model.bse.get(term, np.nan)
    p_value = model.pvalues.get(term, np.nan)

    results.append(
        {
            "event_time": k,
            "beta": beta,
            "std_err": se,
            "p_value": p_value,
            "ci_low": beta - 1.96 * se,
            "ci_high": beta + 1.96 * se,
            "percent_effect": 100 * (np.exp(beta) - 1),
        }
    )

results_df = pd.DataFrame(results)

# ---------------------------------------------------------------------
# Export the full covariance matrix for formal HonestDiD sensitivity
# ---------------------------------------------------------------------

event_time_to_term = {}

for k in sorted(results_df["event_time"].dropna().astype(int).unique()):
    term = (
        "C(event_time_cat, Treatment(reference='-1'))"
        f"[T.{k}]:treated_group"
    )

    if term not in model.params.index:
        raise ValueError(
            "Could not find event-study interaction coefficient for "
            f"event_time={k}: {term}"
        )

    event_time_to_term[k] = term

event_times_for_cov = sorted(event_time_to_term)
event_terms_for_cov = [
    event_time_to_term[k]
    for k in event_times_for_cov
]

event_cov = model.cov_params().loc[
    event_terms_for_cov,
    event_terms_for_cov,
].copy()

event_cov.index = event_times_for_cov
event_cov.columns = event_times_for_cov
event_cov.index.name = "event_time_i"
event_cov.columns.name = "event_time_j"

event_cov_long = (
    event_cov
    .reset_index()
    .melt(
        id_vars="event_time_i",
        var_name="event_time_j",
        value_name="covariance",
    )
)

event_cov_long["event_time_i"] = event_cov_long["event_time_i"].astype(int)
event_cov_long["event_time_j"] = event_cov_long["event_time_j"].astype(int)
event_cov_long["covariance_estimator"] = model.cov_type

event_cov_long.to_csv(OUT_COV, index=False)

results_df.to_csv(OUT_COEFS, index=False)

# ---------------------------------------------------------------------
# Pre-trend joint test
# ---------------------------------------------------------------------

pre_terms = []

for k in range(-12, 0):
    if k == -1:
        continue

    term = (
        "C(event_time_cat, Treatment(reference='-1'))"
        f"[T.{k}]:treated_group"
    )

    if term in model.params.index:
        pre_terms.append(term)

if pre_terms:
    hypothesis = " = 0, ".join(pre_terms) + " = 0"
    pretrend_test = model.f_test(hypothesis)
else:
    pretrend_test = None

# ---------------------------------------------------------------------
# Post-policy average effect
# ---------------------------------------------------------------------

post_policy = results_df[
    results_df["event_time"] >= 0
].dropna(
    subset=["percent_effect"]
).copy()

if len(post_policy) > 0:
    avg_post_percent_effect = post_policy["percent_effect"].mean()
else:
    avg_post_percent_effect = np.nan

# ---------------------------------------------------------------------
# Save concise results
# ---------------------------------------------------------------------

display_results = results_df.copy()

for col in ["beta", "std_err", "ci_low", "ci_high"]:
    display_results[col] = display_results[col].round(4)

display_results["p_value"] = display_results["p_value"].round(4)
display_results["percent_effect"] = display_results["percent_effect"].round(2)

with open(OUT_RESULTS, "w", encoding="utf-8") as f:
    f.write("=" * 90 + "\n")
    f.write("10c Taxi Event Study Results\n")
    f.write("=" * 90 + "\n\n")

    f.write("Input:\n")
    f.write(f"{INPUT}\n\n")

    f.write("Coefficient CSV:\n")
    f.write(f"{OUT_COEFS}\n\n")

    f.write("Figure:\n")
    f.write(f"{OUT_FIGURE}\n\n")

    f.write("Reference month:\n")
    f.write("December 2024 (event_time = -1)\n\n")

    f.write("Event window:\n")
    f.write("-12 to +12 months around January 2025\n\n")

    f.write("Specification:\n")
    f.write("log_taxi_trips ~ event_time × treated_group\n")
    f.write("+ hour FE + day-of-week FE\n")
    f.write("+ holiday_flag + severe_weather_flag\n\n")

    f.write("Note:\n")
    f.write(
        "Year-month FE are intentionally not included because event_time "
        "already represents calendar month relative to the policy start.\n\n"
    )

    f.write("Standard errors:\n")
    f.write("Clustered standard errors by calendar date\n\n")

    f.write("Sample:\n")
    f.write(
        f"{df['transit_timestamp'].min()} "
        f"to {df['transit_timestamp'].max()}\n"
    )
    f.write(f"Rows: {len(df):,}\n\n")

    f.write("-" * 90 + "\n")
    f.write("Pre-trend Joint Test\n")
    f.write("-" * 90 + "\n\n")

    if pretrend_test is None:
        f.write("No pre-policy interaction terms found for joint test.\n\n")
    else:
        f.write(str(pretrend_test))
        f.write("\n\n")

    f.write("-" * 90 + "\n")
    f.write("Post-Policy Average Effect\n")
    f.write("-" * 90 + "\n\n")

    if np.isnan(avg_post_percent_effect):
        f.write("Average post-policy percent effect: NA\n\n")
    else:
        f.write(
            f"Average post-policy percent effect: "
            f"{avg_post_percent_effect:.2f}%\n\n"
        )

    f.write("-" * 90 + "\n")
    f.write("Key Event-Time Coefficients\n")
    f.write("-" * 90 + "\n\n")
    f.write(display_results.to_string(index=False))
    f.write("\n\n")

# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------

plot_df = results_df.dropna(
    subset=[
        "beta",
        "ci_low",
        "ci_high",
    ]
).copy()

plt.figure(figsize=(11, 6))

plt.axhline(y=0, linestyle="--", linewidth=1)
plt.axvline(x=0, linestyle="--", linewidth=1)

plt.plot(
    plot_df["event_time"],
    plot_df["beta"],
    marker="o",
    linewidth=1.8,
)

plt.fill_between(
    plot_df["event_time"],
    plot_df["ci_low"],
    plot_df["ci_high"],
    alpha=0.25,
)

plt.title("Taxi Event Study: Congestion Pricing Effect")
plt.xlabel("Months Relative to Policy Start (January 2025)")
plt.ylabel("Treatment Effect on Log Taxi Trips")
plt.grid(alpha=0.25)

plt.tight_layout()
plt.savefig(OUT_FIGURE, dpi=300)
plt.close()

print("=" * 90)
print("Taxi Event Study complete")
print("=" * 90)
print(f"Saved results to:      {OUT_RESULTS}")
print(f"Saved coefficients to: {OUT_COEFS}")
print(f"Saved figure to:       {OUT_FIGURE}")
print()

print("Pre-trend joint test:")
if pretrend_test is None:
    print("No pre-policy interaction terms found.")
else:
    print(pretrend_test)

print()
print("Average post-policy percent effect:")
if np.isnan(avg_post_percent_effect):
    print("NA")
else:
    print(f"{avg_post_percent_effect:.2f}%")
