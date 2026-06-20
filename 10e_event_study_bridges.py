# 10e_event_study_bridge.py

import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

INPUT = "data/processed/bridge_all_directions_did_panel.parquet"
OUT_RESULTS = "outputs/models/10e_event_study_bridge_results.txt"
OUT_FIGURE = "outputs/figures/10e_event_study_bridge.png"
OUT_COEFS = "outputs/models/10e_event_study_bridge_coefficients.csv"
OUT_COV = "outputs/models/10e_event_study_bridge_covariance.csv"

OUTCOME = "bridge_traffic"
LOG_OUTCOME = "log_bridge_traffic"
GROUP_FE = "bridge_group"

os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

df = pd.read_parquet(INPUT)

required = {
    "transit_timestamp",
    OUTCOME,
    "treated_group",
    GROUP_FE,
}

missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["transit_timestamp"] = pd.to_datetime(df["transit_timestamp"])
df[OUTCOME] = pd.to_numeric(df[OUTCOME], errors="coerce")
df["treated_group"] = df["treated_group"].astype(int)

df = df.dropna(subset=[OUTCOME, "treated_group", GROUP_FE]).copy()
df = df[df[OUTCOME] >= 0].copy()

if LOG_OUTCOME not in df.columns:
    df[LOG_OUTCOME] = np.log1p(df[OUTCOME])

df["hour"] = df["transit_timestamp"].dt.hour
df["day_of_week"] = df["transit_timestamp"].dt.day_name()
df["date_cluster"] = df["transit_timestamp"].dt.date.astype(str)

for col in ["holiday_flag", "severe_weather_flag"]:
    if col not in df.columns:
        df[col] = 0
    df[col] = df[col].fillna(0).astype(int)

df["event_time"] = (
    (df["transit_timestamp"].dt.year - 2025) * 12
    + (df["transit_timestamp"].dt.month - 1)
)

df = df[(df["event_time"] >= -12) & (df["event_time"] <= 12)].copy()
df["event_time_cat"] = df["event_time"].astype(str)

if "-1" not in df["event_time_cat"].unique():
    raise ValueError("Reference month (-1 = Dec 2024) not found in data.")

formula = f"""
{LOG_OUTCOME} ~
C(event_time_cat, Treatment(reference='-1')) * treated_group
+ C({GROUP_FE})
+ C(hour)
+ C(day_of_week)
+ holiday_flag
+ severe_weather_flag
"""

model = smf.ols(
    formula,
    data=df,
).fit(
    cov_type="cluster",
    cov_kwds={"groups": df["date_cluster"]},
)

# Export the full covariance matrix for event-study treatment interactions.
# This is required for formal HonestDiD sensitivity analysis.

event_terms = {}

for k in range(-12, 13):
    if k == -1:
        continue

    term = (
        "C(event_time_cat, Treatment(reference='-1'))"
        f"[T.{k}]:treated_group"
    )

    if term not in model.params.index:
        raise ValueError(f"Missing event-study interaction term: {term}")

    event_terms[k] = term

cov_matrix = model.cov_params()

cov_rows = []

for event_time_i, term_i in event_terms.items():
    for event_time_j, term_j in event_terms.items():
        cov_rows.append({
            "event_time_i": event_time_i,
            "event_time_j": event_time_j,
            "covariance": float(cov_matrix.loc[term_i, term_j]),
        })

covariance_df = pd.DataFrame(cov_rows)
covariance_df.to_csv(OUT_COV, index=False)

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

    results.append({
        "event_time": k,
        "beta": beta,
        "std_err": se,
        "p_value": p_value,
        "ci_low": beta - 1.96 * se,
        "ci_high": beta + 1.96 * se,
        "percent_effect": 100 * (np.exp(beta) - 1),
    })

results_df = pd.DataFrame(results)
results_df.to_csv(OUT_COEFS, index=False)

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

post_policy = results_df[results_df["event_time"] >= 0].dropna(
    subset=["percent_effect"]
)

avg_post_percent_effect = (
    post_policy["percent_effect"].mean()
    if len(post_policy) > 0
    else np.nan
)

display_results = results_df.copy()

for col in ["beta", "std_err", "ci_low", "ci_high"]:
    display_results[col] = display_results[col].round(4)

display_results["p_value"] = display_results["p_value"].round(4)
display_results["percent_effect"] = display_results["percent_effect"].round(2)

with open(OUT_RESULTS, "w", encoding="utf-8") as f:
    f.write("=" * 90 + "\n")
    f.write("10e Bridge Event Study Results\n")
    f.write("=" * 90 + "\n\n")

    f.write(f"Input:\n{INPUT}\n\n")
    f.write(f"Coefficient CSV:\n{OUT_COEFS}\n\n")
    f.write(f"Covariance CSV:\n{OUT_COV}\n\n")
    f.write(f"Figure:\n{OUT_FIGURE}\n\n")

    f.write("Reference month:\n")
    f.write("December 2024 (event_time = -1)\n\n")

    f.write("Event window:\n")
    f.write("-12 to +12 months around January 2025\n\n")

    f.write("Specification:\n")
    f.write("log_bridge_traffic ~ event_time × treated_group\n")
    f.write("+ bridge_group FE + hour FE + day-of-week FE\n")
    f.write("+ holiday_flag + severe_weather_flag\n\n")

    f.write("Important note:\n")
    f.write(
        "The bridge panel contains aggregate treated and control bridge traffic groups, "
        "rather than individual bridge-level units. Results should therefore be interpreted "
        "as aggregate treated-control dynamics.\n\n"
    )

    f.write("Standard errors:\n")
    f.write("Clustered standard errors by calendar date\n\n")
    
    f.write("-" * 90 + "\n")
    f.write("Pre-trend Joint Test\n")
    f.write("-" * 90 + "\n\n")

    if pretrend_test is None:
        f.write("No pre-policy interaction terms found.\n\n")
    else:
        f.write(str(pretrend_test) + "\n\n")

    f.write("-" * 90 + "\n")
    f.write("Post-Policy Average Effect\n")
    f.write("-" * 90 + "\n\n")
    f.write(
        f"Average post-policy percent effect: "
        f"{avg_post_percent_effect:.2f}%\n\n"
    )

    f.write("-" * 90 + "\n")
    f.write("Key Event-Time Coefficients\n")
    f.write("-" * 90 + "\n\n")
    f.write(display_results.to_string(index=False))
    f.write("\n")

plot_df = results_df.dropna(subset=["beta", "ci_low", "ci_high"])

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

plt.title("Bridge Event Study: Congestion Pricing Effect")
plt.xlabel("Months Relative to Policy Start (January 2025)")
plt.ylabel("Treatment Effect on Log Bridge Traffic")
plt.grid(alpha=0.25)

plt.tight_layout()
plt.savefig(OUT_FIGURE, dpi=300)
plt.close()