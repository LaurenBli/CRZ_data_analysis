# 11a_multimodal_substitution_main.py

import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# ---------------------------------------------------------------------
# 11a Multimodal Substitution Main Model
#
# Question:
# Does the relationship between public transport volumes and road
# vehicle activity change after congestion pricing?
#
# Important:
# This is NOT causal mediation proof.
# It is an association / substitution model conditional on temporal controls.
#
# Design:
#   - log road outcomes
#   - subway and bus predictors scaled per 10,000 riders
#   - contemporaneous + 1-hour lagged public transport controls
#   - focus only on Subway × Post and Bus × Post interaction terms
#   - clustered-by-date SEs as main model
#   - HAC(24 hour) SEs as robustness model
# ---------------------------------------------------------------------

INPUT = "data/processed/analysis_panel_model_ready.parquet"
OUT_RESULTS = "outputs/models/11a_multimodal_substitution_main_results.txt"
OUT_COEFS = "outputs/models/11a_multimodal_substitution_main_coefficients.csv"
OUT_CORR = "outputs/models/11a_multimodal_substitution_correlation_matrix.csv"

os.makedirs("outputs/models", exist_ok=True)

ROAD_OUTCOMES = [
    "taxi_trips",
    "forhire_trips",
    "bridge_traffic_total",
]

PUBLIC_TRANSPORT_VARS = [
    "subway_ridership",
    "bus_ridership",
]

POLICY_START = pd.Timestamp("2025-01-05 00:00:00")


def add_lags(df, columns, lags=(1,)):
    df = df.sort_values("transit_timestamp").copy()

    for col in columns:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    return df


def fit_ols_cluster(formula, model_df):
    return smf.ols(
        formula,
        data=model_df,
    ).fit(
        cov_type="cluster",
        cov_kwds={
            "groups": model_df["date_cluster"],
        },
    )


def fit_ols_hac(formula, model_df, maxlags=24):
    return smf.ols(
        formula,
        data=model_df,
    ).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": maxlags,
        },
    )


def extract_terms(model, terms, model_label, extra=None):
    rows = []
    extra = extra or {}

    for term in terms:
        coef = model.params.get(term, np.nan)
        std_err = model.bse.get(term, np.nan)

        rows.append(
            {
                **extra,
                "model": model_label,
                "term": term,
                "coef": coef,
                "std_err": std_err,
                "p_value": model.pvalues.get(term, np.nan),
                "ci_low": coef - 1.96 * std_err,
                "ci_high": coef + 1.96 * std_err,
                "r_squared": model.rsquared,
                "n_obs": int(model.nobs),
            }
        )

    return rows


print("=" * 90)
print("Loading analysis panel")
print("=" * 90)

df = pd.read_parquet(INPUT)

df["transit_timestamp"] = pd.to_datetime(df["transit_timestamp"])
df["date"] = pd.to_datetime(df["transit_timestamp"].dt.date)
df["hour"] = df["transit_timestamp"].dt.hour
df["day_of_week"] = df["transit_timestamp"].dt.day_name()
df["year_month"] = df["transit_timestamp"].dt.to_period("M").astype(str)
df["date_cluster"] = df["transit_timestamp"].dt.date.astype(str)
df["is_weekend"] = (df["transit_timestamp"].dt.dayofweek >= 5).astype(int)

if "post_congestion_pricing" not in df.columns:
    df["post_congestion_pricing"] = (
        df["transit_timestamp"] >= POLICY_START
    ).astype(int)
else:
    df["post_congestion_pricing"] = df["post_congestion_pricing"].astype(int)

for col in ["holiday_flag", "severe_weather_flag"]:
    if col not in df.columns:
        df[col] = 0

    df[col] = df[col].fillna(0).astype(int)

required = ROAD_OUTCOMES + PUBLIC_TRANSPORT_VARS

missing = [
    col for col in required
    if col not in df.columns
]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

for col in ROAD_OUTCOMES + PUBLIC_TRANSPORT_VARS:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce",
    )

    df[f"log_{col}"] = np.log1p(df[col])

df["subway_ridership_per_10k"] = df["subway_ridership"] / 10_000
df["bus_ridership_per_10k"] = df["bus_ridership"] / 10_000

df = add_lags(
    df,
    [
        "subway_ridership_per_10k",
        "bus_ridership_per_10k",
    ],
    lags=(1,),
)


# ---------------------------------------------------------------------
# Model estimation
# ---------------------------------------------------------------------

results_rows = []

key_terms = [
    "subway_ridership_per_10k:post_congestion_pricing",
    "bus_ridership_per_10k:post_congestion_pricing",
]

term_labels = {
    "subway_ridership_per_10k:post_congestion_pricing": "Subway × Post",
    "bus_ridership_per_10k:post_congestion_pricing": "Bus × Post",
}

outcome_labels = {
    "crz_entries": "CRZ entries",
    "taxi_trips": "Taxi trips",
    "forhire_trips": "For-hire trips",
    "bridge_traffic_total": "Bridge traffic",
}

for outcome in ROAD_OUTCOMES:
    model_df = df.dropna(
        subset=[
            f"log_{outcome}",
            "subway_ridership_per_10k",
            "bus_ridership_per_10k",
            "subway_ridership_per_10k_lag1",
            "bus_ridership_per_10k_lag1",
            "post_congestion_pricing",
            "hour",
            "day_of_week",
            "year_month",
            "holiday_flag",
            "severe_weather_flag",
            "is_weekend",
            "date_cluster",
        ]
    ).copy()

    formula = f"""
    log_{outcome} ~
    subway_ridership_per_10k
    + bus_ridership_per_10k
    + subway_ridership_per_10k_lag1
    + bus_ridership_per_10k_lag1
    + post_congestion_pricing
    + subway_ridership_per_10k:post_congestion_pricing
    + bus_ridership_per_10k:post_congestion_pricing
    + subway_ridership_per_10k_lag1:post_congestion_pricing
    + bus_ridership_per_10k_lag1:post_congestion_pricing
    + post_congestion_pricing:is_weekend
    + C(hour)
    + C(day_of_week)
    + C(year_month)
    + holiday_flag
    + severe_weather_flag
    """

    # Correlation diagnostics

    if outcome == ROAD_OUTCOMES[0]:

        corr_vars = [
            "subway_ridership_per_10k",
            "subway_ridership_per_10k_lag1",
            "bus_ridership_per_10k",
            "bus_ridership_per_10k_lag1",
        ]

        corr_matrix = model_df[corr_vars].corr()

        corr_matrix.to_csv(OUT_CORR)

        print("\n" + "=" * 90)
        print("Correlation Matrix: Public Transport Variables")
        print("=" * 90)
        print(corr_matrix.round(3))
        print(f"\nSaved correlation matrix to: {OUT_CORR}\n")


    print("=" * 90)
    print(f"Running multimodal models for: {outcome}")
    print("=" * 90)

    cluster_model = fit_ols_cluster(
        formula,
        model_df,
    )

    hac_model = fit_ols_hac(
        formula,
        model_df,
        maxlags=24,
    )

    for model_name, model in [
        ("log_cluster", cluster_model),
        ("log_hac_24h", hac_model),
    ]:
        results_rows.extend(
            extract_terms(
                model,
                key_terms,
                model_name,
                extra={
                    "road_outcome": outcome,
                    "road_outcome_label": outcome_labels.get(outcome, outcome),
                },
            )
        )

results = pd.DataFrame(results_rows)

results["term_label"] = results["term"].map(term_labels)

ordered_cols = [
    "road_outcome",
    "road_outcome_label",
    "model",
    "term",
    "term_label",
    "coef",
    "std_err",
    "p_value",
    "ci_low",
    "ci_high",
    "r_squared",
    "n_obs",
]

results = results[ordered_cols]

results.to_csv(
    OUT_COEFS,
    index=False,
)


# ---------------------------------------------------------------------
# Save concise TXT report
# ---------------------------------------------------------------------

display_results = results.copy()

for col in [
    "coef",
    "std_err",
    "p_value",
    "ci_low",
    "ci_high",
    "r_squared",
]:
    display_results[col] = display_results[col].round(4)

display_results = display_results.drop(
    columns=[
        "term",
    ],
    errors="ignore",
)

with open(OUT_RESULTS, "w", encoding="utf-8") as f:
    f.write("=" * 90 + "\n")
    f.write("11a Multimodal Substitution Main Results\n")
    f.write("=" * 90 + "\n\n")

    f.write("Input:\n")
    f.write(f"{INPUT}\n\n")

    f.write("Coefficient CSV:\n")
    f.write(f"{OUT_COEFS}\n\n")
    f.write("Correlation matrix:\n")
    f.write(f"{OUT_CORR}\n\n")

    f.write("Interpretation:\n")
    f.write(
        "This is an association / substitution model, not causal mediation proof.\n"
        "It estimates whether the relationship between public transport volume "
        "and road activity changes after congestion pricing.\n\n"
    )

    f.write("Main specification:\n")
    f.write("log(road_outcome + 1) ~ public transport + public transport × post\n")
    f.write("+ 1-hour lagged public transport controls\n")
    f.write("+ post-policy × weekend control\n")
    f.write("+ hour FE + day-of-week FE + year-month FE\n")
    f.write("+ holiday_flag + severe_weather_flag\n\n")

    f.write("Main model:\n")
    f.write("Clustered standard errors by calendar date.\n\n")

    f.write("Robustness model:\n")
    f.write("HAC standard errors with 24-hour lag window.\n\n")

    f.write("Public transport variables:\n")
    f.write("- subway_ridership_per_10k\n")
    f.write("- bus_ridership_per_10k\n\n")

    f.write("Reported terms:\n")
    f.write("- Subway × Post\n")
    f.write("- Bus × Post\n\n")

    f.write("Road outcomes:\n")
    for outcome in ROAD_OUTCOMES:
        f.write(f"- {outcome}\n")
    f.write("\n")

    f.write("-" * 90 + "\n")
    f.write("Main Results: Change in Transit-Road Relationship After Policy\n")
    f.write("-" * 90 + "\n\n")
    f.write(display_results.to_string(index=False))
    f.write("\n\n")


print("=" * 90)
print("11a multimodal substitution main complete")
print("=" * 90)
print(f"Saved results to:      {OUT_RESULTS}")
print(f"Saved coefficients to: {OUT_COEFS}")
print()
print("Main Results")
print(display_results.to_string(index=False))
