# 11c_multimodal_substitution_post_policy_shift.py

import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# ---------------------------------------------------------------------
# 11c Multimodal Substitution: Post-Policy Shift
#
# Purpose:
# Estimate whether the road/public-transport relationship changed
# after congestion pricing using standardized log variables.
#
# Important:
# This is association / substitution analysis, not causal mediation proof.
#
# Design:
#   - standardized log road outcomes
#   - standardized log public transport predictors
#   - contemporaneous + 1-hour lagged public transport controls
#   - focus only on Current PT × Post and Lagged PT × Post
#   - clustered-by-date SEs as main model
#   - HAC(24 hour) SEs as robustness model
# ---------------------------------------------------------------------

INPUT = "data/processed/analysis_panel_model_ready.parquet"
OUT_RESULTS = "outputs/models/11c_multimodal_substitution_post_policy_shift_results.txt"
OUT_COEFS = "outputs/models/11c_multimodal_substitution_post_policy_shift_coefficients.csv"

os.makedirs("outputs/models", exist_ok=True)

ROAD_OUTCOMES = [
    "taxi_trips",
    "forhire_trips",
    "bridge_traffic_total",
]

PT_VARS = [
    "subway_ridership",
    "bus_ridership",
]

POLICY_START = pd.Timestamp("2025-01-05 00:00:00")


def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    std = s.std()

    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=s.index)

    return (s - s.mean()) / std


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

required = ROAD_OUTCOMES + PT_VARS

missing = [
    col for col in required
    if col not in df.columns
]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

for col in ROAD_OUTCOMES + PT_VARS:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce",
    )

    df[f"log_{col}"] = np.log1p(df[col])
    df[f"z_log_{col}"] = zscore(df[f"log_{col}"])

df = add_lags(
    df,
    [
        f"z_log_{col}"
        for col in PT_VARS
    ],
    lags=(1,),
)


# ---------------------------------------------------------------------
# Model estimation
# ---------------------------------------------------------------------

rows = []

term_labels = {
    "z_log_subway_ridership:post_congestion_pricing":
        "Current Subway × Post",

    "z_log_subway_ridership_lag1:post_congestion_pricing":
        "Lagged Subway × Post",

    "z_log_bus_ridership:post_congestion_pricing":
        "Current Bus × Post",

    "z_log_bus_ridership_lag1:post_congestion_pricing":
        "Lagged Bus × Post",
}

outcome_labels = {
    "taxi_trips": "Taxi trips",
    "forhire_trips": "For-hire trips",
    "bridge_traffic_total": "Bridge traffic",
}

pt_labels = {
    "subway_ridership": "Subway",
    "bus_ridership": "Bus",
}

for road in ROAD_OUTCOMES:
    for pt in PT_VARS:

        pt_z = f"z_log_{pt}"
        pt_lag1 = f"{pt_z}_lag1"

        model_df = df.dropna(
            subset=[
                f"z_log_{road}",
                pt_z,
                pt_lag1,
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
        z_log_{road} ~
        {pt_z}
        + {pt_lag1}
        + post_congestion_pricing
        + {pt_z}:post_congestion_pricing
        + {pt_lag1}:post_congestion_pricing
        + post_congestion_pricing:is_weekend
        + C(hour)
        + C(day_of_week)
        + C(year_month)
        + holiday_flag
        + severe_weather_flag
        """

        print("=" * 90)
        print(f"Running standardized shift model: road={road}, pt={pt}")
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

        key_terms = [
            f"{pt_z}:post_congestion_pricing",
            f"{pt_lag1}:post_congestion_pricing",
        ]

        for model_name, model in [
            ("cluster", cluster_model),
            ("hac_24h", hac_model),
        ]:
            rows.extend(
                extract_terms(
                    model,
                    key_terms,
                    model_name,
                    extra={
                        "road_outcome": road,
                        "road_outcome_label": outcome_labels.get(road, road),
                        "public_transport_var": pt,
                        "public_transport_mode": pt_labels.get(pt, pt),
                    },
                )
            )

results = pd.DataFrame(rows)

results["term_label"] = results["term"].map(term_labels)

ordered_cols = [
    "road_outcome",
    "road_outcome_label",
    "public_transport_var",
    "public_transport_mode",
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
    f.write("11c Multimodal Substitution Post-Policy Shift Results\n")
    f.write("=" * 90 + "\n\n")

    f.write("Input:\n")
    f.write(f"{INPUT}\n\n")

    f.write("Coefficient CSV:\n")
    f.write(f"{OUT_COEFS}\n\n")

    f.write("Interpretation:\n")
    f.write(
        "This is a standardized association model, not causal mediation proof.\n"
        "Both road outcomes and public transport predictors are z-scored log variables.\n"
        "Coefficients are therefore more directly comparable across road outcomes "
        "and transport modes.\n\n"
    )

    f.write("Main specification:\n")
    f.write("z_log(road_outcome) ~ z_log(public_transport)\n")
    f.write("+ z_log(public_transport) × post\n")
    f.write("+ 1-hour lagged public transport controls\n")
    f.write("+ post-policy × weekend control\n")
    f.write("+ hour FE + day-of-week FE + year-month FE\n")
    f.write("+ holiday_flag + severe_weather_flag\n\n")

    f.write("Main model:\n")
    f.write("Clustered standard errors by calendar date.\n\n")

    f.write("Robustness model:\n")
    f.write("HAC standard errors with 24-hour lag window.\n\n")

    f.write("Reported terms:\n")
    f.write("- Current Subway × Post\n")
    f.write("- Lagged Subway × Post\n")
    f.write("- Current Bus × Post\n")
    f.write("- Lagged Bus × Post\n\n")

    f.write("Road outcomes:\n")
    for outcome in ROAD_OUTCOMES:
        f.write(f"- {outcome}\n")
    f.write("\n")

    f.write("-" * 90 + "\n")
    f.write("Main Results: Standardized Post-Policy Relationship Shift\n")
    f.write("-" * 90 + "\n\n")
    f.write(display_results.to_string(index=False))
    f.write("\n\n")


print("=" * 90)
print("11c multimodal substitution post-policy shift complete")
print("=" * 90)
print(f"Saved results to:      {OUT_RESULTS}")
print(f"Saved coefficients to: {OUT_COEFS}")
print()
print("Main Results")
print(display_results.to_string(index=False))
