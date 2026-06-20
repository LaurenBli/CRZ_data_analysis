# 11b_multimodal_substitution_by_mode.py

import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

INPUT = "data/processed/analysis_panel_model_ready.parquet"
OUT_RESULTS = "outputs/models/11b_multimodal_substitution_by_mode_results.txt"
OUT_COEFS = "outputs/models/11b_multimodal_substitution_by_mode_coefficients.csv"

os.makedirs("outputs/models", exist_ok=True)

ROAD_OUTCOMES = [
    "taxi_trips",
    "forhire_trips",
    "bridge_traffic_total",
]

PT_VARS = {
    "subway": "subway_ridership",
    "bus": "bus_ridership",
}

POLICY_START = pd.Timestamp("2025-01-05 00:00:00")


def add_lags(df, columns, lags=(1,)):
    df = df.sort_values("transit_timestamp").copy()

    for col in columns:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    return df


def fit_ols_cluster(formula, model_df):
    return smf.ols(formula, data=model_df).fit(
        cov_type="cluster",
        cov_kwds={"groups": model_df["date_cluster"]},
    )


def fit_ols_hac(formula, model_df, maxlags=24):
    return smf.ols(formula, data=model_df).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": maxlags},
    )


def extract_terms(model, terms, model_label, extra=None):
    rows = []
    extra = extra or {}

    for term in terms:
        coef = model.params.get(term, np.nan)
        se = model.bse.get(term, np.nan)

        rows.append(
            {
                **extra,
                "model": model_label,
                "term": term,
                "coef": coef,
                "std_err": se,
                "p_value": model.pvalues.get(term, np.nan),
                "ci_low": coef - 1.96 * se,
                "ci_high": coef + 1.96 * se,
                "r_squared": model.rsquared,
                "n_obs": int(model.nobs),
            }
        )

    return rows


df = pd.read_parquet(INPUT)

df["post_congestion_pricing"] = (
    df["post_congestion_pricing"]
    .astype(int)
)

df["transit_timestamp"] = pd.to_datetime(df["transit_timestamp"])
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
    df["post_congestion_pricing"]

for col in ["holiday_flag", "severe_weather_flag"]:
    if col not in df.columns:
        df[col] = 0
    df[col] = df[col].fillna(0).astype(int)

for outcome in ROAD_OUTCOMES:
    df[f"log_{outcome}"] = np.log1p(pd.to_numeric(df[outcome], errors="coerce"))

for _, col in PT_VARS.items():
    df[f"{col}_per_10k"] = pd.to_numeric(df[col], errors="coerce") / 10000

df = add_lags(
    df,
    [f"{col}_per_10k" for col in PT_VARS.values()],
    lags=(1,),
)

rows = []

term_labels = {
    "subway_ridership_per_10k:post_congestion_pricing": "Current Subway × Post",
    "subway_ridership_per_10k_lag1:post_congestion_pricing": "Lagged Subway × Post",
    "bus_ridership_per_10k:post_congestion_pricing": "Current Bus × Post",
    "bus_ridership_per_10k_lag1:post_congestion_pricing": "Lagged Bus × Post",
}

for outcome in ROAD_OUTCOMES:

    for mode_label, pt_col in PT_VARS.items():

        scaled_col = f"{pt_col}_per_10k"
        lag1_col = f"{scaled_col}_lag1"

        model_df = df.dropna(
            subset=[
                f"log_{outcome}",
                scaled_col,
                lag1_col,
                "post_congestion_pricing",
                "hour",
                "day_of_week",
                "year_month",
                "date_cluster",
            ]
        ).copy()

        formula = f"""
        log_{outcome} ~
        {scaled_col}
        + {lag1_col}
        + post_congestion_pricing
        + {scaled_col}:post_congestion_pricing
        + {lag1_col}:post_congestion_pricing
        + post_congestion_pricing:is_weekend
        + C(hour)
        + C(day_of_week)
        + C(year_month)
        + holiday_flag
        + severe_weather_flag
        """

        cluster_model = fit_ols_cluster(formula, model_df)
        hac_model = fit_ols_hac(formula, model_df)

        key_terms = [
            f"{scaled_col}:post_congestion_pricing",
            f"{lag1_col}:post_congestion_pricing",
        ]

        for model_name, model in [
            ("log_cluster", cluster_model),
            ("log_hac_24h", hac_model),
        ]:
            rows.extend(
                extract_terms(
                    model,
                    key_terms,
                    model_name,
                    {
                        "road_outcome": outcome,
                        "public_transport_mode": mode_label,
                    },
                )
            )

results = pd.DataFrame(rows)
results["term_label"] = results["term"].map(term_labels)

results.to_csv(OUT_COEFS, index=False)

display_results = results.drop(columns=["term"], errors="ignore").copy()

for col in ["coef","std_err","p_value","ci_low","ci_high","r_squared"]:
    display_results[col] = display_results[col].round(4)

with open(OUT_RESULTS, "w", encoding="utf-8") as f:

    f.write("=" * 90 + "\n")
    f.write("11b Multimodal Substitution by Mode Results\n")
    f.write("=" * 90 + "\n\n")

    f.write("Input:\n")
    f.write(f"{INPUT}\n\n")

    f.write("Interpretation:\n")
    f.write(
        "Separate association models for subway-road and bus-road relationships.\n"
        "Not causal mediation.\n\n"
    )

    f.write("Main model:\n")
    f.write("Clustered standard errors by calendar date.\n\n")

    f.write("Robustness model:\n")
    f.write("HAC(24-hour) standard errors.\n\n")

    f.write("-" * 90 + "\n")
    f.write("Main Results\n")
    f.write("-" * 90 + "\n\n")

    f.write(display_results.to_string(index=False))
    f.write("\n")

print(display_results.to_string(index=False))

