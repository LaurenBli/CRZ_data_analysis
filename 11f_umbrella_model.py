# 11f_multimodal_aggregate_indices_clean.py

import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


INPUT = "data/processed/analysis_panel_model_ready.parquet"
OUT_RESULTS = "outputs/models/11f_multimodal_aggregate_indices_results.txt"
OUT_COEFS = "outputs/models/11f_multimodal_aggregate_indices_coefficients.csv"

os.makedirs("outputs/models", exist_ok=True)

POLICY_START = pd.Timestamp("2025-01-05 00:00:00")

PUBLIC_TRANSPORT_COLS = [
    "subway_ridership",
    "bus_ridership",
]

# Main system-level road index:
# broad road-based mobility with full pre/post coverage.
ROAD_COLS_BROAD = [
    "bridge_traffic_total",
    "taxi_trips",
    "forhire_trips",
]

# Subsystem / robustness index:
# for-hire road mobility only.
ROAD_COLS_FORHIRE = [
    "taxi_trips",
    "forhire_trips",
]

# ---------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------

BROAD_MODEL_SPECS = [
    {
        "model_family": "A_simple_log_broad_road",
        "model_group": "broad_road_bt_taxi_fhv",
        "outcome": "log_road_total_broad",
        "predictor": "log_public_transport_total",
        "reported_term": "log_public_transport_total:post_congestion_pricing",
        "interpretation": (
            "Broad road-mobility log model. Road mobility includes bridge/tunnel "
            "traffic, taxi trips, and FHV trips."
        ),
    },
    {
        "model_family": "B_equal_weight_broad_road",
        "model_group": "broad_road_bt_taxi_fhv",
        "outcome": "road_index_equal_broad",
        "predictor": "pt_index_equal",
        "reported_term": "pt_index_equal:post_congestion_pricing",
        "interpretation": (
            "Broad road-mobility equal-weight standardized index. Bridge/tunnel "
            "traffic, taxi trips, and FHV trips contribute equally."
        ),
    },
    {
        "model_family": "C_activity_weighted_broad_road",
        "model_group": "broad_road_bt_taxi_fhv",
        "outcome": "road_index_weighted_broad",
        "predictor": "pt_index_weighted",
        "reported_term": "pt_index_weighted:post_congestion_pricing",
        "interpretation": (
            "Broad road-mobility activity-weighted standardized index. Components "
            "are weighted by average activity levels over the sample period."
        ),
    },
]

FORHIRE_MODEL_SPECS = [
    {
        "model_family": "D_simple_log_forhire",
        "model_group": "forhire_taxi_fhv_only",
        "outcome": "log_road_total_forhire",
        "predictor": "log_public_transport_total",
        "reported_term": "log_public_transport_total:post_congestion_pricing",
        "interpretation": (
            "For-hire mobility log model. Road mobility includes taxi and FHV trips only."
        ),
    },
    {
        "model_family": "E_equal_weight_forhire",
        "model_group": "forhire_taxi_fhv_only",
        "outcome": "road_index_equal_forhire",
        "predictor": "pt_index_equal",
        "reported_term": "pt_index_equal:post_congestion_pricing",
        "interpretation": (
            "For-hire mobility equal-weight standardized index. Taxi and FHV trips "
            "contribute equally."
        ),
    },
    {
        "model_family": "F_activity_weighted_forhire",
        "model_group": "forhire_taxi_fhv_only",
        "outcome": "road_index_weighted_forhire",
        "predictor": "pt_index_weighted",
        "reported_term": "pt_index_weighted:post_congestion_pricing",
        "interpretation": (
            "For-hire mobility activity-weighted standardized index. Taxi and FHV "
            "trips are weighted by average activity levels over the sample period."
        ),
    },
]

MODEL_SPECS = BROAD_MODEL_SPECS + FORHIRE_MODEL_SPECS


def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    std = s.std()

    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=s.index)

    return (s - s.mean()) / std


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


def extract_term(model, term, model_label, extra=None):
    extra = extra or {}
    coef = model.params.get(term, np.nan)
    se = model.bse.get(term, np.nan)

    return {
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


def make_weight_dict(df, columns):
    means = {
        col: pd.to_numeric(df[col], errors="coerce").mean()
        for col in columns
    }

    total = sum(value for value in means.values() if pd.notna(value))

    if total == 0 or pd.isna(total):
        raise ValueError(f"Cannot create weights for columns {columns}")

    return {col: means[col] / total for col in columns}


def make_equal_index(df, columns, suffix):
    df[f"road_index_equal_{suffix}"] = sum(
        df[f"z_log_{col}"] for col in columns
    ) / len(columns)


def make_weighted_index(df, columns, weights, suffix):
    df[f"road_index_weighted_{suffix}"] = 0.0
    for col in columns:
        df[f"road_index_weighted_{suffix}"] += (
            weights[col] * df[f"z_log_{col}"]
        )


print("=" * 90)
print("Loading hourly analysis panel")
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

required = sorted(
    set(PUBLIC_TRANSPORT_COLS + ROAD_COLS_BROAD + ROAD_COLS_FORHIRE)
)

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns in {INPUT}: {missing}")

for col in required:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ---------------------------------------------------------------------
# Aggregate variables
# ---------------------------------------------------------------------

df["public_transport_total"] = df[PUBLIC_TRANSPORT_COLS].sum(axis=1)

df["road_total_broad"] = df[ROAD_COLS_BROAD].sum(axis=1)
df["road_total_forhire"] = df[ROAD_COLS_FORHIRE].sum(axis=1)

df["log_public_transport_total"] = np.log1p(df["public_transport_total"])
df["log_road_total_broad"] = np.log1p(df["road_total_broad"])
df["log_road_total_forhire"] = np.log1p(df["road_total_forhire"])

for col in required:
    df[f"log_{col}"] = np.log1p(df[col])
    df[f"z_log_{col}"] = zscore(df[f"log_{col}"])

# Public transport indices
df["pt_index_equal"] = sum(
    df[f"z_log_{col}"] for col in PUBLIC_TRANSPORT_COLS
) / len(PUBLIC_TRANSPORT_COLS)

pt_weights = make_weight_dict(df, PUBLIC_TRANSPORT_COLS)

df["pt_index_weighted"] = 0.0
for col in PUBLIC_TRANSPORT_COLS:
    df["pt_index_weighted"] += pt_weights[col] * df[f"z_log_{col}"]

# Road indices
road_weights_broad = make_weight_dict(df, ROAD_COLS_BROAD)
road_weights_forhire = make_weight_dict(df, ROAD_COLS_FORHIRE)

make_equal_index(df, ROAD_COLS_BROAD, "broad")
make_equal_index(df, ROAD_COLS_FORHIRE, "forhire")

make_weighted_index(df, ROAD_COLS_BROAD, road_weights_broad, "broad")
make_weighted_index(df, ROAD_COLS_FORHIRE, road_weights_forhire, "forhire")

# ---------------------------------------------------------------------
# Run models
# ---------------------------------------------------------------------

rows = []

for spec in MODEL_SPECS:
    outcome = spec["outcome"]
    predictor = spec["predictor"]

    model_df = df.dropna(
        subset=[
            outcome,
            predictor,
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
    {outcome} ~
    {predictor}
    + post_congestion_pricing
    + {predictor}:post_congestion_pricing
    + post_congestion_pricing:is_weekend
    + C(hour)
    + C(day_of_week)
    + C(year_month)
    + holiday_flag
    + severe_weather_flag
    """

    print("=" * 90)
    print(f"Running aggregate model: {spec['model_family']}")
    print("=" * 90)

    cluster_model = fit_ols_cluster(formula, model_df)
    hac_model = fit_ols_hac(formula, model_df, maxlags=24)

    for model_name, model in [
        ("cluster", cluster_model),
        ("hac_24h", hac_model),
    ]:
        rows.append(
            extract_term(
                model,
                spec["reported_term"],
                model_name,
                extra={
                    "model_group": spec["model_group"],
                    "model_family": spec["model_family"],
                    "outcome": outcome,
                    "predictor": predictor,
                    "interpretation": spec["interpretation"],
                },
            )
        )

results = pd.DataFrame(rows)
results["term_label"] = "Public Transport × Post"

ordered_cols = [
    "model_group",
    "model_family",
    "model",
    "outcome",
    "predictor",
    "term",
    "term_label",
    "coef",
    "std_err",
    "p_value",
    "ci_low",
    "ci_high",
    "r_squared",
    "n_obs",
    "interpretation",
]

results = results[ordered_cols]
results.to_csv(OUT_COEFS, index=False)

# ---------------------------------------------------------------------
# Save TXT report
# ---------------------------------------------------------------------

display_results = results.copy()

for col in ["coef", "std_err", "p_value", "ci_low", "ci_high", "r_squared"]:
    display_results[col] = display_results[col].round(4)

display_results = display_results.drop(
    columns=["term", "interpretation"],
    errors="ignore",
)

broad_display = display_results[
    display_results["model_group"] == "broad_road_bt_taxi_fhv"
].copy()

forhire_display = display_results[
    display_results["model_group"] == "forhire_taxi_fhv_only"
].copy()

with open(OUT_RESULTS, "w", encoding="utf-8") as f:
    f.write("=" * 90 + "\n")
    f.write("11f Aggregate Public Transport / Road Mobility Relationship Models\n")
    f.write("=" * 90 + "\n\n")

    f.write("Input:\n")
    f.write(f"{INPUT}\n\n")

    f.write("Coefficient CSV:\n")
    f.write(f"{OUT_COEFS}\n\n")

    f.write("Purpose:\n")
    f.write(
        "This file estimates whether aggregate relationships between public "
        "transport activity and road-based mobility changed after congestion pricing.\n\n"
    )

    f.write("Important interpretation note:\n")
    f.write(
        "These are association / moderation models, not causal mediation models. "
        "The coefficient Public Transport × Post indicates whether the relationship "
        "between public transport and the road-mobility index changed after policy "
        "implementation.\n\n"
    )

    f.write("Policy date:\n")
    f.write(f"{POLICY_START}\n\n")

    f.write("Public transport components:\n")
    for col in PUBLIC_TRANSPORT_COLS:
        f.write(f"- {col}\n")
    f.write("\n")

    f.write("Broad road mobility components:\n")
    for col in ROAD_COLS_BROAD:
        f.write(f"- {col}\n")
    f.write("\n")

    f.write("For-hire mobility components:\n")
    for col in ROAD_COLS_FORHIRE:
        f.write(f"- {col}\n")
    f.write("\n")

    f.write("Excluded from aggregate road indices:\n")
    f.write(
        "- crz_entries, because the CRZ entry series begins at policy implementation "
        "and therefore does not provide a comparable pre-policy period.\n\n"
    )

    f.write("-" * 90 + "\n")
    f.write("Aggregation Weights\n")
    f.write("-" * 90 + "\n\n")

    f.write("Public transport activity weights:\n")
    for col, weight in pt_weights.items():
        f.write(f"- {col}: {weight:.4f}\n")
    f.write("\n")

    f.write("Broad road mobility activity weights:\n")
    for col, weight in road_weights_broad.items():
        f.write(f"- {col}: {weight:.4f}\n")
    f.write("\n")

    f.write("For-hire mobility activity weights:\n")
    for col, weight in road_weights_forhire.items():
        f.write(f"- {col}: {weight:.4f}\n")
    f.write("\n")

    f.write("-" * 90 + "\n")
    f.write("Model Definitions\n")
    f.write("-" * 90 + "\n\n")

    f.write("A-C. Broad road mobility models\n")
    f.write(
        "Road mobility is defined using bridge_traffic_total, taxi_trips, "
        "and forhire_trips. Three specifications are estimated: simple log aggregate, "
        "equal-weight standardized index, and activity-weighted standardized index.\n\n"
    )

    f.write("D-F. For-hire mobility models\n")
    f.write(
        "Road mobility is defined using taxi_trips and forhire_trips only. "
        "These models test whether the public-transport relationship changed within "
        "the for-hire road mobility subsystem.\n\n"
    )

    f.write("Controls included in all models:\n")
    f.write("- post-policy × weekend control\n")
    f.write("- hour fixed effects\n")
    f.write("- day-of-week fixed effects\n")
    f.write("- year-month fixed effects\n")
    f.write("- holiday_flag\n")
    f.write("- severe_weather_flag\n\n")

    f.write("Inference:\n")
    f.write("- cluster model: clustered standard errors by calendar date\n")
    f.write("- HAC model: HAC standard errors with 24-hour lag window\n\n")

    f.write("-" * 90 + "\n")
    f.write("Main Results: Broad Road Mobility System\n")
    f.write("-" * 90 + "\n\n")
    f.write(broad_display.to_string(index=False))
    f.write("\n\n")

    f.write("-" * 90 + "\n")
    f.write("Subsystem Results: Taxi and FHV Mobility Only\n")
    f.write("-" * 90 + "\n\n")
    f.write(forhire_display.to_string(index=False))
    f.write("\n\n")

    f.write("Interpretation guide:\n")
    f.write(
        "Negative Public Transport × Post coefficient: public transport and road "
        "mobility became more inversely related after congestion pricing.\n"
    )
    f.write(
        "Positive Public Transport × Post coefficient: public transport and road "
        "mobility became more positively related after congestion pricing.\n"
    )
    f.write(
        "Insignificant coefficient: no clear aggregate shift in the PT-road relationship.\n"
    )

print("=" * 90)
print("11f aggregate multimodal models complete")
print("=" * 90)
print(f"Saved results to:      {OUT_RESULTS}")
print(f"Saved coefficients to: {OUT_COEFS}")
print()
print("Main Results: Broad Road Mobility System")
print(broad_display.to_string(index=False))
print()
print("Subsystem Results: Taxi and FHV Mobility Only")
print(forhire_display.to_string(index=False))
