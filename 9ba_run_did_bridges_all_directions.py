"""
9ba_run_did_bridges_all_directions.py

All-directions bridge/tunnel DiD robustness analysis.

Treatment definition
--------------------
Treated: every direction of the CRZ-relevant facilities
    facility_id in {21, 22, 24, 27, 28}

Comparison group: every direction of the existing spillover facilities
    facility_id in {23, 29, 30}

This preserves the original directional bridge analysis and produces a
separate, explicitly labelled all-directions comparison. The comparison
facilities may still experience spillovers, so interpret this as a
robustness / comparative analysis rather than an unaffected-control design.
"""

import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

BRIDGES_MASTER = "data/processed/bridges_master_2024_01_2026_03.parquet"
EVENT_PANEL = "data/processed/analysis_panel_with_event_flags.parquet"

OUT_PANEL = "data/processed/bridge_all_directions_did_panel.parquet"
OUT_RESULTS = "outputs/models/9ba_bridge_all_directions_did_results.txt"
OUT_PRETRENDS = "outputs/models/9ba_bridge_all_directions_pretrend_summary.csv"
OUT_MAIN_RESULTS = "outputs/models/9ba_bridge_all_directions_key_results.csv"
OUT_FACILITIES = "outputs/models/9ba_bridge_all_directions_facilities.csv"

POLICY_START = pd.Timestamp("2025-01-05 00:00:00")

# These match the existing bridge treatment-map logic.
TREATED_FACILITY_IDS = {"21", "22", "24", "27", "28"}
COMPARISON_FACILITY_IDS = {"23", "29", "30"}

os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def extract_terms(model, terms, model_name):
    """Return a compact, consistently formatted coefficient table."""
    conf = model.conf_int()
    rows = []

    for term in terms:
        if term not in model.params.index:
            rows.append(
                {
                    "model": model_name,
                    "term": term,
                    "coef": np.nan,
                    "std_err": np.nan,
                    "p_value": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "r_squared": model.rsquared,
                    "note": "Term not estimable / not found",
                }
            )
            continue

        rows.append(
            {
                "model": model_name,
                "term": term,
                "coef": model.params[term],
                "std_err": model.bse[term],
                "p_value": model.pvalues[term],
                "ci_low": conf.loc[term, 0],
                "ci_high": conf.loc[term, 1],
                "r_squared": model.rsquared,
                "note": "",
            }
        )

    return pd.DataFrame(rows)


def validate_required(df, columns, source_name):
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {missing}")


# ---------------------------------------------------------------------
# Load bridge master and label facilities
# ---------------------------------------------------------------------

print("=" * 90)
print("9ba All-Directions Bridge/Tunnel DiD")
print("=" * 90)
print("Loading bridge master...")

bridge = pd.read_parquet(
    BRIDGES_MASTER,
    columns=[
        "transit_timestamp",
        "facility_id",
        "facility",
        "direction",
        "traffic_count",
    ],
)

validate_required(
    bridge,
    ["transit_timestamp", "facility_id", "facility", "direction", "traffic_count"],
    BRIDGES_MASTER,
)

bridge["transit_timestamp"] = pd.to_datetime(bridge["transit_timestamp"])
bridge["facility_id"] = pd.to_numeric(
    bridge["facility_id"],
    errors="coerce",
).astype("Int64").astype(str)
bridge["traffic_count"] = pd.to_numeric(
    bridge["traffic_count"],
    errors="coerce",
)

if bridge["traffic_count"].isna().any():
    raise ValueError(
        f"Bridge master has {int(bridge['traffic_count'].isna().sum()):,} "
        "missing traffic_count values."
    )

if (bridge["traffic_count"] < 0).any():
    raise ValueError("Bridge master has negative traffic_count values.")

bridge = bridge[
    bridge["facility_id"].isin(TREATED_FACILITY_IDS | COMPARISON_FACILITY_IDS)
].copy()

if bridge.empty:
    raise ValueError("No rows matched the selected treated/comparison facility IDs.")

bridge["treated_group"] = bridge["facility_id"].isin(TREATED_FACILITY_IDS).astype(int)
bridge["bridge_group"] = np.where(
    bridge["treated_group"].eq(1),
    "treated_all_directions",
    "comparison_all_directions",
)

# Save a transparent facility/direction inventory.
facility_inventory = (
    bridge[
        ["facility_id", "facility", "direction", "treated_group", "bridge_group"]
    ]
    .drop_duplicates()
    .sort_values(["treated_group", "facility_id", "direction"])
    .reset_index(drop=True)
)
facility_inventory.to_csv(OUT_FACILITIES, index=False)

print("\nFacility/direction inventory:")
print(facility_inventory.to_string(index=False))

# ---------------------------------------------------------------------
# Aggregate to hour x group, retaining both directions in each facility
# ---------------------------------------------------------------------

bridge["facility_direction"] = (
    bridge["facility_id"].astype(str)
    + " | "
    + bridge["direction"].astype(str)
)

hour_group = (
    bridge.groupby(
        ["transit_timestamp", "treated_group", "bridge_group"],
        as_index=False,
    )
    .agg(
        bridge_traffic=("traffic_count", "sum"),
        n_facilities=("facility_id", "nunique"),
        n_facility_directions=("facility_direction", "nunique"),
    )
)

# Require one treated and one comparison observation per analyzed hour.
wide = (
    hour_group.pivot(
        index="transit_timestamp",
        columns="treated_group",
        values="bridge_traffic",
    )
    .rename(columns={0: "comparison", 1: "treated"})
)

if not {0, 1}.issubset(set(
    hour_group["treated_group"].unique()
)):
    raise ValueError("Expected both treated and comparison bridge groups.")

missing_group_hours = wide[wide.isna().any(axis=1)]
if not missing_group_hours.empty:
    print(
        f"\nWARNING: Dropping {len(missing_group_hours):,} hour(s) without "
        "both treated and comparison traffic totals."
    )

complete_hours = wide.dropna().index
did = hour_group[
    hour_group["transit_timestamp"].isin(complete_hours)
].copy()

if did.groupby("transit_timestamp").size().ne(2).any():
    raise ValueError(
        "The all-directions bridge panel does not have exactly two rows "
        "(treated and comparison) for every retained hour."
    )

# ---------------------------------------------------------------------
# Add calendar/event controls from the final event panel
# ---------------------------------------------------------------------

print("\nLoading event controls...")

controls = pd.read_parquet(EVENT_PANEL)
validate_required(controls, ["transit_timestamp"], EVENT_PANEL)

controls["transit_timestamp"] = pd.to_datetime(controls["transit_timestamp"])

for col in ["holiday_flag", "severe_weather_flag"]:
    if col not in controls.columns:
        controls[col] = 0
    controls[col] = pd.to_numeric(controls[col], errors="coerce").fillna(0).astype(int)

controls = controls[
    ["transit_timestamp", "holiday_flag", "severe_weather_flag"]
].drop_duplicates("transit_timestamp")

did = did.merge(
    controls,
    on="transit_timestamp",
    how="left",
    validate="many_to_one",
)

if did[["holiday_flag", "severe_weather_flag"]].isna().any().any():
    missing_controls = int(
        did[["holiday_flag", "severe_weather_flag"]].isna().any(axis=1).sum()
    )
    raise ValueError(
        f"{missing_controls:,} bridge DiD rows have no matching event controls."
    )

did["date"] = pd.to_datetime(did["transit_timestamp"].dt.date)
did["hour"] = did["transit_timestamp"].dt.hour
did["day_of_week"] = did["transit_timestamp"].dt.day_name()
did["year_month"] = did["transit_timestamp"].dt.to_period("M").astype(str)
did["date_cluster"] = did["transit_timestamp"].dt.date.astype(str)
did["post_congestion_pricing"] = (
    did["transit_timestamp"] >= POLICY_START
).astype(int)

did["bridge_group"] = pd.Categorical(
    did["bridge_group"],
    categories=["comparison_all_directions", "treated_all_directions"],
    ordered=True,
)
did["log_bridge_traffic"] = np.log1p(did["bridge_traffic"])

hour_index = (
    did[["transit_timestamp"]]
    .drop_duplicates()
    .sort_values("transit_timestamp")
    .reset_index(drop=True)
)
hour_index["time_index"] = np.arange(len(hour_index))

did = did.merge(
    hour_index,
    on="transit_timestamp",
    how="left",
    validate="many_to_one",
)

did = did.sort_values(["transit_timestamp", "treated_group"]).reset_index(drop=True)
did.to_parquet(OUT_PANEL, index=False)

# ---------------------------------------------------------------------
# Pre-period descriptive comparison
# ---------------------------------------------------------------------

pre_summary = (
    did[did["post_congestion_pricing"].eq(0)]
    .groupby(["year_month", "bridge_group"], observed=True)["bridge_traffic"]
    .mean()
    .reset_index()
)

pre_pivot = (
    pre_summary.pivot(
        index="year_month",
        columns="bridge_group",
        values="bridge_traffic",
    )
    .reset_index()
)

if {
    "treated_all_directions",
    "comparison_all_directions",
}.issubset(pre_pivot.columns):
    pre_pivot["treated_minus_comparison"] = (
        pre_pivot["treated_all_directions"]
        - pre_pivot["comparison_all_directions"]
    )
    pre_pivot["treated_over_comparison"] = (
        pre_pivot["treated_all_directions"]
        / pre_pivot["comparison_all_directions"].replace(0, np.nan)
    )

pre_pivot.to_csv(OUT_PRETRENDS, index=False)

# ---------------------------------------------------------------------
# Models
#
# No C(bridge_group) term: it is identical to treated_group and would be
# redundant. The interaction below is the DiD term of interest.
# ---------------------------------------------------------------------

level_formula = """
bridge_traffic ~ post_congestion_pricing * treated_group
+ C(hour)
+ C(day_of_week)
+ C(year_month)
+ holiday_flag
+ severe_weather_flag
"""

log_formula = """
log_bridge_traffic ~ post_congestion_pricing * treated_group
+ C(hour)
+ C(day_of_week)
+ C(year_month)
+ holiday_flag
+ severe_weather_flag
"""

level_trend_formula = """
bridge_traffic ~ post_congestion_pricing * treated_group
+ treated_group:time_index
+ C(hour)
+ C(day_of_week)
+ C(year_month)
+ holiday_flag
+ severe_weather_flag
"""

log_trend_formula = """
log_bridge_traffic ~ post_congestion_pricing * treated_group
+ treated_group:time_index
+ C(hour)
+ C(day_of_week)
+ C(year_month)
+ holiday_flag
+ severe_weather_flag
"""

fit_kwargs = {
    "cov_type": "cluster",
    "cov_kwds": {"groups": did["date_cluster"]},
}

level_model = smf.ols(level_formula, data=did).fit(**fit_kwargs)
log_model = smf.ols(log_formula, data=did).fit(**fit_kwargs)
level_trend_model = smf.ols(level_trend_formula, data=did).fit(**fit_kwargs)
log_trend_model = smf.ols(log_trend_formula, data=did).fit(**fit_kwargs)

term = "post_congestion_pricing:treated_group"

main_results = pd.concat(
    [
        extract_terms(level_model, [term], "All-directions bridge DiD - level"),
        extract_terms(log_model, [term], "All-directions bridge DiD - log"),
        extract_terms(
            level_trend_model,
            [term],
            "All-directions bridge DiD - level trend",
        ),
        extract_terms(
            log_trend_model,
            [term],
            "All-directions bridge DiD - log trend",
        ),
    ],
    ignore_index=True,
)

log_rows = main_results["model"].str.contains("log", case=False, na=False)
main_results.loc[log_rows, "percent_effect"] = (
    100 * (np.exp(main_results.loc[log_rows, "coef"]) - 1)
)
main_results.to_csv(OUT_MAIN_RESULTS, index=False)

# ---------------------------------------------------------------------
# Human-readable results file
# ---------------------------------------------------------------------

with open(OUT_RESULTS, "w", encoding="utf-8") as f:
    f.write("=" * 90 + "\n")
    f.write("9ba All-Directions Bridge/Tunnel DiD Results\n")
    f.write("=" * 90 + "\n\n")

    f.write("Design\n")
    f.write("-" * 90 + "\n")
    f.write(
        "Treated group: all inbound and outbound directions at direct "
        "CRZ-relevant facilities.\n"
    )
    f.write(f"Treated facility IDs: {', '.join(sorted(TREATED_FACILITY_IDS))}\n")
    f.write(
        "Comparison group: all directions at the existing spillover facilities.\n"
    )
    f.write(
        f"Comparison facility IDs: {', '.join(sorted(COMPARISON_FACILITY_IDS))}\n"
    )
    f.write(
        "Interpretation: comparative robustness analysis. Spillover facilities "
        "may also be affected, so they are not assumed to be untouched controls.\n\n"
    )

    f.write("Inputs and outputs\n")
    f.write("-" * 90 + "\n")
    f.write(f"Bridge master: {BRIDGES_MASTER}\n")
    f.write(f"Event controls: {EVENT_PANEL}\n")
    f.write(f"Panel: {OUT_PANEL}\n")
    f.write(f"Facility inventory: {OUT_FACILITIES}\n")
    f.write(f"Pre-trend summary: {OUT_PRETRENDS}\n")
    f.write(f"Key results: {OUT_MAIN_RESULTS}\n\n")

    f.write("Specification\n")
    f.write("-" * 90 + "\n")
    f.write(
        "Outcome ~ post_congestion_pricing * treated_group + hour FE "
        "+ day-of-week FE + year-month FE + holiday_flag + severe_weather_flag\n"
    )
    f.write("Trend robustness: baseline + treated_group:time_index\n")
    f.write("Standard errors clustered by calendar date.\n\n")

    f.write("Sample\n")
    f.write("-" * 90 + "\n")
    f.write(
        f"{did['transit_timestamp'].min()} to "
        f"{did['transit_timestamp'].max()}\n"
    )
    f.write(f"Rows: {len(did):,}\n")
    f.write(f"Hours retained: {did['transit_timestamp'].nunique():,}\n\n")

    f.write("Facility/direction inventory\n")
    f.write("-" * 90 + "\n")
    f.write(facility_inventory.to_string(index=False))
    f.write("\n\n")

    f.write("Pre-period monthly group means\n")
    f.write("-" * 90 + "\n")
    f.write(pre_pivot.to_string(index=False))
    f.write("\n\n")

    display = main_results.copy()
    for col in ["coef", "std_err", "ci_low", "ci_high", "r_squared"]:
        display[col] = display[col].round(4)
    display["p_value"] = display["p_value"].round(4)
    if "percent_effect" in display.columns:
        display["percent_effect"] = display["percent_effect"].round(2)

    f.write("Main results: post_congestion_pricing:treated_group\n")
    f.write("-" * 90 + "\n")
    f.write(display.to_string(index=False))
    f.write("\n")

print("\n" + "=" * 90)
print("9ba all-directions bridge DiD complete")
print("=" * 90)
print(f"Saved panel:              {OUT_PANEL}")
print(f"Saved facility inventory: {OUT_FACILITIES}")
print(f"Saved pre-trend summary:  {OUT_PRETRENDS}")
print(f"Saved key results:        {OUT_MAIN_RESULTS}")
print(f"Saved results report:     {OUT_RESULTS}")
print("\nMain results")
print(main_results.to_string(index=False))
