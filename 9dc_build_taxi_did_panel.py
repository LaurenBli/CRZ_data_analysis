# 9dc_build_taxi_did_panel.py

import os
import pandas as pd

# ---------------------------------------------------------------------
# 9dc Build taxi DiD panel
#
# Input:
#   data/processed/taxi_master_with_crz_groups.parquet
#
# Output:
#   data/processed/taxi_did_panel.parquet
#
# Structure:
#   one row per hour × treated_group
# ---------------------------------------------------------------------

INPUT = "data/processed/taxi_master_with_crz_groups.parquet"
OUT_PANEL = "data/processed/taxi_did_panel.parquet"

os.makedirs("data/processed", exist_ok=True)

df = pd.read_parquet(INPUT)

required = {
    "transit_timestamp",
    "trip_count",
    "treated_group",
    "main_did_sample",
    "pickup_crz_group",
}

missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["transit_timestamp"] = pd.to_datetime(df["transit_timestamp"])
df["trip_count"] = pd.to_numeric(df["trip_count"], errors="coerce").fillna(0)
df["treated_group"] = df["treated_group"].astype(int)
df["main_did_sample"] = df["main_did_sample"].astype(int)

# Clean baseline DiD sample:
# excludes pickup-border zones.
df = df[df["main_did_sample"] == 1].copy()

# Aggregate to hour × treated_group.
did = (
    df.groupby(["transit_timestamp", "treated_group"], as_index=False)
    .agg(
        taxi_trips=("trip_count", "sum"),
        n_od_cells=("trip_count", "size"),
    )
)

did["taxi_zone_group"] = did["treated_group"].map(
    {
        1: "core_crz_pickup",
        0: "outside_crz_pickup",
    }
)

# Add time controls expected by 9dd_run_did_taxi.py.
did["date"] = pd.to_datetime(did["transit_timestamp"].dt.date)
did["hour"] = did["transit_timestamp"].dt.hour
did["day_of_week"] = did["transit_timestamp"].dt.day_name()
did["is_weekend"] = did["day_of_week"].isin(["Saturday", "Sunday"])

policy_start = pd.Timestamp("2025-01-05 00:00:00")

did["post_congestion_pricing"] = (
    did["transit_timestamp"] >= policy_start
).astype(int)

# Save.
did.to_parquet(
    OUT_PANEL,
    index=False,
)

print("=" * 90)
print("9dc taxi DiD panel build complete")
print("=" * 90)
print(f"Saved panel to: {OUT_PANEL}")
print()

print("Rows:")
print(f"{len(did):,}")
print()

print("Date range:")
print(did["transit_timestamp"].min(), "to", did["transit_timestamp"].max())
print()

print("Treated group counts:")
print(did["treated_group"].value_counts())
print()

print("Taxi zone groups:")
print(did["taxi_zone_group"].value_counts())
print()

print("Preview:")
print(did.head(10))
print()

print("Trip totals by group:")
print(
    did.groupby("taxi_zone_group")["taxi_trips"]
    .sum()
    .sort_values(ascending=False)
)
print()

full_hours = pd.date_range(
    did["transit_timestamp"].min(),
    did["transit_timestamp"].max(),
    freq="h",
)

observed_hours = did["transit_timestamp"].drop_duplicates()

missing_hours = full_hours.difference(observed_hours)

print("Expected hours:", len(full_hours))
print("Observed hours:", observed_hours.nunique())
print("Missing hours:")
print(missing_hours)
