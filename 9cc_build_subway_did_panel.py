# 9cc_build_subway_did_panel.py

import os
import pandas as pd

INPUT = "data/processed/subway_master_with_crz_groups.parquet"
OUT_PANEL = "data/processed/subway_did_panel.parquet"

os.makedirs("data/processed", exist_ok=True)

df = pd.read_parquet(INPUT)

required = {
    "transit_timestamp",
    "ridership",
    "transfers",
    "treated_group",
    "subway_station_group",
}

missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["transit_timestamp"] = pd.to_datetime(df["transit_timestamp"])
df["ridership"] = pd.to_numeric(df["ridership"], errors="coerce").fillna(0)
df["transfers"] = pd.to_numeric(df["transfers"], errors="coerce").fillna(0)
df["treated_group"] = df["treated_group"].astype(int)

panel = (
    df.groupby(
        [
            "transit_timestamp",
            "treated_group",
            "subway_station_group",
        ],
        as_index=False,
    )
    .agg(
        subway_ridership=("ridership", "sum"),
        subway_transfers=("transfers", "sum"),
        n_station_payment_rows=("ridership", "size"),
        n_station_complexes=("station_complex_id", "nunique"),
    )
)

panel["date"] = pd.to_datetime(panel["transit_timestamp"].dt.date)
panel["hour"] = panel["transit_timestamp"].dt.hour
panel["day_of_week"] = panel["transit_timestamp"].dt.day_name()
panel["is_weekend"] = panel["transit_timestamp"].dt.dayofweek >= 5

policy_start = pd.Timestamp("2025-01-05 00:00:00")
panel["post_congestion_pricing"] = (
    panel["transit_timestamp"] >= policy_start
).astype(int)

panel.to_parquet(OUT_PANEL, index=False)

print("=" * 90)
print("9cc subway DiD panel build complete")
print("=" * 90)
print(f"Saved panel to: {OUT_PANEL}")
print()

print("Rows:")
print(f"{len(panel):,}")
print()

print("Date range:")
print(panel["transit_timestamp"].min(), "to", panel["transit_timestamp"].max())
print()

print("Treated group counts:")
print(panel["treated_group"].value_counts(dropna=False))
print()

print("Subway station groups:")
print(panel["subway_station_group"].value_counts(dropna=False))
print()

print("Trip/ridership totals by group:")
print(panel.groupby("subway_station_group")["subway_ridership"].sum())
print()

print("Station-complex counts by group:")
print(panel.groupby("subway_station_group")["n_station_complexes"].median())
print()

print("Preview:")
print(panel.head(10))