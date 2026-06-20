# 9ec_build_bus_did_panel.py

import os
import pandas as pd

INPUT = "data/processed/bus_master_with_crz_groups.parquet"
OUT_PANEL = "data/processed/bus_did_panel.parquet"

os.makedirs("data/processed", exist_ok=True)

print("=" * 90)
print("Loading bus master with CRZ groups...")
print("=" * 90)

df = pd.read_parquet(INPUT)

required = {
    "transit_timestamp",
    "bus_route",
    "ridership",
    "treated_group",
    "bus_route_group",
    "main_did_sample",
}

missing = required - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns in {INPUT}: {missing}"
    )

did = df[df["main_did_sample"] == 1].copy()

did["transit_timestamp"] = pd.to_datetime(did["transit_timestamp"])
did["ridership"] = pd.to_numeric(did["ridership"], errors="coerce").fillna(0)
did["treated_group"] = did["treated_group"].astype(int)

did["date"] = pd.to_datetime(did["transit_timestamp"].dt.date)
did["hour"] = did["transit_timestamp"].dt.hour
did["day_of_week"] = did["transit_timestamp"].dt.day_name()
did["is_weekend"] = did["transit_timestamp"].dt.dayofweek >= 5

policy_start = pd.Timestamp("2025-01-05 00:00:00")

did["post_congestion_pricing"] = (
    did["transit_timestamp"] >= policy_start
).astype(int)

panel = (
    did.groupby(
        [
            "transit_timestamp",
            "treated_group",
            "bus_route_group",
        ],
        as_index=False,
    )
    .agg(
        bus_ridership=("ridership", "sum"),
        n_routes=("bus_route", "nunique"),
    )
)

panel["date"] = pd.to_datetime(panel["transit_timestamp"].dt.date)
panel["hour"] = panel["transit_timestamp"].dt.hour
panel["day_of_week"] = panel["transit_timestamp"].dt.day_name()
panel["is_weekend"] = panel["transit_timestamp"].dt.dayofweek >= 5

panel["post_congestion_pricing"] = (
    panel["transit_timestamp"] >= policy_start
).astype(int)

panel = (
    panel.sort_values(
        [
            "transit_timestamp",
            "treated_group",
        ]
    )
    .reset_index(drop=True)
)

panel.to_parquet(OUT_PANEL, index=False)

print("=" * 90)
print("9ec bus DiD panel build complete")
print("=" * 90)
print(f"Saved panel to: {OUT_PANEL}")
print()

print("Rows:")
print(f"{len(panel):,}")
print()

print("Date range:")
print(
    panel["transit_timestamp"].min(),
    "to",
    panel["transit_timestamp"].max(),
)
print()

print("Treated group counts:")
print(panel["treated_group"].value_counts(dropna=False))
print()

print("Bus route groups:")
print(panel["bus_route_group"].value_counts(dropna=False))
print()

print("Ridership totals by group:")
print(
    panel.groupby("bus_route_group")["bus_ridership"]
    .sum()
)
print()

print("Route counts by group:")
print(
    panel.groupby("bus_route_group")["n_routes"]
    .max()
)
print()

full_hours = pd.date_range(
    panel["transit_timestamp"].min(),
    panel["transit_timestamp"].max(),
    freq="h",
)

observed_hours = panel["transit_timestamp"].drop_duplicates()
missing_hours = full_hours.difference(observed_hours)

print("Expected hours:", len(full_hours))
print("Observed hours:", observed_hours.nunique())
print("Missing hours:")
print(missing_hours)
print()

print("Preview:")
print(panel.head(10))
