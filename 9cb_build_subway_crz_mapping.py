# 9cb_build_subway_crz_mapping.py

import os
import requests
import pandas as pd

INPUT = "data/processed/subway_master_2024_01_2026_03.parquet"

DATASET_ID = "39hk-dx4f"
STATIONS_URL = f"https://data.ny.gov/resource/{DATASET_ID}.json"

OUT_SUBWAY = "data/processed/subway_master_with_crz_groups.parquet"
OUT_MAPPING = "data/mappings/subway_crz_station_mapping.csv"

os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/mappings", exist_ok=True)

# ---------------------------------------------------------------------
# Load subway master
# ---------------------------------------------------------------------
df = pd.read_parquet(INPUT)

required = {
    "transit_timestamp",
    "station_complex_id",
    "station_complex",
    "borough",
    "ridership",
    "transfers",
}

missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in subway panel: {missing}")

# ---------------------------------------------------------------------
# Load official MTA subway station reference
# ---------------------------------------------------------------------
print("=" * 90)
print("Loading MTA Subway Stations reference file...")
print("=" * 90)

params = {
    "$select": """
        gtfs_stop_id,
        station_id,
        complex_id,
        division,
        line,
        stop_name,
        borough,
        cbd,
        daytime_routes,
        structure,
        gtfs_latitude,
        gtfs_longitude
    """,
    "$limit": 5000,
}

response = requests.get(STATIONS_URL, params=params, timeout=300)
response.raise_for_status()

stations = pd.DataFrame.from_records(response.json())

print("Station file shape:")
print(stations.shape)
print()

print("Columns:")
print(list(stations.columns))
print()

# ---------------------------------------------------------------------
# Build complex-level CRZ mapping
# ---------------------------------------------------------------------
stations["complex_id"] = pd.to_numeric(stations["complex_id"], errors="coerce")
stations["gtfs_latitude"] = pd.to_numeric(stations["gtfs_latitude"], errors="coerce")
stations["gtfs_longitude"] = pd.to_numeric(stations["gtfs_longitude"], errors="coerce")

mapping = stations[
    [
        "complex_id",
        "stop_name",
        "borough",
        "cbd",
        "gtfs_latitude",
        "gtfs_longitude",
    ]
].copy()

mapping = mapping.dropna(subset=["complex_id"]).copy()
mapping["complex_id"] = mapping["complex_id"].astype(int)

mapping["cbd"] = mapping["cbd"].astype(str).str.upper().str.strip()

mapping["core_crz"] = (mapping["cbd"] == "TRUE").astype(int)
mapping["treated_group"] = mapping["core_crz"]

mapping["crz_group"] = mapping["core_crz"].map(
    {
        1: "core_crz",
        0: "outside_crz",
    }
)

mapping = (
    mapping.groupby("complex_id", as_index=False)
    .agg(
        stop_name=("stop_name", "first"),
        borough=("borough", "first"),
        cbd=("cbd", lambda x: x.mode().iloc[0]),
        gtfs_latitude=("gtfs_latitude", "median"),
        gtfs_longitude=("gtfs_longitude", "median"),
        core_crz=("core_crz", "max"),
        treated_group=("treated_group", "max"),
    )
)

mapping["crz_group"] = mapping["core_crz"].map({
    1: "core_crz",
    0: "outside_crz",
})
# ---------------------------------------------------------------------
# Merge mapping onto subway master
# ---------------------------------------------------------------------
df["station_complex_id"] = pd.to_numeric(
    df["station_complex_id"],
    errors="coerce",
)

df = df.dropna(subset=["station_complex_id"]).copy()
df["station_complex_id"] = df["station_complex_id"].astype(int)

df = df.merge(
    mapping,
    left_on="station_complex_id",
    right_on="complex_id",
    how="left",
    suffixes=("", "_mta"),
)

# unmatched stations default to outside CRZ
df["core_crz"] = df["core_crz"].fillna(0).astype(int)
df["treated_group"] = df["treated_group"].fillna(0).astype(int)
df["crz_group"] = df["crz_group"].fillna("outside_crz")

# no border group needed because MTA CBD field already defines the zone
df["main_did_sample"] = 1
df["subway_station_group"] = df["crz_group"]

# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------
df.to_parquet(OUT_SUBWAY, index=False)
mapping.to_csv(OUT_MAPPING, index=False)

# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------
print("=" * 90)
print("9cb Subway CRZ treatment groups complete")
print("=" * 90)
print(f"Saved subway file to: {OUT_SUBWAY}")
print(f"Saved mapping file to: {OUT_MAPPING}")
print()

print("Rows:")
print(f"{len(df):,}")
print()

print("Treated group counts:")
print(df["treated_group"].value_counts(dropna=False))
print()

print("CRZ group counts:")
print(df["crz_group"].value_counts(dropna=False))
print()

print("Main DiD sample counts:")
print(df["main_did_sample"].value_counts(dropna=False))
print()

print("Unique station complexes in subway master:")
print(df["station_complex_id"].nunique())
print()

print("Unique station complexes in mapping:")
print(mapping["complex_id"].nunique())
print()

print("Mapping CRZ counts:")
print(mapping["crz_group"].value_counts(dropna=False))
print()

print("Unmatched subway rows:")
print(df["complex_id"].isna().sum())
print()

print("Unmatched subway station complexes:")
unmatched = (
    df[df["complex_id"].isna()]
    [["station_complex_id", "station_complex", "borough"]]
    .drop_duplicates()
    .sort_values("station_complex_id")
)

print(unmatched.head(50))
print(f"Total unmatched complexes: {len(unmatched):,}")