import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/processed/lookups")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------
# BRIDGES
# -------------------

df = pd.read_parquet(
    PROCESSED_DIR / "bridges_master_2024_01_2026_03.parquet",
    columns=[
        "facility_id",
        "facility",
        "direction",
        "vehicle_class_category"
    ]
)

df.drop_duplicates().sort_values(
    ["facility_id", "direction"]
).to_csv(
    OUTPUT_DIR / "unique_bridges.csv",
    index=False
)

print("Saved unique_bridges.csv")


# -------------------
# SUBWAY
# -------------------

df = pd.read_parquet(
    PROCESSED_DIR / "subway_master_2024_01_2026_03.parquet",
    columns=[
        "station_complex_id",
        "station_complex",
        "borough"
    ]
)

df.drop_duplicates().sort_values(
    ["station_complex_id"]
).to_csv(
    OUTPUT_DIR / "unique_subway_stations.csv",
    index=False
)

print("Saved unique_subway_stations.csv")


# -------------------
# BUS
# -------------------

df = pd.read_parquet(
    PROCESSED_DIR / "bus_master_2024_01_2026_03.parquet",
    columns=[
        "bus_route"
    ]
)

df.drop_duplicates().sort_values(
    ["bus_route"]
).to_csv(
    OUTPUT_DIR / "unique_bus_routes.csv",
    index=False
)

print("Saved unique_bus_routes.csv")

# -------------------
# CITIBIKE
# -------------------

df = pd.read_parquet(
    PROCESSED_DIR / "citibike_master_2024_01_2026_03.parquet",
    columns=[
        "start_station_id",
        "start_station_name",
        "start_lat",
        "start_lng",
        "end_station_id",
        "end_station_name",
        "end_lat",
        "end_lng"
    ]
)

# Start stations
start_df = df[
    [
        "start_station_id",
        "start_station_name",
        "start_lat",
        "start_lng"
    ]
].rename(
    columns={
        "start_station_id": "station_id",
        "start_station_name": "station_name",
        "start_lat": "lat",
        "start_lng": "lng"
    }
)

# End stations
end_df = df[
    [
        "end_station_id",
        "end_station_name",
        "end_lat",
        "end_lng"
    ]
].rename(
    columns={
        "end_station_id": "station_id",
        "end_station_name": "station_name",
        "end_lat": "lat",
        "end_lng": "lng"
    }
)

# Combine both so you get full station universe
stations_df = pd.concat(
    [start_df, end_df],
    ignore_index=True
)

stations_df = (
    stations_df
    .sort_values(["station_id"])
    .groupby("station_id", as_index=False)
    .agg(
        station_name=("station_name", "first"),
        lat=("lat", "median"),
        lng=("lng", "median"),
    )
)

stations_df.to_csv(
    OUTPUT_DIR / "unique_citibike_stations.csv",
    index=False
)

print("Saved unique_citibike_stations.csv")

# -------------------
# TAXI
# -------------------

df = pd.read_parquet(
    PROCESSED_DIR / "taxi_master_2024_01_2026_03.parquet",
    columns=[
        "pickup_location_id",
        "dropoff_location_id"
    ]
)

# Pickup zones
pickup_df = df[
    ["pickup_location_id"]
].rename(
    columns={
        "pickup_location_id": "location_id"
    }
)

# Dropoff zones
dropoff_df = df[
    ["dropoff_location_id"]
].rename(
    columns={
        "dropoff_location_id": "location_id"
    }
)

# Combine full taxi zone universe
taxi_zones_df = pd.concat(
    [pickup_df, dropoff_df],
    ignore_index=True
)

taxi_zones_df = (
    taxi_zones_df
    .drop_duplicates()
    .sort_values(["location_id"])
)

taxi_zones_df.to_csv(
    OUTPUT_DIR / "unique_taxi_zones.csv",
    index=False
)

print("Saved unique_taxi_zones.csv")
print(f"Total unique taxi zones: {len(taxi_zones_df):,}")


