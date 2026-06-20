from pathlib import Path
import pandas as pd
import duckdb

# -----------------------------
# Settings
# -----------------------------

PROCESSED_DIR = Path("data/processed")
MAPPING_DIR = Path("data/mappings")
OUTPUT_FILE = PROCESSED_DIR / "analysis_panel_hourly_2024_01_2026_03.parquet"

START_TS = "2024-01-01 00:00:00"
END_TS = "2026-03-31 23:00:00"
POLICY_START = pd.Timestamp("2025-01-05 00:00:00")

# -----------------------------
# Helpers
# -----------------------------


def base_time_panel() -> pd.DataFrame:
    """Create one row per hour for the full study period."""
    timestamps = pd.date_range(START_TS, END_TS, freq="h")
    df = pd.DataFrame({"transit_timestamp": timestamps})

    df["date"] = df["transit_timestamp"].dt.date
    df["year"] = df["transit_timestamp"].dt.year
    df["month"] = df["transit_timestamp"].dt.month
    df["hour"] = df["transit_timestamp"].dt.hour
    df["day_of_week"] = df["transit_timestamp"].dt.day_name()
    df["is_weekend"] = df["transit_timestamp"].dt.dayofweek >= 5
    df["post_congestion_pricing"] = df["transit_timestamp"] >= POLICY_START

    df["peak_period"] = "off_peak"
    df.loc[df["hour"].between(6, 9), "peak_period"] = "am_peak"
    df.loc[df["hour"].between(16, 19), "peak_period"] = "pm_peak"
    df.loc[df["hour"].between(0, 4), "peak_period"] = "overnight"

    return df


def read_parquet(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_parquet(path, columns=columns)


def aggregate_subway() -> pd.DataFrame:
    df = read_parquet(
        PROCESSED_DIR / "subway_master_2024_01_2026_03.parquet",
        columns=["transit_timestamp", "ridership", "transfers"],
    )

    out = (
        df.groupby("transit_timestamp", as_index=False)
        .agg(
            subway_ridership=("ridership", "sum"),
            subway_transfers=("transfers", "sum"),
        )
    )
    return out


def aggregate_bus() -> pd.DataFrame:
    df = read_parquet(
        PROCESSED_DIR / "bus_master_2024_01_2026_03.parquet",
        columns=["transit_timestamp", "ridership", "transfers"],
    )

    out = (
        df.groupby("transit_timestamp", as_index=False)
        .agg(
            bus_ridership=("ridership", "sum"),
            bus_transfers=("transfers", "sum"),
        )
    )
    return out


def aggregate_taxi() -> pd.DataFrame:
    df = read_parquet(
        PROCESSED_DIR / "taxi_master_2024_01_2026_03.parquet",
        columns=[
            "transit_timestamp",
            "trip_count",
            "passenger_count_sum",
            "total_trip_distance",
            "total_fare_amount",
            "total_amount_sum",
            "congestion_surcharge_sum",
        ],
    )

    out = (
        df.groupby("transit_timestamp", as_index=False)
        .agg(
            taxi_trips=("trip_count", "sum"),
            taxi_passengers=("passenger_count_sum", "sum"),
            taxi_total_distance=("total_trip_distance", "sum"),
            taxi_total_fare=("total_fare_amount", "sum"),
            taxi_total_amount=("total_amount_sum", "sum"),
            taxi_congestion_surcharge=("congestion_surcharge_sum", "sum"),
        )
    )
    return out

def aggregate_forhire() -> pd.DataFrame:
    path = PROCESSED_DIR / "forhire_master_2024_01_2026_03.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    p = str(path).replace("\\", "/")

    con = duckdb.connect()

    query = f"""
    SELECT
        transit_timestamp,

        SUM(forhire_trip_count) AS forhire_trips,
        SUM(total_trip_miles) AS forhire_total_trip_miles,
        SUM(total_base_passenger_fare) AS forhire_total_base_fare,
        SUM(tips_sum) AS forhire_tips,
        SUM(driver_pay_sum) AS forhire_driver_pay,
        SUM(congestion_surcharge_sum) AS forhire_congestion_surcharge,
        SUM(cbd_congestion_fee_sum) AS forhire_cbd_fee,

        SUM(shared_request_count) AS forhire_shared_requests,
        SUM(shared_match_count) AS forhire_shared_matches,

        SUM(wav_request_count) AS forhire_wav_requests,
        SUM(wav_match_count) AS forhire_wav_matches,

        SUM(uber_trip_count) AS uber_trips,
        SUM(lyft_trip_count) AS lyft_trips,
        SUM(via_trip_count) AS via_trips,
        SUM(juno_trip_count) AS juno_trips

    FROM read_parquet('{p}')

    GROUP BY transit_timestamp
    ORDER BY transit_timestamp
    """

    out = con.execute(query).fetchdf()
    out["transit_timestamp"] = pd.to_datetime(out["transit_timestamp"])

    return out

def aggregate_citibike() -> pd.DataFrame:
    df = read_parquet(
        PROCESSED_DIR / "citibike_master_2024_01_2026_03.parquet",
        columns=[
            "transit_timestamp",
            "ride_count",
            "member_count",
            "casual_count",
            "classic_bike_count",
            "electric_bike_count",
            "docked_bike_count",
        ],
    )

    out = (
        df.groupby("transit_timestamp", as_index=False)
        .agg(
            citibike_rides=("ride_count", "sum"),
            citibike_member_rides=("member_count", "sum"),
            citibike_casual_rides=("casual_count", "sum"),
            citibike_classic_rides=("classic_bike_count", "sum"),
            citibike_electric_rides=("electric_bike_count", "sum"),
            citibike_docked_rides=("docked_bike_count", "sum"),
        )
    )
    return out


def aggregate_bridges() -> pd.DataFrame:
    df = read_parquet(
        PROCESSED_DIR / "bridges_master_2024_01_2026_03.parquet",
        columns=[
            "transit_timestamp",
            "facility_id",
            "direction",
            "vehicle_class",
            "vehicle_class_category",
            "traffic_count",
        ],
    )

    # Optional merge with manually reviewed treatment map
    map_path = MAPPING_DIR / "bridge_treatment_map.csv"
    if map_path.exists():
        bridge_map = pd.read_csv(map_path)
        bridge_map["facility_id"] = bridge_map["facility_id"].astype(str)
        df["facility_id"] = df["facility_id"].astype(str)
        df = df.merge(
            bridge_map[["facility_id", "direction", "treatment_group", "exposure_type", "keep_first_pass"]],
            on=["facility_id", "direction"],
            how="left",
        )
    else:
        df["treatment_group"] = "unmapped"
        df["keep_first_pass"] = True

    # First-pass: total traffic and passenger vehicle traffic
    df["is_passenger_vehicle"] = df["vehicle_class"].astype(str).eq("31")

    total = (
        df.groupby("transit_timestamp", as_index=False)
        .agg(
            bridge_traffic_total=("traffic_count", "sum"),
            bridge_passenger_vehicle_traffic=(
                "traffic_count",
                lambda x: x[df.loc[x.index, "is_passenger_vehicle"]].sum(),
            ),
        )
    )

    # Group-level bridge volumes if mapping exists
    group = (
        df.groupby(["transit_timestamp", "treatment_group"], as_index=False)["traffic_count"]
        .sum()
        .pivot(index="transit_timestamp", columns="treatment_group", values="traffic_count")
        .reset_index()
    )

    group.columns = [
        "transit_timestamp" if col == "transit_timestamp" else f"bridge_traffic_{col}"
        for col in group.columns
    ]

    out = total.merge(group, on="transit_timestamp", how="left")
    return out


def aggregate_crz() -> pd.DataFrame:
    df = read_parquet(
        PROCESSED_DIR / "crz_master_2025_01_2026_03.parquet",
        columns=[
            "transit_timestamp",
            "vehicle_class",
            "detection_group",
            "detection_region",
            "crz_entries",
            "excluded_roadway_entries",
        ],
    )

    total = (
        df.groupby("transit_timestamp", as_index=False)
        .agg(
            crz_entries=("crz_entries", "sum"),
            crz_excluded_roadway_entries=("excluded_roadway_entries", "sum"),
        )
    )

    # Vehicle class breakdown
    vehicle = (
        df.groupby(["transit_timestamp", "vehicle_class"], as_index=False)["crz_entries"]
        .sum()
        .pivot(index="transit_timestamp", columns="vehicle_class", values="crz_entries")
        .reset_index()
    )

    vehicle.columns = [
        "transit_timestamp" if col == "transit_timestamp" else "crz_entries_" + str(col).lower().replace(" ", "_").replace("-", "_").replace("/", "_").replace(",", "")
        for col in vehicle.columns
    ]

    # Detection region breakdown
    region = (
        df.groupby(["transit_timestamp", "detection_region"], as_index=False)["crz_entries"]
        .sum()
        .pivot(index="transit_timestamp", columns="detection_region", values="crz_entries")
        .reset_index()
    )

    region.columns = [
        "transit_timestamp" if col == "transit_timestamp" else "crz_entries_region_" + str(col).lower().replace(" ", "_").replace("-", "_")
        for col in region.columns
    ]

    out = total.merge(vehicle, on="transit_timestamp", how="left")
    out = out.merge(region, on="transit_timestamp", how="left")
    return out


# -----------------------------
# Main
# -----------------------------


def main():
    print("Creating base hourly panel...")
    panel = base_time_panel()

    aggregators = [
        ("subway", aggregate_subway),
        ("bus", aggregate_bus),
        ("taxi", aggregate_taxi),
        ("forhire", aggregate_forhire),
        ("citibike", aggregate_citibike),
        ("bridges", aggregate_bridges),
        ("crz", aggregate_crz),
    ]

    for name, func in aggregators:
        print(f"Aggregating {name}...")
        agg = func()
        print(f"  Rows: {len(agg):,}")
        panel = panel.merge(agg, on="transit_timestamp", how="left")

    # CRZ does not exist before Jan 5, 2025; fill missing CRZ counts with zero.
    crz_cols = [col for col in panel.columns if col.startswith("crz_")]
    panel[crz_cols] = panel[crz_cols].fillna(0)

    # For transport outcomes, missing should usually be zero only if the timestamp exists but no activity exists.
    # Because all five source files cover the same period, fill remaining numeric NAs with 0 for hourly totals.
    numeric_cols = panel.select_dtypes(include="number").columns
    panel[numeric_cols] = panel[numeric_cols].fillna(0)

    panel = panel.sort_values("transit_timestamp").reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUTPUT_FILE, index=False)

    print("\nDone.")
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Rows: {len(panel):,}")
    print(f"Columns: {len(panel.columns):,}")
    print(f"Date range: {panel['transit_timestamp'].min()} to {panel['transit_timestamp'].max()}")
    print("\nColumns:")
    for col in panel.columns:
        print(f"- {col}")


if __name__ == "__main__":
    main()
