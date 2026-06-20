from pathlib import Path
import json

import pandas as pd
import duckdb

PROCESSED_DIR = Path("data/processed")

FILES = [
    "bridges_master_2024_01_2026_03.parquet",
    "subway_master_2024_01_2026_03.parquet",
    "bus_master_2024_01_2026_03.parquet",
    "taxi_master_2024_01_2026_03.parquet",
    "citibike_master_2024_01_2026_03.parquet",
    "forhire_master_2024_01_2026_03.parquet",
]

REQUIRED_COLUMNS = {
    "bridges": ["transit_timestamp", "traffic_count"],
    "subway": ["transit_timestamp", "ridership", "transfers"],
    "bus": ["transit_timestamp", "ridership", "transfers"],
    "taxi": ["transit_timestamp", "trip_count"],
    "forhire": ["transit_timestamp", "forhire_trip_count"],
    "citibike": ["transit_timestamp", "ride_count"],
}

GRAIN_COLUMNS = {
    "bridges": ["transit_timestamp", "facility_id", "direction", "payment_method", "vehicle_class"],
    "subway": ["transit_timestamp", "station_complex_id", "payment_method"],
    "bus": ["transit_timestamp", "bus_route", "payment_method"],
    "taxi": ["transit_timestamp", "pickup_location_id", "dropoff_location_id", "source_file"],
    "forhire": ["transit_timestamp", "pickup_location_id", "dropoff_location_id", "source_file"],
    "citibike": ["transit_timestamp", "start_station_id", "end_station_id", "source_file"],
}

NON_NEGATIVE_COLUMNS = [
    "traffic_count",
    "ridership",
    "transfers",
    "trip_count",
    "forhire_trip_count",
    "ride_count",
    "passenger_count_sum",
    "total_trip_distance",
    "total_amount_sum",
    "total_trip_miles",
    "total_base_passenger_fare",
    "driver_pay_sum",
    "cbd_congestion_fee_sum",
]


def dataset_key(filename: str) -> str:
    return filename.split("_master_")[0]

def validate_large_forhire_file(path: Path, metadata_path: Path):
    con = duckdb.connect()

    p = str(path).replace("\\", "/")

    print("Large file detected: using DuckDB validation")

    row_count = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{p}')"
    ).fetchone()[0]

    cols = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{p}')"
    ).fetchdf()

    print(f"Rows: {row_count:,}")
    print(f"Columns: {len(cols):,}")

    required = REQUIRED_COLUMNS["forhire"]
    existing_cols = set(cols["column_name"])
    missing_required = [col for col in required if col not in existing_cols]

    if missing_required:
        print(f"FAIL: Missing required columns: {missing_required}")
    else:
        print("PASS: Required columns present")

    ts = con.execute(
        f"""
        SELECT
            MIN(transit_timestamp) AS min_ts,
            MAX(transit_timestamp) AS max_ts,
            SUM(CASE WHEN transit_timestamp IS NULL THEN 1 ELSE 0 END) AS null_ts,
            SUM(
                CASE
                    WHEN EXTRACT(minute FROM transit_timestamp) != 0
                      OR EXTRACT(second FROM transit_timestamp) != 0
                    THEN 1 ELSE 0
                END
            ) AS non_hourly
        FROM read_parquet('{p}')
        """
    ).fetchone()

    print(f"Date start: {ts[0]}")
    print(f"Date end:   {ts[1]}")
    print(f"Null timestamps: {ts[2]:,}")
    print(f"Non-hourly timestamps: {ts[3]:,}")

    if ts[2] == 0 and ts[3] == 0:
        print("PASS: Timestamps look valid")
    else:
        print("FAIL: Timestamp problems found")

    grain_duplicates = con.execute(
        f"""
        SELECT COUNT(*) - COUNT(DISTINCT
            transit_timestamp || '|' ||
            pickup_location_id || '|' ||
            dropoff_location_id
        )
        FROM read_parquet('{p}')
        """
    ).fetchone()[0]

    print(f"Duplicate grain rows: {grain_duplicates:,}")

    if grain_duplicates == 0:
        print("PASS: No duplicate grain rows")
    else:
        print("WARNING: Duplicate grain rows found")

    print("\nNegative-value checks:")

    for col in NON_NEGATIVE_COLUMNS:
        if col in existing_cols:
            neg = con.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet('{p}')
                WHERE {col} < 0
                """
            ).fetchone()[0]

            print(f"{col}: {neg:,} negative values")

    if metadata_path.exists():
        try:
            json.loads(metadata_path.read_text(encoding="utf-8"))
            print(f"PASS: Metadata found and valid JSON: {metadata_path}")
        except json.JSONDecodeError:
            print(f"FAIL: Metadata exists but is not valid JSON: {metadata_path}")
    else:
        print(f"WARNING: Metadata missing: {metadata_path}")

def validate_file(filename: str):
    path = PROCESSED_DIR / filename
    metadata_path = path.with_suffix(".metadata.json")

    print("\n" + "=" * 70)
    print(f"VALIDATING: {filename}")
    print("=" * 70)

    if not path.exists():
        print(f"FAIL: File not found: {path}")
        return

    key = dataset_key(filename)

    if key == "forhire":
        validate_large_forhire_file(path, metadata_path)
        return

    df = pd.read_parquet(path)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    if df.empty:
        print("FAIL: File is empty")
        return

    required = REQUIRED_COLUMNS.get(key, [])
    missing_required = [col for col in required if col not in df.columns]

    if missing_required:
        print(f"FAIL: Missing required columns: {missing_required}")
    else:
        print("PASS: Required columns present")

    if "transit_timestamp" in df.columns:
        df["transit_timestamp"] = pd.to_datetime(df["transit_timestamp"], errors="coerce")

        null_timestamps = int(df["transit_timestamp"].isna().sum())
        non_hourly = int(
            (
                (df["transit_timestamp"].dt.minute != 0)
                | (df["transit_timestamp"].dt.second != 0)
                | (df["transit_timestamp"].dt.microsecond != 0)
            ).sum()
        )

        print(f"Date start: {df['transit_timestamp'].min()}")
        print(f"Date end:   {df['transit_timestamp'].max()}")
        print(f"Null timestamps: {null_timestamps:,}")
        print(f"Non-hourly timestamps: {non_hourly:,}")

        if null_timestamps == 0 and non_hourly == 0:
            print("PASS: Timestamps look valid")
        else:
            print("FAIL: Timestamp problems found")

    exact_duplicates = int(df.duplicated().sum())
    print(f"Exact duplicate rows: {exact_duplicates:,}")

    if exact_duplicates == 0:
        print("PASS: No exact duplicate rows")
    else:
        print("WARNING: Exact duplicate rows found")

    grain_cols = [col for col in GRAIN_COLUMNS.get(key, []) if col in df.columns]
    missing_grain_cols = [col for col in GRAIN_COLUMNS.get(key, []) if col not in df.columns]

    if missing_grain_cols:
        print(f"WARNING: Missing grain-check columns: {missing_grain_cols}")

    if grain_cols:
        grain_duplicates = int(df.duplicated(subset=grain_cols).sum())
        print(f"Duplicate grain rows: {grain_duplicates:,}")

        if grain_duplicates == 0:
            print("PASS: No duplicate grain rows")
        else:
            print("WARNING: Duplicate grain rows found")

    print("\nTop missing-value columns:")
    print(df.isna().sum().sort_values(ascending=False).head(10))

    print("\nNegative-value checks:")
    checked_any_negative = False

    for col in NON_NEGATIVE_COLUMNS:
        if col in df.columns:
            checked_any_negative = True
            numeric_col = pd.to_numeric(df[col], errors="coerce")
            negative_count = int((numeric_col < 0).sum())
            print(f"{col}: {negative_count:,} negative values")

    if not checked_any_negative:
        print("No configured non-negative columns found for this file")

    if metadata_path.exists():
        try:
            json.loads(metadata_path.read_text(encoding="utf-8"))
            print(f"PASS: Metadata found and valid JSON: {metadata_path}")
        except json.JSONDecodeError:
            print(f"FAIL: Metadata exists but is not valid JSON: {metadata_path}")
    else:
        print(f"WARNING: Metadata missing: {metadata_path}")


def main():
    for file in FILES:
        validate_file(file)

    print("\nValidation complete.")


if __name__ == "__main__":
    main()