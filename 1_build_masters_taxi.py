from datetime import datetime
from pathlib import Path
import json

import pandas as pd

# -----------------------------
# Settings
# -----------------------------

RAW_DIR = Path("data/raw/taxi")
PROCESSED_DIR = Path("data/processed")

START = "2024_01"
END = "2026_03"

OUTPUT_FILE = "taxi_master_2024_01_2026_03.parquet"
FILE_PATTERN = "yellow_tripdata_*.parquet"

ALLOW_MISSING_MONTHS = False
OVERWRITE_EXISTING = False

# -----------------------------
# Helpers
# -----------------------------

def month_from_filename(path: Path) -> str:
    ym = path.stem[-7:].replace("-", "_")

    if (
        len(ym) != 7
        or ym[4] != "_"
        or not ym[:4].isdigit()
        or not ym[5:7].isdigit()
    ):
        raise ValueError(f"Could not extract YYYY_MM from filename: {path.name}")

    return ym


def expected_months(start: str, end: str) -> list[str]:
    start_dt = pd.Timestamp(start.replace("_", "-") + "-01")
    end_dt = pd.Timestamp(end.replace("_", "-") + "-01")

    return [
        dt.strftime("%Y_%m")
        for dt in pd.date_range(start_dt, end_dt, freq="MS")
    ]


def month_in_range(path: Path) -> bool:
    ym = month_from_filename(path)
    return START <= ym <= END


def write_metadata(metadata_path: Path, metadata: dict) -> None:
    metadata_path.write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )


def add_time_fields(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = df["transit_timestamp"].dt.date
    df["hour"] = df["transit_timestamp"].dt.hour
    df["day_of_week"] = df["transit_timestamp"].dt.day_name()
    df["is_weekend"] = df["transit_timestamp"].dt.dayofweek >= 5
    return df


def clean_one_file(path: Path) -> pd.DataFrame:
    print(f"Reading {path.name}")

    df = pd.read_parquet(path)
    raw_rows = len(df)

    pickup_col = "tpep_pickup_datetime"
    dropoff_col = "tpep_dropoff_datetime"

    required = [
        pickup_col,
        dropoff_col,
        "PULocationID",
        "DOLocationID",
        "trip_distance",
        "fare_amount",
        "total_amount",
        "passenger_count",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing expected columns in {path.name}: {missing}"
        )

    optional_cols = [
        "congestion_surcharge",
        "airport_fee",
        "payment_type",
        "RatecodeID",
        "VendorID",
    ]

    keep_cols = required + [
        col for col in optional_cols
        if col in df.columns
    ]

    df = df[keep_cols].copy()

    df[pickup_col] = pd.to_datetime(
        df[pickup_col],
        errors="coerce",
    )

    df[dropoff_col] = pd.to_datetime(
        df[dropoff_col],
        errors="coerce",
    )

    numeric_cols = [
        "PULocationID",
        "DOLocationID",
        "trip_distance",
        "fare_amount",
        "total_amount",
        "passenger_count",
        *[col for col in optional_cols if col in df.columns],
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=[
            pickup_col,
            dropoff_col,
            "PULocationID",
            "DOLocationID",
        ]
    )

    df["duration_minutes"] = (
        df[dropoff_col] - df[pickup_col]
    ).dt.total_seconds() / 60

    df = df[
        (df["duration_minutes"] > 0)
        & (df["duration_minutes"] <= 240)
        & (df["trip_distance"] >= 0)
        & (df["trip_distance"] <= 100)
        & (df["total_amount"] >= 0)
    ].copy()

    df["transit_timestamp"] = df[pickup_col].dt.floor("h")
    df["source_month"] = df["transit_timestamp"].dt.strftime("%Y_%m")

    df = df[
        (df["source_month"] >= START)
        & (df["source_month"] <= END)
    ].copy()

    for col in ["congestion_surcharge", "airport_fee"]:
        if col not in df.columns:
            df[col] = 0

    grouped = (
        df.groupby(
            [
                "transit_timestamp",
                "PULocationID",
                "DOLocationID",
            ],
            as_index=False,
        )
        .agg(
            trip_count=("transit_timestamp", "size"),
            passenger_count_sum=("passenger_count", "sum"),
            avg_trip_distance=("trip_distance", "mean"),
            total_trip_distance=("trip_distance", "sum"),
            avg_duration_minutes=("duration_minutes", "mean"),
            avg_fare_amount=("fare_amount", "mean"),
            total_fare_amount=("fare_amount", "sum"),
            avg_total_amount=("total_amount", "mean"),
            total_amount_sum=("total_amount", "sum"),
            congestion_surcharge_sum=("congestion_surcharge", "sum"),
            airport_fee_sum=("airport_fee", "sum"),
        )
    )

    grouped = grouped.rename(
        columns={
            "PULocationID": "pickup_location_id",
            "DOLocationID": "dropoff_location_id",
        }
    )

    grouped["source_file"] = path.name
    grouped["source_month"] = month_from_filename(path)

    grouped = add_time_fields(grouped)

    print(f"  Raw rows: {raw_rows:,}")
    print(f"  Aggregated rows: {len(grouped):,}")
    print(f"  Trips: {int(grouped['trip_count'].sum()):,}")

    return grouped


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_DIR / OUTPUT_FILE
    metadata_path = output_path.with_suffix(".metadata.json")

    if output_path.exists() and not OVERWRITE_EXISTING:
        print(f"Skipping existing master: {output_path}")
        return

    files = sorted(
        [p for p in RAW_DIR.glob(FILE_PATTERN) if month_in_range(p)]
    )

    found_months = [month_from_filename(p) for p in files]

    missing_months = [
        ym for ym in expected_months(START, END)
        if ym not in found_months
    ]

    print(f"Files found: {len(files)}")

    if missing_months and not ALLOW_MISSING_MONTHS:
        raise RuntimeError(f"Missing taxi months: {missing_months}")

    if missing_months:
        print(f"WARNING: Missing taxi months: {missing_months}")

    if not files:
        raise FileNotFoundError(
            f"No taxi files found in {RAW_DIR} matching "
            f"{FILE_PATTERN} between {START} and {END}"
        )

    taxi = pd.concat(
        [clean_one_file(file) for file in files],
        ignore_index=True,
    )

    taxi["transit_timestamp"] = pd.to_datetime(
        taxi["transit_timestamp"],
        errors="coerce",
    )

    if taxi["transit_timestamp"].isna().any():
        raise ValueError("Found invalid transit_timestamp values in taxi master")

    duplicate_rows = int(
        taxi.duplicated(
            subset=[
                "transit_timestamp",
                "pickup_location_id",
                "dropoff_location_id",
                "source_file",
            ]
        ).sum()
    )

    taxi = taxi.sort_values(
        [
            "transit_timestamp",
            "pickup_location_id",
            "dropoff_location_id",
        ]
    ).reset_index(drop=True)

    taxi.to_parquet(output_path, index=False)

    metadata = {
        "dataset_name": "taxi",
        "output_file": str(output_path),
        "source_folder": str(RAW_DIR),
        "file_pattern": FILE_PATTERN,
        "start_month": START,
        "end_month": END,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "files_used": [str(p) for p in files],
        "file_count": len(files),
        "months_found": found_months,
        "missing_months": missing_months,
        "allow_missing_months": ALLOW_MISSING_MONTHS,
        "overwrite_existing": OVERWRITE_EXISTING,
        "row_count": len(taxi),
        "duplicate_rows_on_check_keys": duplicate_rows,
        "trip_count_total": int(taxi["trip_count"].sum()),
        "passenger_count_total": int(taxi["passenger_count_sum"].sum()),
        "total_trip_distance": float(taxi["total_trip_distance"].sum()),
        "total_fare_amount": float(taxi["total_fare_amount"].sum()),
        "total_amount_sum": float(taxi["total_amount_sum"].sum()),
        "congestion_surcharge_sum": float(taxi["congestion_surcharge_sum"].sum()),
        "airport_fee_sum": float(taxi["airport_fee_sum"].sum()),
        "min_timestamp": str(taxi["transit_timestamp"].min()),
        "max_timestamp": str(taxi["transit_timestamp"].max()),
        "columns": list(taxi.columns),
    }

    write_metadata(metadata_path, metadata)

    print("\nDone.")
    print(f"Saved: {output_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Rows: {len(taxi):,}")
    print(f"Trips: {int(taxi['trip_count'].sum()):,}")

    print(
        f"Date range: "
        f"{taxi['transit_timestamp'].min()} to "
        f"{taxi['transit_timestamp'].max()}"
    )

    if duplicate_rows:
        print(
            f"WARNING: {duplicate_rows:,} duplicate rows found "
            f"on check keys"
        )


if __name__ == "__main__":
    main()