from datetime import datetime
from pathlib import Path
import json

import pandas as pd

# -----------------------------
# Settings
# -----------------------------

RAW_DIR = Path("data/raw/citi_bike")
PROCESSED_DIR = Path("data/processed")

START = "2024_01"
END = "2026_03"

OUTPUT_FILE = "citibike_master_2024_01_2026_03.parquet"
FILE_PATTERN = "*citibike-tripdata*.csv*"

ALLOW_MISSING_MONTHS = False
OVERWRITE_EXISTING = False

# -----------------------------
# Helpers
# -----------------------------

def month_from_filename(path: Path) -> str:
    name = path.name
    digits = "".join(ch for ch in name if ch.isdigit())
    if len(digits) < 6:
        raise ValueError(f"Could not extract YYYYMM from filename: {path.name}")
    yyyymm = digits[:6]
    return f"{yyyymm[:4]}_{yyyymm[4:6]}"


def expected_months(start: str, end: str) -> list[str]:
    start_dt = pd.Timestamp(start.replace("_", "-") + "-01")
    end_dt = pd.Timestamp(end.replace("_", "-") + "-01")
    return [dt.strftime("%Y_%m") for dt in pd.date_range(start_dt, end_dt, freq="MS")]


def month_in_range(path: Path) -> bool:
    ym = month_from_filename(path)
    return START <= ym <= END


def find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"None of these columns found: {candidates}. "
        f"Existing columns: {list(df.columns)}"
    )


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

    df = pd.read_csv(path, low_memory=False)
    raw_rows = len(df)

    started_col = find_col(df, ["started_at", "starttime", "start_time"])
    ended_col = find_col(df, ["ended_at", "stoptime", "end_time"])
    start_station_id_col = find_col(df, ["start_station_id", "start station id"])
    start_station_name_col = find_col(df, ["start_station_name", "start station name"])
    end_station_id_col = find_col(df, ["end_station_id", "end station id"])
    end_station_name_col = find_col(df, ["end_station_name", "end station name"])

    optional_cols = [
        "rideable_type",
        "member_casual",
        "start_lat",
        "start_lng",
        "end_lat",
        "end_lng",
        "start station latitude",
        "start station longitude",
        "end station latitude",
        "end station longitude",
    ]

    keep_cols = [
        started_col,
        ended_col,
        start_station_id_col,
        start_station_name_col,
        end_station_id_col,
        end_station_name_col,
        *[col for col in optional_cols if col in df.columns],
    ]

    df = df[keep_cols].copy()

    df = df.rename(
        columns={
            started_col: "started_at",
            ended_col: "ended_at",
            start_station_id_col: "start_station_id",
            start_station_name_col: "start_station_name",
            end_station_id_col: "end_station_id",
            end_station_name_col: "end_station_name",
            "start station latitude": "start_lat",
            "start station longitude": "start_lng",
            "end station latitude": "end_lat",
            "end station longitude": "end_lng",
        }
    )

    df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")
    df["ended_at"] = pd.to_datetime(df["ended_at"], errors="coerce")

    df = df.dropna(
        subset=[
            "started_at",
            "ended_at",
            "start_station_id",
            "start_station_name",
            "end_station_id",
            "end_station_name",
        ]
    )

    df["duration_minutes"] = (
        df["ended_at"] - df["started_at"]
    ).dt.total_seconds() / 60

    df = df[
        (df["duration_minutes"] > 0)
        & (df["duration_minutes"] <= 240)
    ].copy()

    df["transit_timestamp"] = df["started_at"].dt.floor("h")
    df["source_month"] = df["transit_timestamp"].dt.strftime("%Y_%m")
    df = df[(df["source_month"] >= START) & (df["source_month"] <= END)].copy()

    if "member_casual" in df.columns:
        df["member_count"] = (df["member_casual"] == "member").astype("int64")
        df["casual_count"] = (df["member_casual"] == "casual").astype("int64")
    else:
        df["member_count"] = 0
        df["casual_count"] = 0

    if "rideable_type" in df.columns:
        df["classic_bike_count"] = (df["rideable_type"] == "classic_bike").astype("int64")
        df["electric_bike_count"] = (df["rideable_type"] == "electric_bike").astype("int64")
        df["docked_bike_count"] = (df["rideable_type"] == "docked_bike").astype("int64")
    else:
        df["classic_bike_count"] = 0
        df["electric_bike_count"] = 0
        df["docked_bike_count"] = 0

    for col in ["start_lat", "start_lng", "end_lat", "end_lng"]:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")

    grouped = (
        df.groupby(
            [
                "transit_timestamp",
                "start_station_id",
                "start_station_name",
                "end_station_id",
                "end_station_name",
            ],
            as_index=False,
        )
        .agg(
            ride_count=("transit_timestamp", "size"),
            member_count=("member_count", "sum"),
            casual_count=("casual_count", "sum"),
            classic_bike_count=("classic_bike_count", "sum"),
            electric_bike_count=("electric_bike_count", "sum"),
            docked_bike_count=("docked_bike_count", "sum"),
            avg_duration_minutes=("duration_minutes", "mean"),
            start_lat=("start_lat", "median"),
            start_lng=("start_lng", "median"),
            end_lat=("end_lat", "median"),
            end_lng=("end_lng", "median"),
        )
    )

    grouped["source_file"] = path.name
    grouped["source_month"] = month_from_filename(path)
    grouped = add_time_fields(grouped)

    print(f"  Raw rows: {raw_rows:,}")
    print(f"  Aggregated rows: {len(grouped):,}")
    print(f"  Rides: {int(grouped['ride_count'].sum()):,}")

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

    files = sorted([p for p in RAW_DIR.glob(FILE_PATTERN) if month_in_range(p)])
    found_months = [month_from_filename(p) for p in files]
    missing_months = [ym for ym in expected_months(START, END) if ym not in found_months]

    print(f"Files found: {len(files)}")

    if missing_months and not ALLOW_MISSING_MONTHS:
        raise RuntimeError(f"Missing Citi Bike months: {missing_months}")

    if missing_months:
        print(f"WARNING: Missing Citi Bike months: {missing_months}")

    if not files:
        raise FileNotFoundError(
            f"No Citi Bike files found in {RAW_DIR} matching "
            f"{FILE_PATTERN} between {START} and {END}"
        )

    bike = pd.concat([clean_one_file(file) for file in files], ignore_index=True)

    bike["transit_timestamp"] = pd.to_datetime(bike["transit_timestamp"], errors="coerce")
    if bike["transit_timestamp"].isna().any():
        raise ValueError("Found invalid transit_timestamp values in Citi Bike master")

    bike = bike.sort_values(
        ["transit_timestamp", "start_station_id", "end_station_id"]
    ).reset_index(drop=True)

    duplicate_rows = int(
        bike.duplicated(
            subset=[
                "transit_timestamp",
                "start_station_id",
                "end_station_id",
                "source_file",
            ]
        ).sum()
    )

    bike.to_parquet(output_path, index=False)

    metadata = {
        "dataset_name": "citibike",
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
        "row_count": len(bike),
        "duplicate_rows_on_check_keys": duplicate_rows,
        "ride_count_total": int(bike["ride_count"].sum()),
        "member_count_total": int(bike["member_count"].sum()),
        "casual_count_total": int(bike["casual_count"].sum()),
        "classic_bike_count_total": int(bike["classic_bike_count"].sum()),
        "electric_bike_count_total": int(bike["electric_bike_count"].sum()),
        "docked_bike_count_total": int(bike["docked_bike_count"].sum()),
        "min_timestamp": str(bike["transit_timestamp"].min()),
        "max_timestamp": str(bike["transit_timestamp"].max()),
        "columns": list(bike.columns),
    }

    write_metadata(metadata_path, metadata)

    print("\nDone.")
    print(f"Saved: {output_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Rows: {len(bike):,}")
    print(f"Rides: {int(bike['ride_count'].sum()):,}")
    print(f"Date range: {bike['transit_timestamp'].min()} to {bike['transit_timestamp'].max()}")

    if duplicate_rows:
        print(f"WARNING: {duplicate_rows:,} duplicate rows found on check keys")


if __name__ == "__main__":
    main()