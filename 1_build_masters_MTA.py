from datetime import datetime
from pathlib import Path
import json

import pandas as pd

# -----------------------------
# Settings
# -----------------------------

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

START = "2024_01"
END = "2026_03"

ALLOW_MISSING_MONTHS = False
OVERWRITE_EXISTING = False

MASTER_BUILDS = [
    {
        "name": "bridges",
        "input_folder": RAW_DIR / "mta_bridges_tunnels",
        "pattern": "bridges_hourly_facility_*.parquet",
        "output_name": "bridges_master_2024_01_2026_03.parquet",
        "metric_columns": ["traffic_count"],
        "sort_columns": [
            "transit_timestamp",
            "facility_id",
            "direction",
            "payment_method",
            "vehicle_class",
        ],
        "grain_columns": [
            "transit_timestamp",
            "facility_id",
            "direction",
            "payment_method",
            "vehicle_class",
        ],
        "start": START,
        "end": END,
    },
    {
        "name": "subway",
        "input_folder": RAW_DIR / "mta_subway",
        "pattern": "subway_hourly_station_*.parquet",
        "output_name": "subway_master_2024_01_2026_03.parquet",
        "metric_columns": ["ridership", "transfers"],
        "sort_columns": [
            "transit_timestamp",
            "station_complex_id",
            "payment_method",
        ],
        "grain_columns": [
            "transit_timestamp",
            "station_complex_id",
            "payment_method",
        ],
        "start": START,
        "end": END,
    },
    {
        "name": "bus",
        "input_folder": RAW_DIR / "mta_bus",
        "pattern": "bus_hourly_route_*.parquet",
        "output_name": "bus_master_2024_01_2026_03.parquet",
        "metric_columns": ["ridership", "transfers"],
        "sort_columns": [
            "transit_timestamp",
            "bus_route",
            "payment_method",
        ],
        "grain_columns": [
            "transit_timestamp",
            "bus_route",
            "payment_method",
        ],
        "start": START,
        "end": END,
    },
    {
        "name": "crz",
        "input_folder": RAW_DIR / "mta_crz",
        "pattern": "crz_hourly_entries_*.parquet",
        "output_name": "crz_master_2025_01_2026_03.parquet",
        "metric_columns": ["crz_entries", "excluded_roadway_entries"],
        "sort_columns": [
            "transit_timestamp",
            "detection_group",
            "vehicle_class",
        ],
        "grain_columns": [
            "transit_timestamp",
            "time_period",
            "vehicle_class",
            "detection_group",
            "detection_region",
        ],
        "start": "2025_01",
        "end": END,
    },
]

# -----------------------------
# Helpers
# -----------------------------

def month_from_parquet_name(path: Path) -> str:
    ym = path.stem[-7:]

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


def month_in_range(path: Path, start: str, end: str) -> bool:
    ym = month_from_parquet_name(path)
    return start <= ym <= end


def validate_required_columns(
    df: pd.DataFrame,
    required: list[str],
    source_name: str,
) -> None:
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns in {source_name}: {missing}")


def safe_sort_columns(df: pd.DataFrame, preferred_columns: list[str]) -> list[str]:
    return [col for col in preferred_columns if col in df.columns]


def write_metadata(metadata_path: Path, metadata: dict) -> None:
    metadata_path.write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )


def build_master(config: dict) -> None:
    name = config["name"]
    input_folder = Path(config["input_folder"])
    pattern = config["pattern"]
    output_name = config["output_name"]
    metric_columns = config["metric_columns"]
    sort_columns = config["sort_columns"]
    grain_columns = config["grain_columns"]
    start = config["start"]
    end = config["end"]

    output_path = PROCESSED_DIR / output_name
    metadata_path = output_path.with_suffix(".metadata.json")

    print(f"\nBuilding {output_name}")

    if output_path.exists() and not OVERWRITE_EXISTING:
        print(f"Skipping existing master: {output_path}")
        return

    files = sorted(
        [p for p in input_folder.glob(pattern) if month_in_range(p, start, end)]
    )

    found_months = [month_from_parquet_name(p) for p in files]
    missing_months = [
        ym for ym in expected_months(start, end)
        if ym not in found_months
    ]

    print(f"Files found: {len(files)}")

    if missing_months and not ALLOW_MISSING_MONTHS:
        raise RuntimeError(f"Missing months for {name}: {missing_months}")

    if missing_months:
        print(f"WARNING: Missing months for {name}: {missing_months}")

    if not files:
        raise FileNotFoundError(
            f"No files found for {output_name} in {input_folder} using {pattern}"
        )

    dfs = []

    for file in files:
        print(f"Reading {file.name}")

        monthly = pd.read_parquet(file)

        validate_required_columns(
            monthly,
            ["transit_timestamp", *metric_columns, *grain_columns],
            file.name,
        )

        monthly["source_file"] = file.name
        monthly["source_month"] = month_from_parquet_name(file)

        dfs.append(monthly)

    df = pd.concat(dfs, ignore_index=True)

    df["transit_timestamp"] = pd.to_datetime(
        df["transit_timestamp"],
        errors="coerce",
    )

    if df["transit_timestamp"].isna().any():
        bad_rows = int(df["transit_timestamp"].isna().sum())
        raise ValueError(
            f"Found {bad_rows:,} rows with invalid transit_timestamp in {output_name}"
        )

    for col in metric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        if (df[col] < 0).any():
            raise ValueError(f"Found negative values in {col} for {output_name}")

    duplicate_check_columns = [
        col for col in grain_columns
        if col in df.columns
    ]

    duplicate_rows = (
        int(df.duplicated(subset=duplicate_check_columns).sum())
        if duplicate_check_columns
        else 0
    )

    sort_cols = safe_sort_columns(df, sort_columns)

    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    else:
        df = df.sort_values("transit_timestamp").reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    metric_totals = {
        f"{col}_total": int(df[col].sum())
        for col in metric_columns
    }

    df.to_parquet(output_path, index=False)

    metadata = {
        "dataset_name": name,
        "output_file": str(output_path),
        "source_folder": str(input_folder),
        "file_pattern": pattern,
        "start_month": start,
        "end_month": end,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "files_used": [str(p) for p in files],
        "file_count": len(files),
        "months_found": found_months,
        "missing_months": missing_months,
        "allow_missing_months": ALLOW_MISSING_MONTHS,
        "overwrite_existing": OVERWRITE_EXISTING,
        "row_count": len(df),
        "duplicate_rows_on_grain_keys": duplicate_rows,
        "grain_columns": grain_columns,
        "min_timestamp": str(df["transit_timestamp"].min()),
        "max_timestamp": str(df["transit_timestamp"].max()),
        "columns": list(df.columns),
        **metric_totals,
    }

    write_metadata(metadata_path, metadata)

    print(f"Saved: {output_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Rows: {len(df):,}")
    print(
        f"Date range: "
        f"{df['transit_timestamp'].min()} to "
        f"{df['transit_timestamp'].max()}"
    )

    for key, value in metric_totals.items():
        print(f"{key}: {value:,}")

    if duplicate_rows:
        print(
            f"WARNING: {duplicate_rows:,} duplicate rows found "
            f"on grain/check keys"
        )


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    for config in MASTER_BUILDS:
        build_master(config)


if __name__ == "__main__":
    main()