from pathlib import Path
from datetime import datetime
import json
import time

import requests
import pandas as pd

# -----------------------------
# Settings
# -----------------------------

DATASET_ID = "gxb3-akrn"  # MTA Bus Hourly Ridership: Beginning 2025
BASE_URL = f"https://data.ny.gov/resource/{DATASET_ID}.json"

START_MONTH = "2025-01-01"
END_MONTH = None

OUTPUT_DIR = Path("data/raw/mta_bus")

APP_TOKEN = None

LIMIT = 1_000_000
SLEEP_SECONDS = 1

INCLUDE_CURRENT_MONTH = False
OVERWRITE_EXISTING = False

# -----------------------------
# Helpers
# -----------------------------

def month_starts(
    start: str,
    end: str | None = None,
    include_current_month: bool = False,
):
    start_dt = pd.Timestamp(start).replace(day=1)

    if end is None:
        end_dt = pd.Timestamp.today().replace(day=1)
        if not include_current_month:
            end_dt = end_dt - pd.DateOffset(months=1)
    else:
        end_dt = pd.Timestamp(end).replace(day=1)

    if end_dt < start_dt:
        return []

    return list(pd.date_range(start=start_dt, end=end_dt, freq="MS"))


def write_metadata(metadata_path: Path, metadata: dict):
    metadata_path.write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )


def validate_month(df: pd.DataFrame, month_start: pd.Timestamp, ym: str):
    month_end = month_start + pd.DateOffset(months=1)

    if df["transit_timestamp"].isna().any():
        raise ValueError(f"{ym} has null timestamps")

    if df["transit_timestamp"].min() < month_start:
        raise ValueError(f"{ym} includes data before month start")

    if df["transit_timestamp"].max() >= month_end:
        raise ValueError(f"{ym} includes data after month end")

    for col in ["ridership", "transfers"]:
        if df[col].isna().any():
            raise ValueError(f"{ym} has null {col} values")

        if (df[col] < 0).any():
            raise ValueError(f"{ym} has negative {col} values")


def pull_one_month(month_start: pd.Timestamp) -> pd.DataFrame:
    month_end = month_start + pd.DateOffset(months=1)

    start_str = month_start.strftime("%Y-%m-%dT00:00:00")
    end_str = month_end.strftime("%Y-%m-%dT00:00:00")

    params = {
        "$select": """
            transit_timestamp,
            bus_route,
            payment_method,
            sum(ridership) AS ridership,
            sum(transfers) AS transfers
        """,
        "$where": f"""
            transit_timestamp >= '{start_str}'
            AND transit_timestamp < '{end_str}'
        """,
        "$group": """
            transit_timestamp,
            bus_route,
            payment_method
        """,
        "$order": "transit_timestamp, bus_route, payment_method",
        "$limit": LIMIT,
    }

    headers = {}
    if APP_TOKEN:
        headers["X-App-Token"] = APP_TOKEN

    print(f"Pulling {month_start.strftime('%Y-%m')}...")

    response = requests.get(
        BASE_URL,
        params=params,
        headers=headers,
        timeout=300,
    )
    response.raise_for_status()

    records = response.json()
    print(f"  Retrieved {len(records):,} grouped rows")

    if not records:
        return pd.DataFrame()

    if len(records) >= LIMIT:
        raise RuntimeError(
            f"{month_start.strftime('%Y_%m')} hit the Socrata row limit "
            f"({LIMIT:,}). Split this month into smaller pulls before saving."
        )

    df = pd.DataFrame.from_records(records)

    df["transit_timestamp"] = pd.to_datetime(df["transit_timestamp"])
    df["ridership"] = (
        pd.to_numeric(df["ridership"], errors="coerce")
        .fillna(0)
        .astype("int64")
    )
    df["transfers"] = (
        pd.to_numeric(df["transfers"], errors="coerce")
        .fillna(0)
        .astype("int64")
    )

    df["date"] = df["transit_timestamp"].dt.date
    df["hour"] = df["transit_timestamp"].dt.hour
    df["day_of_week"] = df["transit_timestamp"].dt.day_name()
    df["is_weekend"] = df["transit_timestamp"].dt.dayofweek >= 5

    return df


# -----------------------------
# Main pull
# -----------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    months = month_starts(
        START_MONTH,
        END_MONTH,
        include_current_month=INCLUDE_CURRENT_MONTH,
    )

    if not months:
        print("No months to pull.")
        return

    print(
        f"Pulling {len(months)} month(s) from "
        f"{months[0].strftime('%Y-%m')} to {months[-1].strftime('%Y-%m')}"
    )
    print(f"Saving monthly Parquet files to: {OUTPUT_DIR}")

    total_rows = 0
    total_ridership = 0
    total_transfers = 0

    for month_start in months:
        ym = month_start.strftime("%Y_%m")
        output_path = OUTPUT_DIR / f"bus_hourly_route_{ym}.parquet"
        metadata_path = output_path.with_suffix(".metadata.json")

        if output_path.exists() and not OVERWRITE_EXISTING:
            print(f"Skipping {ym}; file already exists: {output_path}")
            continue

        df = pull_one_month(month_start)

        if df.empty:
            print(f"  No data for {ym}; skipping save")
            continue

        validate_month(df, month_start, ym)

        df.to_parquet(output_path, index=False)

        rows = len(df)
        ridership = int(df["ridership"].sum())
        transfers = int(df["transfers"].sum())

        metadata = {
            "dataset_id": DATASET_ID,
            "base_url": BASE_URL,
            "source_month": ym,
            "pulled_at": datetime.now().isoformat(timespec="seconds"),
            "output_file": str(output_path),
            "row_count": rows,
            "min_timestamp": str(df["transit_timestamp"].min()),
            "max_timestamp": str(df["transit_timestamp"].max()),
            "columns": list(df.columns),
            "ridership_total": ridership,
            "transfers_total": transfers,
            "include_current_month": INCLUDE_CURRENT_MONTH,
            "overwrite_existing": OVERWRITE_EXISTING,
        }

        write_metadata(metadata_path, metadata)

        total_rows += rows
        total_ridership += ridership
        total_transfers += transfers

        print(f"  Saved: {output_path}")
        print(f"  Metadata: {metadata_path}")
        print(
            f"  Date range: "
            f"{df['transit_timestamp'].min()} to "
            f"{df['transit_timestamp'].max()}"
        )
        print(f"  Rows: {rows:,}")
        print(f"  Ridership: {ridership:,}")
        print(f"  Transfers: {transfers:,}")

        time.sleep(SLEEP_SECONDS)

    print("\nDone.")
    print(f"New rows pulled this run: {total_rows:,}")
    print(f"New ridership pulled this run: {total_ridership:,}")
    print(f"New transfers pulled this run: {total_transfers:,}")


if __name__ == "__main__":
    main()