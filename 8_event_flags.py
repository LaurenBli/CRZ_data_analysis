from pathlib import Path
import pandas as pd

# -----------------------------
# Settings
# -----------------------------

PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/processed")

INPUT_FILE = PROCESSED_DIR / "analysis_panel_with_weather.parquet"
OUTPUT_FILE = OUTPUT_DIR / "analysis_panel_with_event_flags.parquet"


HOLIDAYS = [
    # 2024
    "2024-01-01",  # New Year's Day
    "2024-05-27",  # Memorial Day
    "2024-07-04",  # Independence Day
    "2024-09-02",  # Labor Day
    "2024-11-28",  # Thanksgiving
    "2024-12-25",  # Christmas

    # 2025
    "2025-01-01",
    "2025-05-26",
    "2025-07-04",
    "2025-09-01",
    "2025-11-27",
    "2025-12-25",

    # 2026
    "2026-01-01",
]

# -----------------------------
# Main
# -----------------------------

def main():
    print(f"Loading: {INPUT_FILE}")

    df = pd.read_parquet(INPUT_FILE)

    df["transit_timestamp"] = pd.to_datetime(
        df["transit_timestamp"]
    )

    df["date"] = pd.to_datetime(df["date"])

    duplicate_ts = (
        df["transit_timestamp"]
        .duplicated()
        .sum()
    )

    if duplicate_ts:
        raise ValueError(
            f"Found {duplicate_ts:,} duplicate timestamps"
        )

    # -----------------------------
    # Holiday flag
    # -----------------------------

    holiday_dates = pd.to_datetime(HOLIDAYS)

    df["holiday_flag"] = (
        df["date"].isin(holiday_dates)
    ).astype(int)

    # -----------------------------
    # Weather fallback
    # -----------------------------

    if "severe_weather_flag" not in df.columns:
        print(
            "WARNING: severe_weather_flag missing. "
            "Creating zero-filled column."
        )

        df["severe_weather_flag"] = 0

    df["severe_weather_flag"] = (
        pd.to_numeric(
            df["severe_weather_flag"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    # -----------------------------
    # Combined disruption flag
    # -----------------------------

    df["major_event_flag"] = (
        (
            df["holiday_flag"]
            == 1
        )
        |
        (
            df["severe_weather_flag"]
            == 1
        )
    ).astype(int)

    # -----------------------------
    # Save
    # -----------------------------

    df = (
        df.sort_values("transit_timestamp")
        .reset_index(drop=True)
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print("\nDone.")
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Rows: {len(df):,}")

    print("\nFlag counts:")

    print(
        f"holiday_flag: "
        f"{int(df['holiday_flag'].sum()):,}"
    )

    print(
        f"severe_weather_flag: "
        f"{int(df['severe_weather_flag'].sum()):,}"
    )

    print(
        f"major_event_flag: "
        f"{int(df['major_event_flag'].sum()):,}"
    )

    print("\nHoliday dates used:")

    for d in HOLIDAYS:
        print(f"- {d}")


if __name__ == "__main__":
    main()
