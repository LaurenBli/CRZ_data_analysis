from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------
# Settings
# -----------------------------

PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("outputs/outlier_checks")

PANEL_FILE = PROCESSED_DIR / "analysis_panel_hourly_2024_01_2026_03.parquet"
DAILY_FILE = PROCESSED_DIR / "daily_summary_2024_01_2026_03.parquet"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTCOME_COLS = [
    "subway_ridership",
    "bus_ridership",

    # TLC
    "taxi_trips",
    "forhire_trips",
    "uber_trips",
    "lyft_trips",

    # Bikes
    "citibike_rides",

    # Bridges
    "bridge_traffic_total",
    "bridge_traffic_treated",
    "bridge_traffic_control",
    "bridge_traffic_spillover",

    # CRZ
    "crz_entries",
    "crz_excluded_roadway_entries",
]

EXCLUDE_AGG_COLS = {
    "year",
    "month",
    "hour",
    "is_weekend",
    "post_congestion_pricing",
}

LOWEST_N = 20
HIGHEST_N = 20
Z_THRESHOLD = 3.0
CRZ_START = pd.Timestamp("2025-01-05")

# -----------------------------
# Helpers
# -----------------------------


def load_daily_panel() -> pd.DataFrame:
    if DAILY_FILE.exists():
        print(f"Loading daily summary: {DAILY_FILE}")
        df = pd.read_parquet(DAILY_FILE)
        df["date"] = pd.to_datetime(df["date"])
        return df

    print(f"Daily summary not found. Building from hourly panel: {PANEL_FILE}")

    if not PANEL_FILE.exists():
        raise FileNotFoundError(f"Missing hourly panel: {PANEL_FILE}")

    hourly = pd.read_parquet(PANEL_FILE)
    hourly["date"] = pd.to_datetime(hourly["date"])

    numeric_cols = [
        col
        for col in hourly.select_dtypes(include="number").columns
        if col not in EXCLUDE_AGG_COLS
    ]

    daily = hourly.groupby("date", as_index=False)[numeric_cols].sum()
    daily.to_parquet(DAILY_FILE, index=False)

    return daily


def add_calendar_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.day_name()
    df["is_weekend"] = df["date"].dt.dayofweek >= 5
    return df


def robust_z_score(series: pd.Series) -> pd.Series:
    """Median absolute deviation z-score. More robust than ordinary z-score."""
    median = series.median()
    mad = (series - median).abs().median()

    if mad == 0 or pd.isna(mad):
        return pd.Series(np.nan, index=series.index)

    return 0.6745 * (series - median) / mad


def check_outliers_for_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    temp = df[["date", "year", "month", "day_of_week", "is_weekend", col]].copy()
    temp = temp.dropna(subset=[col])

    # Avoid flagging pre-policy CRZ zeros as real CRZ outliers.
    if col.startswith("crz_"):
        temp = temp[temp["date"] >= CRZ_START].copy()

    if temp.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "year",
                "month",
                "day_of_week",
                "is_weekend",
                col,
                "robust_z",
                "weekday_robust_z",
                "pct_rank",
                "outlier_flag",
                "metric",
                "tail",
            ]
        )

    temp["robust_z"] = robust_z_score(temp[col])
    temp["weekday_robust_z"] = temp.groupby("day_of_week")[col].transform(robust_z_score)
    temp["pct_rank"] = temp[col].rank(pct=True)

    # Weekday-normalized z-score is better for transit data because weekends and
    # weekdays have structurally different volumes.
    temp["outlier_flag"] = temp["weekday_robust_z"].abs() >= Z_THRESHOLD
    temp["metric"] = col

    lowest = temp.nsmallest(LOWEST_N, col).copy()
    lowest["tail"] = "lowest"

    highest = temp.nlargest(HIGHEST_N, col).copy()
    highest["tail"] = "highest"

    flagged = temp[temp["outlier_flag"]].copy()
    flagged["tail"] = "weekday_z_flagged"

    combined = pd.concat([lowest, highest, flagged], ignore_index=True)
    combined = combined.drop_duplicates(subset=["metric", "date", "tail"])

    return combined


def print_column_summary(df: pd.DataFrame, col: str) -> None:
    temp = df[["date", col]].dropna().copy()

    if col.startswith("crz_"):
        temp = temp[temp["date"] >= CRZ_START]

    print("\n" + "=" * 90)
    print(f"OUTLIER CHECK: {col}")
    print("=" * 90)
    print(f"Observations: {len(temp):,}")

    if temp.empty:
        print("No observations available after filtering.")
        return

    print(f"Min: {temp[col].min():,.2f}")
    print(f"P01: {temp[col].quantile(0.01):,.2f}")
    print(f"P05: {temp[col].quantile(0.05):,.2f}")
    print(f"Median: {temp[col].median():,.2f}")
    print(f"Mean: {temp[col].mean():,.2f}")
    print(f"P95: {temp[col].quantile(0.95):,.2f}")
    print(f"P99: {temp[col].quantile(0.99):,.2f}")
    print(f"Max: {temp[col].max():,.2f}")

    print("\nLowest dates:")
    print(temp.nsmallest(LOWEST_N, col).to_string(index=False))

    print("\nHighest dates:")
    print(temp.nlargest(HIGHEST_N, col).to_string(index=False))


def validate_daily_panel(df: pd.DataFrame) -> None:
    duplicate_dates = int(df["date"].duplicated().sum())

    if duplicate_dates:
        raise ValueError(f"Found {duplicate_dates:,} duplicate daily rows")

    if df["date"].isna().any():
        bad_dates = int(df["date"].isna().sum())
        raise ValueError(f"Found {bad_dates:,} rows with invalid date values")


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    df = load_daily_panel()
    df = add_calendar_columns(df)
    validate_daily_panel(df)

    existing_cols = [col for col in OUTCOME_COLS if col in df.columns]
    missing_cols = [col for col in OUTCOME_COLS if col not in df.columns]

    print("\nExisting outcome columns:")
    for col in existing_cols:
        print(f"- {col}")

    if missing_cols:
        print("\nMissing outcome columns:")
        for col in missing_cols:
            print(f"- {col}")

    all_outliers = []

    for col in existing_cols:
        print_column_summary(df, col)
        out = check_outliers_for_column(df, col)

        if not out.empty:
            all_outliers.append(out)

    if all_outliers:
        outlier_df = pd.concat(all_outliers, ignore_index=True)
        outlier_df = outlier_df.sort_values(["metric", "tail", "date"]).reset_index(drop=True)
    else:
        outlier_df = pd.DataFrame()

    output_csv = OUTPUT_DIR / "daily_outlier_candidates.csv"
    outlier_df.to_csv(output_csv, index=False)

    print("\n" + "=" * 90)
    print("SAVED OUTLIER CANDIDATES")
    print("=" * 90)
    print(f"Saved CSV: {output_csv}")
    print(f"Rows: {len(outlier_df):,}")

    if not outlier_df.empty:
        print("\nOutlier candidate counts by metric and tail:")
        print(outlier_df.groupby(["metric", "tail"]).size().unstack(fill_value=0))
    else:
        print("\nNo outlier candidates generated.")

    print("\nNext step: review daily_outlier_candidates.csv and classify dates as:")
    print("- real_event")
    print("- holiday")
    print("- weather")
    print("- data_issue")
    print("- keep")


if __name__ == "__main__":
    main()
