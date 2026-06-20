from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Settings
# -----------------------------

PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("outputs/figures")

PANEL_FILE = (
    PROCESSED_DIR
    / "analysis_panel_hourly_2024_01_2026_03.parquet"
)

POLICY_START = pd.Timestamp("2025-01-05 00:00:00")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EXCLUDE_AGG_COLS = {
    "year",
    "month",
    "hour",
    "is_weekend",
    "post_congestion_pricing",
}

# -----------------------------
# Helpers
# -----------------------------

def save_line_plot(
    df,
    x,
    y,
    title,
    ylabel,
    filename,
):
    plt.figure(figsize=(12, 6))

    plt.plot(df[x], df[y])

    plt.axvline(
        POLICY_START,
        linestyle="--",
        linewidth=1,
    )

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)

    plt.tight_layout()

    out = OUTPUT_DIR / filename

    plt.savefig(
        out,
        dpi=150,
    )

    plt.close()

    print(f"Saved: {out}")


def print_pre_post(df, cols):
    print("\n" + "=" * 80)
    print("PRE / POST HOURLY AVERAGES")
    print("=" * 80)

    tmp = df.copy()

    tmp["period"] = tmp["post_congestion_pricing"].map({
        False: "pre",
        True: "post",
    })

    summary = (
        tmp.groupby("period")[cols]
        .mean()
        .T
    )

    summary["pct_change_post_vs_pre"] = (
        (
            summary["post"]
            - summary["pre"]
        )
        / summary["pre"].replace(0, pd.NA)
    ) * 100

    print(summary.round(2))


def print_missing_and_ranges(df):
    print("\n" + "=" * 80)
    print("BASIC PANEL CHECKS")
    print("=" * 80)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    print(
        f"Date range: "
        f"{df['transit_timestamp'].min()} to "
        f"{df['transit_timestamp'].max()}"
    )

    expected_hours = pd.date_range(
        df["transit_timestamp"].min(),
        df["transit_timestamp"].max(),
        freq="h",
    )

    missing_hours = expected_hours.difference(
        df["transit_timestamp"]
    )

    print(f"Expected hourly rows: {len(expected_hours):,}")
    print(f"Missing hourly timestamps: {len(missing_hours):,}")

    non_hourly_minutes = (
        df["transit_timestamp"].dt.minute != 0
    ).sum()

    non_hourly_seconds = (
        df["transit_timestamp"].dt.second != 0
    ).sum()

    print(f"Non-hourly minutes: {non_hourly_minutes:,}")
    print(f"Non-hourly seconds: {non_hourly_seconds:,}")

    print("\nTop missing-value columns:")

    print(
        df.isna()
        .sum()
        .sort_values(ascending=False)
        .head(15)
    )


def create_daily_panel(df):
    numeric_cols = [
        c
        for c in df.select_dtypes(include="number").columns
        if c not in EXCLUDE_AGG_COLS
    ]

    daily = (
        df.groupby("date", as_index=False)[numeric_cols]
        .sum()
    )

    daily["date"] = pd.to_datetime(daily["date"])

    return daily


def create_weekly_panel(df):
    tmp = df.copy()

    tmp["week"] = (
        tmp["transit_timestamp"]
        .dt.to_period("W")
        .dt.start_time
    )

    numeric_cols = [
        c
        for c in tmp.select_dtypes(include="number").columns
        if c not in EXCLUDE_AGG_COLS
    ]

    weekly = (
        tmp.groupby("week", as_index=False)[numeric_cols]
        .sum()
    )

    return weekly


# -----------------------------
# Main
# -----------------------------

def main():
    print(f"Loading panel: {PANEL_FILE}")

    df = pd.read_parquet(PANEL_FILE)

    df["transit_timestamp"] = pd.to_datetime(
        df["transit_timestamp"]
    )

    duplicate_ts = (
        df["transit_timestamp"]
        .duplicated()
        .sum()
    )

    if duplicate_ts:
        raise ValueError(
            f"Found {duplicate_ts:,} duplicate timestamps"
        )

    print_missing_and_ranges(df)

    core_cols = [
        "subway_ridership",
        "bus_ridership",
        "taxi_trips",
        "forhire_trips",
        "uber_trips",
        "lyft_trips",
        "citibike_rides",
        "bridge_traffic_total",
        "bridge_traffic_treated",
        "bridge_traffic_control",
        "bridge_traffic_spillover",
        "crz_entries",
        "crz_excluded_roadway_entries",
    ]

    existing_core_cols = [
        c for c in core_cols
        if c in df.columns
    ]

    print_pre_post(
        df,
        existing_core_cols,
    )

    print("\n" + "=" * 80)
    print("HOURLY AVERAGE BY PEAK PERIOD")
    print("=" * 80)

    print(
        df.groupby("peak_period")[existing_core_cols]
        .mean()
        .round(2)
    )

    print("\n" + "=" * 80)
    print("HOURLY AVERAGE BY WEEKEND STATUS")
    print("=" * 80)

    print(
        df.groupby("is_weekend")[existing_core_cols]
        .mean()
        .round(2)
    )

    daily = create_daily_panel(df)
    weekly = create_weekly_panel(df)

    # Optional smoothing for cleaner thesis plots
    smooth_cols = [
        "subway_ridership",
        "bus_ridership",
        "taxi_trips",
        "forhire_trips",
        "uber_trips",
        "lyft_trips",
        "citibike_rides",
        "bridge_traffic_total",
        "crz_entries",
    ]

    for col in smooth_cols:
        if col in daily.columns:
            daily[f"{col}_7dma"] = (
                daily[col]
                .rolling(7, center=True)
                .mean()
            )

    # Save compact summary tables
    daily_summary_path = (
        PROCESSED_DIR
        / "daily_summary_2024_01_2026_03.parquet"
    )

    weekly_summary_path = (
        PROCESSED_DIR
        / "weekly_summary_2024_01_2026_03.parquet"
    )

    daily.to_parquet(
        daily_summary_path,
        index=False,
    )

    weekly.to_parquet(
        weekly_summary_path,
        index=False,
    )

    print(f"\nSaved: {daily_summary_path}")
    print(f"Saved: {weekly_summary_path}")

    # -----------------------------
    # Daily trend plots
    # -----------------------------

    plot_specs = [
        (
            "subway_ridership_7dma",
            "Daily Subway Ridership (7DMA)",
            "Ridership",
            "daily_subway_ridership.png",
        ),
        (
            "bus_ridership_7dma",
            "Daily Bus Ridership (7DMA)",
            "Ridership",
            "daily_bus_ridership.png",
        ),
        (
            "taxi_trips_7dma",
            "Daily Yellow Taxi Trips (7DMA)",
            "Trips",
            "daily_taxi_trips.png",
        ),
        (
            "citibike_rides_7dma",
            "Daily Citi Bike Rides (7DMA)",
            "Rides",
            "daily_citibike_rides.png",
        ),
        (
            "bridge_traffic_total_7dma",
            "Daily MTA Bridge/Tunnel Traffic (7DMA)",
            "Traffic count",
            "daily_bridge_traffic_total.png",
        ),
        (
            "crz_entries_7dma",
            "Daily CRZ Entries (7DMA)",
            "Entries",
            "daily_crz_entries.png",
        ),
        (
            "crz_excluded_roadway_entries",
            "Daily Excluded Roadway Entries",
            "Entries",
            "daily_crz_excluded_roadway_entries.png",
        ),
        (
            "forhire_trips_7dma",
            "Daily For-Hire Vehicle Trips (7DMA)",
            "Trips",
            "daily_forhire_trips.png",
        ),
        (
            "uber_trips_7dma",
            "Daily Uber Trips (7DMA)",
            "Trips",
            "daily_uber_trips.png",
        ),
        (
            "lyft_trips_7dma",
            "Daily Lyft Trips (7DMA)",
            "Trips",
            "daily_lyft_trips.png",
        ),
    ]

    for (
        col,
        title,
        ylabel,
        filename,
    ) in plot_specs:

        if col in daily.columns:
            save_line_plot(
                daily,
                "date",
                col,
                title,
                ylabel,
                filename,
            )

    # -----------------------------
    # Weekly trend plots
    # -----------------------------

    weekly_specs = [
        (
            "subway_ridership",
            "Weekly Subway Ridership",
            "Ridership",
            "weekly_subway_ridership.png",
        ),
        (
            "bus_ridership",
            "Weekly Bus Ridership",
            "Ridership",
            "weekly_bus_ridership.png",
        ),
        (
            "taxi_trips",
            "Weekly Yellow Taxi Trips",
            "Trips",
            "weekly_taxi_trips.png",
        ),
        (
            "citibike_rides",
            "Weekly Citi Bike Rides",
            "Rides",
            "weekly_citibike_rides.png",
        ),
        (
            "bridge_traffic_total",
            "Weekly MTA Bridge/Tunnel Traffic",
            "Traffic count",
            "weekly_bridge_traffic_total.png",
        ),
        (
            "crz_entries",
            "Weekly CRZ Entries",
            "Entries",
            "weekly_crz_entries.png",
        ),
        (
            "forhire_trips",
            "Weekly For-Hire Vehicle Trips",
            "Trips",
            "weekly_forhire_trips.png",
        ),
        (
            "uber_trips",
            "Weekly Uber Trips",
            "Trips",
            "weekly_uber_trips.png",
        ),
        (
            "lyft_trips",
            "Weekly Lyft Trips",
            "Trips",
            "weekly_lyft_trips.png",
        ),
    ]

    for (
        col,
        title,
        ylabel,
        filename,
    ) in weekly_specs:

        if col in weekly.columns:
            save_line_plot(
                weekly,
                "week",
                col,
                title,
                ylabel,
                filename,
            )

    print(
        "\nDone. "
        "Review printed summaries and plots in outputs/figures/."
    )


if __name__ == "__main__":
    main()