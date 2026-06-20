from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw/weather")
PROCESSED_DIR = Path("data/processed")

PANEL_FILE = PROCESSED_DIR / "analysis_panel_hourly_2024_01_2026_03.parquet"
OUTPUT_FILE = PROCESSED_DIR / "analysis_panel_with_weather.parquet"
WEATHER_OUTPUT_FILE = PROCESSED_DIR / "weather_daily_controls.parquet"

WEATHER_FILE_PATTERN = "*.csv"


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().upper() for c in df.columns]
    return df


def load_weather_file(path: Path) -> pd.DataFrame:
    print(f"Loading: {path}")

    df = pd.read_csv(path)
    df = standardize_columns(df)

    if "DATE" not in df.columns:
        raise ValueError(f"DATE column missing in {path.name}")

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce").dt.date
    df = df.dropna(subset=["DATE"])

    keep_map = {
        "PRCP": "precipitation",
        "SNOW": "snowfall",
        "SNWD": "snow_depth",
        "TMAX": "tmax",
        "TMIN": "tmin",
        "AWND": "avg_wind",
    }

    cols = ["DATE"] + [c for c in keep_map if c in df.columns]
    df = df[cols].copy()

    df = df.rename(
        columns={k: v for k, v in keep_map.items() if k in df.columns}
    )

    for col in df.columns:
        if col != "DATE":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "tmax" in df.columns and "tmin" in df.columns:
        df["avg_temp"] = (df["tmax"] + df["tmin"]) / 2

    df["source_file"] = path.name

    return df


def build_weather_controls() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob(WEATHER_FILE_PATTERN))

    if not files:
        raise FileNotFoundError(f"No weather CSV files found in {RAW_DIR}")

    dfs = [load_weather_file(path) for path in files]
    combined = pd.concat(dfs, ignore_index=True)

    numeric_cols = [
        c for c in combined.columns
        if c not in {"DATE", "source_file"}
    ]

    weather_daily = (
        combined
        .groupby("DATE", as_index=False)[numeric_cols]
        .mean()
    )

    weather_daily = weather_daily.rename(columns={"DATE": "date"})
    weather_daily["date"] = pd.to_datetime(weather_daily["date"])

    print("\nWeather date range:")
    print(f"{weather_daily['date'].min()} to {weather_daily['date'].max()}")

    print("\nWeather files used:")
    for file in files:
        print(f"- {file.name}")

    print("\nWeather summary before thresholds:")
    summary_cols = [
        c for c in [
            "precipitation",
            "snowfall",
            "tmax",
            "tmin",
            "avg_temp",
            "avg_wind",
        ]
        if c in weather_daily.columns
    ]
    print(weather_daily[summary_cols].describe())

    precip_threshold = weather_daily["precipitation"].quantile(0.95)
    snow_threshold = weather_daily["snowfall"].quantile(0.95)
    cold_threshold = weather_daily["avg_temp"].quantile(0.05)

    print(f"Precipitation P95 threshold: {precip_threshold}")
    print(f"Snowfall P95 threshold: {snow_threshold}")
    print(f"Temperature P05 threshold: {cold_threshold}")

    weather_daily["severe_weather_flag"] = (
        (
            (weather_daily["precipitation"] > precip_threshold)
            | (weather_daily["snowfall"] > snow_threshold)
            | (weather_daily["avg_temp"] < cold_threshold)
        )
    ).astype(int)

    return weather_daily


def merge_with_panel(weather_daily: pd.DataFrame):
    print(f"\nLoading panel: {PANEL_FILE}")

    panel = pd.read_parquet(PANEL_FILE)
    panel["date"] = pd.to_datetime(panel["date"])

    merged = panel.merge(
        weather_daily,
        on="date",
        how="left",
    )

    missing_weather_days = (
        merged.loc[merged["precipitation"].isna(), "date"]
        .drop_duplicates()
        .sort_values()
    )

    if len(missing_weather_days):
        print(f"\nWARNING: Missing weather for {len(missing_weather_days):,} dates")
        print(missing_weather_days.head(20).to_string(index=False))

    merged.to_parquet(OUTPUT_FILE, index=False)
    weather_daily.to_parquet(WEATHER_OUTPUT_FILE, index=False)

    print("\nDone.")
    print(f"Saved merged panel: {OUTPUT_FILE}")
    print(f"Saved weather controls: {WEATHER_OUTPUT_FILE}")
    print(f"Rows: {len(merged):,}")

    print("\nWeather severe flag count:")
    print(merged["severe_weather_flag"].value_counts(dropna=False))

    print("\nColumns added:")
    for col in weather_daily.columns:
        if col != "date":
            print(f"- {col}")


def main():
    weather_daily = build_weather_controls()
    merge_with_panel(weather_daily)


if __name__ == "__main__":
    main()