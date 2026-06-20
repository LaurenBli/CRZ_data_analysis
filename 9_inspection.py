from pathlib import Path
import pandas as pd

PROCESSED_DIR = Path("data/processed")

files = [
    "analysis_panel_hourly_2024_01_2026_03.parquet",
    "analysis_panel_with_weather.parquet",
    "analysis_panel_with_event_flags.parquet",
]

for file in files:
    path = PROCESSED_DIR / file
    print("\n" + "=" * 90)
    print(file)
    print("=" * 90)

    df = pd.read_parquet(path)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print(f"Date range: {df['transit_timestamp'].min()} to {df['transit_timestamp'].max()}")

    print("\nColumn properties:")
    props = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "non_null": df.notna().sum().values,
        "missing": df.isna().sum().values,
        "unique": [df[c].nunique(dropna=True) for c in df.columns],
    })

    print(props.to_string(index=False))