"""
11_make_model_ready_panel.py

Preserve the final hourly panel containing for-hire trips and attach the
holiday/weather/event controls from the earlier event-flags panel by timestamp.
"""

from pathlib import Path
import pandas as pd

BASE_PANEL = Path("data/processed/analysis_panel_hourly_2024_01_2026_03.parquet")
FLAGS_PANEL = Path("data/processed/analysis_panel_with_event_flags.parquet")
OUT_PANEL = Path("data/processed/analysis_panel_model_ready.parquet")

KEY = "transit_timestamp"
FLAG_COLS = ["holiday_flag", "severe_weather_flag", "major_event_flag"]


def main():
    if not BASE_PANEL.exists():
        raise FileNotFoundError(f"Missing base panel: {BASE_PANEL}")
    if not FLAGS_PANEL.exists():
        raise FileNotFoundError(f"Missing flags panel: {FLAGS_PANEL}")

    base = pd.read_parquet(BASE_PANEL)
    flags = pd.read_parquet(FLAGS_PANEL)

    missing_base = {"transit_timestamp", "forhire_trips"} - set(base.columns)
    missing_flags = {"transit_timestamp", *FLAG_COLS} - set(flags.columns)
    if missing_base:
        raise ValueError(f"Base panel missing: {sorted(missing_base)}")
    if missing_flags:
        raise ValueError(f"Flags panel missing: {sorted(missing_flags)}")

    base = base.copy()
    flags = flags[["transit_timestamp"] + FLAG_COLS].copy()

    base[KEY] = pd.to_datetime(base[KEY])
    flags[KEY] = pd.to_datetime(flags[KEY])

    if base[KEY].duplicated().any():
        raise ValueError("Base panel has duplicate timestamps.")
    if flags[KEY].duplicated().any():
        raise ValueError("Flags panel has duplicate timestamps.")

    missing_in_flags = base.loc[~base[KEY].isin(flags[KEY]), KEY]
    extra_in_flags = flags.loc[~flags[KEY].isin(base[KEY]), KEY]

    if not missing_in_flags.empty or not extra_in_flags.empty:
        details = []
        if not missing_in_flags.empty:
            details.append(
                f"{len(missing_in_flags):,} base timestamps lack flags "
                f"(first: {missing_in_flags.iloc[0]})"
            )
        if not extra_in_flags.empty:
            details.append(
                f"{len(extra_in_flags):,} flag timestamps are not in base "
                f"(first: {extra_in_flags.iloc[0]})"
            )
        raise ValueError("Timestamp mismatch: " + "; ".join(details))

    # Flags must come from the event-flags file, never stale base-panel columns.
    base = base.drop(columns=FLAG_COLS, errors="ignore")

    final = base.merge(
        flags,
        on=KEY,
        how="left",
        validate="one_to_one",
    )

    if len(final) != len(base):
        raise RuntimeError("Merge changed row count unexpectedly.")

    for col in FLAG_COLS:
        final[col] = pd.to_numeric(final[col], errors="coerce")
        if final[col].isna().any():
            raise ValueError(f"{col} has missing values after merge.")
        if not final[col].isin([0, 1]).all():
            invalid = sorted(final.loc[~final[col].isin([0, 1]), col].unique())
            raise ValueError(f"{col} has non-binary values: {invalid[:10]}")
        final[col] = final[col].astype(int)

    if final["forhire_trips"].isna().any():
        raise ValueError("forhire_trips has missing values in final panel.")

    final = final.sort_values(KEY).reset_index(drop=True)
    OUT_PANEL.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(OUT_PANEL, index=False)

    print("=" * 90)
    print("Model-ready panel created")
    print("=" * 90)
    print(f"Saved: {OUT_PANEL}")
    print(f"Rows: {len(final):,}")
    print(f"Range: {final[KEY].min()} to {final[KEY].max()}")
    print("\nEvent-control totals:")
    print(final[FLAG_COLS].sum().to_string())


if __name__ == "__main__":
    main()
