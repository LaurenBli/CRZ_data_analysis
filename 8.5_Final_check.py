"""Final, non-destructive quality-assurance check for the thesis dataset.

Run this after building the hourly panel (code file 8), and preferably after
weather/event flags if those are part of the model:

    python 9_final_quality_check.py

The script does not alter any parquet or CSV file. It writes a CSV and JSON
report to outputs/final_quality_check/ and prints PASS / WARNING / FAIL items.

Exit codes:
    0 = no FAIL findings
    1 = one or more FAIL findings

Use --no-source-checks for a quick panel-only check.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from pandas.api.types import is_numeric_dtype


# ---------------------------------------------------------------------
# Project settings
# ---------------------------------------------------------------------
PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("outputs/final_quality_check")
POLICY_START = pd.Timestamp("2025-01-05 00:00:00")

PANEL_CANDIDATES = [
    PROCESSED_DIR / "analysis_panel_with_event_flags.parquet",
    PROCESSED_DIR / "analysis_panel_with_weather.parquet",
    PROCESSED_DIR / "analysis_panel_hourly_2024_01_2026_03.parquet",
]

CORE_PANEL_METRICS = [
    "subway_ridership",
    "bus_ridership",
    "taxi_trips",
    "forhire_trips",
    "citibike_rides",
    "bridge_traffic_total",
    "crz_entries",
    "crz_excluded_roadway_entries",
]

# These checks assess saved master files. They do not reconstruct the API pulls.
MASTER_SPECS: dict[str, dict[str, Any]] = {
    "bridges": {
        "path": PROCESSED_DIR / "bridges_master_2024_01_2026_03.parquet",
        "metric_columns": ["traffic_count"],
        "expected_start": "2024-01-01 00:00:00",
        "expected_end": "2026-03-31 23:00:00",
    },
    "subway": {
        "path": PROCESSED_DIR / "subway_master_2024_01_2026_03.parquet",
        "metric_columns": ["ridership", "transfers"],
        "expected_start": "2024-01-01 00:00:00",
        "expected_end": "2026-03-31 23:00:00",
    },
    "bus": {
        "path": PROCESSED_DIR / "bus_master_2024_01_2026_03.parquet",
        "metric_columns": ["ridership", "transfers"],
        "expected_start": "2024-01-01 00:00:00",
        "expected_end": "2026-03-31 23:00:00",
    },
    "taxi": {
        "path": PROCESSED_DIR / "taxi_master_2024_01_2026_03.parquet",
        "metric_columns": ["trip_count"],
        "expected_start": "2024-01-01 00:00:00",
        "expected_end": "2026-03-31 23:00:00",
    },
    "citibike": {
        "path": PROCESSED_DIR / "citibike_master_2024_01_2026_03.parquet",
        "metric_columns": ["ride_count"],
        "expected_start": "2024-01-01 00:00:00",
        "expected_end": "2026-03-31 23:00:00",
    },
    "forhire": {
        "path": PROCESSED_DIR / "forhire_master_2024_01_2026_03.parquet",
        "metric_columns": ["forhire_trip_count"],
        "expected_start": "2024-01-01 00:00:00",
        "expected_end": "2026-03-31 23:00:00",
    },
    "crz": {
        "path": PROCESSED_DIR / "crz_master_2025_01_2026_03.parquet",
        "metric_columns": ["crz_entries", "excluded_roadway_entries"],
        # The CRZ policy/data series begins on Jan. 5, 2025; Jan. 1–4 are not zeros.
        "expected_start": "2025-01-05 00:00:00",
        "expected_end": "2026-03-31 23:00:00",
    },
}

SEVERITY_ORDER = {"PASS": 0, "WARNING": 1, "FAIL": 2}


# ---------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------
def new_report() -> list[dict[str, Any]]:
    return []


def add(
    report: list[dict[str, Any]],
    status: str,
    section: str,
    check: str,
    detail: str,
    value: Any | None = None,
) -> None:
    report.append(
        {
            "status": status,
            "section": section,
            "check": check,
            "detail": detail,
            "value": value,
        }
    )


def quote_path(path: Path) -> str:
    """Return a DuckDB-safe absolute-path literal."""
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def choose_panel(user_path: str | None) -> Path:
    if user_path:
        path = Path(user_path)
        if not path.exists():
            raise FileNotFoundError(f"Requested panel does not exist: {path}")
        return path

    for path in PANEL_CANDIDATES:
        if path.exists():
            return path

    candidates = "\n  - ".join(str(p) for p in PANEL_CANDIDATES)
    raise FileNotFoundError(
        "No analysis panel found. Expected one of:\n  - " + candidates
    )


# ---------------------------------------------------------------------
# Final-panel checks
# ---------------------------------------------------------------------
def check_panel(panel_path: Path, report: list[dict[str, Any]]) -> None:
    print(f"\nChecking final panel: {panel_path}")
    panel = pd.read_parquet(panel_path)

    if panel.empty:
        add(report, "FAIL", "panel", "nonempty", "Panel contains zero rows.", 0)
        return
    add(report, "PASS", "panel", "nonempty", "Panel contains rows.", len(panel))

    if "transit_timestamp" not in panel.columns:
        add(report, "FAIL", "panel", "timestamp column", "transit_timestamp is missing.")
        return

    panel["transit_timestamp"] = pd.to_datetime(
        panel["transit_timestamp"], errors="coerce"
    )
    null_ts = int(panel["transit_timestamp"].isna().sum())
    if null_ts:
        add(report, "FAIL", "panel", "null timestamps", "Invalid/null timestamps found.", null_ts)
        return
    add(report, "PASS", "panel", "null timestamps", "No null timestamps.", 0)

    duplicate_ts = int(panel["transit_timestamp"].duplicated().sum())
    if duplicate_ts:
        add(report, "FAIL", "panel", "duplicate timestamps", "One-row-per-hour panel is violated.", duplicate_ts)
    else:
        add(report, "PASS", "panel", "duplicate timestamps", "Exactly one row per timestamp.", 0)

    ts = panel["transit_timestamp"]
    non_hourly = int(((ts.dt.minute != 0) | (ts.dt.second != 0) | (ts.dt.microsecond != 0)).sum())
    if non_hourly:
        add(report, "FAIL", "panel", "hourly timestamps", "Timestamps are not all hour-aligned.", non_hourly)
    else:
        add(report, "PASS", "panel", "hourly timestamps", "All timestamps are hour-aligned.", 0)

    observed_start, observed_end = ts.min(), ts.max()
    expected = pd.date_range(observed_start, observed_end, freq="h")
    missing_hours = expected.difference(pd.DatetimeIndex(ts))
    if len(missing_hours):
        preview = ", ".join(str(x) for x in missing_hours[:5])
        add(
            report,
            "FAIL",
            "panel",
            "continuous hourly coverage",
            f"Missing {len(missing_hours)} hourly timestamp(s). First: {preview}",
            len(missing_hours),
        )
    else:
        add(report, "PASS", "panel", "continuous hourly coverage", "No missing hourly timestamps.", 0)

    # Calendar fields should agree with the timestamp whenever they are present.
    if "date" in panel.columns:
        stored_date = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
        expected_date = ts.dt.normalize()
        inconsistent_date = int((stored_date != expected_date).sum())
        if stored_date.isna().any() or inconsistent_date:
            add(report, "WARNING", "panel", "date consistency", "date does not always match transit_timestamp.", inconsistent_date)
        else:
            add(report, "PASS", "panel", "date consistency", "date matches transit_timestamp.", 0)

    if "hour" in panel.columns:
        coerced_hour = pd.to_numeric(panel["hour"], errors="coerce")
        inconsistent_hour = int((coerced_hour != ts.dt.hour).sum())
        if coerced_hour.isna().any() or inconsistent_hour:
            add(report, "WARNING", "panel", "hour consistency", "hour does not always match transit_timestamp.", inconsistent_hour)
        else:
            add(report, "PASS", "panel", "hour consistency", "hour matches transit_timestamp.", 0)

    numeric_cols = [c for c in panel.columns if is_numeric_dtype(panel[c])]
    panel_missing = panel[numeric_cols].isna().sum().sort_values(ascending=False)
    missing_numeric = panel_missing[panel_missing > 0]
    if missing_numeric.empty:
        add(report, "PASS", "panel", "numeric missingness", "No numeric missing values in the panel.", 0)
    else:
        # Outcome gaps are failures; control/data-availability gaps are warnings.
        missing_outcomes = [c for c in CORE_PANEL_METRICS if c in missing_numeric.index]
        if missing_outcomes:
            details = "; ".join(f"{c}={int(missing_numeric[c])}" for c in missing_outcomes)
            add(report, "FAIL", "panel", "core outcome missingness", details)
        other = missing_numeric.drop(labels=missing_outcomes, errors="ignore")
        if not other.empty:
            details = "; ".join(f"{c}={int(n)}" for c, n in other.head(10).items())
            add(report, "WARNING", "panel", "other numeric missingness", details)

    present_core = [c for c in CORE_PANEL_METRICS if c in panel.columns]
    absent_core = [c for c in CORE_PANEL_METRICS if c not in panel.columns]
    if absent_core:
        add(report, "WARNING", "panel", "core columns", f"Not present: {', '.join(absent_core)}")
    else:
        add(report, "PASS", "panel", "core columns", "All planned core outcomes are present.")

    # CRZ needs special treatment: its pre-policy period is structurally not observed.
    crz_cols = [c for c in ["crz_entries", "crz_excluded_roadway_entries"] if c in panel.columns]
    if crz_cols:
        post = panel.loc[ts >= POLICY_START, crz_cols]
        post_missing = int(post.isna().sum().sum())
        if post_missing:
            add(report, "FAIL", "CRZ", "post-policy missing values", "CRZ has missing values after program start.", post_missing)
        else:
            add(report, "PASS", "CRZ", "post-policy missing values", "No CRZ missing values after program start.", 0)

        # A whole hour at zero for both measures is unusual and worth inspecting,
        # but is not automatically an error (e.g., a true overnight zero).
        both_zero = (post[crz_cols].fillna(0).sum(axis=1) == 0)
        zero_hours = int(both_zero.sum())
        if zero_hours:
            add(report, "WARNING", "CRZ", "post-policy zero hours", "Inspect whether these are true zeros or source gaps.", zero_hours)
        else:
            add(report, "PASS", "CRZ", "post-policy zero hours", "No all-zero CRZ hours after program start.", 0)

    crz_map = Path("data/mappings/crz_treatment_map.csv")
    treatment_like = [c for c in panel.columns if c.startswith("crz_") and ("treated" in c or "spillover" in c or "control" in c)]
    if crz_map.exists() and not treatment_like:
        add(
            report,
            "WARNING",
            "CRZ",
            "treatment-map integration",
            "A CRZ treatment map exists, but no treated/control/spillover CRZ outcome is in this panel. Do not claim zone-treatment results.",
        )

    if present_core:
        activity = panel[present_core].copy()
        activity = activity[[c for c in activity.columns if not c.startswith("crz_")]]
        if not activity.empty:
            all_zero = int((activity.fillna(0).sum(axis=1) == 0).sum())
            if all_zero:
                add(report, "WARNING", "panel", "all-system zero hours", "All non-CRZ core outcomes are zero in these hours; inspect source coverage.", all_zero)
            else:
                add(report, "PASS", "panel", "all-system zero hours", "No hours have zero activity across all non-CRZ core outcomes.", 0)

    print(f"Panel range: {observed_start} to {observed_end}; rows: {len(panel):,}")


# ---------------------------------------------------------------------
# Saved-master checks (DuckDB: avoids loading the huge for-hire master)
# ---------------------------------------------------------------------
def check_master_sources(report: list[dict[str, Any]]) -> None:
    print("\nChecking saved master files with DuckDB...")
    con = duckdb.connect()

    for name, spec in MASTER_SPECS.items():
        path = Path(spec["path"])
        section = f"master:{name}"

        if not path.exists():
            add(report, "WARNING", section, "file exists", f"Not found: {path}")
            continue

        p = quote_path(path)
        cols = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{p}')").fetchdf()
        existing = set(cols["column_name"])
        required = {"transit_timestamp", *spec["metric_columns"]}
        absent = sorted(required - existing)
        if absent:
            add(report, "FAIL", section, "required columns", f"Missing: {', '.join(absent)}")
            continue

        metric_null_expr = ",\n                ".join(
            f"SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) AS {col}_nulls"
            for col in spec["metric_columns"]
        )
        metric_negative_expr = ",\n                ".join(
            f"SUM(CASE WHEN {col} < 0 THEN 1 ELSE 0 END) AS {col}_negative"
            for col in spec["metric_columns"]
        )

        stats = con.execute(
            f"""
            SELECT
                COUNT(*) AS row_count,
                MIN(transit_timestamp) AS min_ts,
                MAX(transit_timestamp) AS max_ts,
                COUNT(DISTINCT transit_timestamp) AS distinct_hours,
                SUM(CASE WHEN transit_timestamp IS NULL THEN 1 ELSE 0 END) AS null_ts,
                SUM(CASE WHEN EXTRACT(minute FROM transit_timestamp) != 0
                          OR EXTRACT(second FROM transit_timestamp) != 0
                         THEN 1 ELSE 0 END) AS non_hourly,
                {metric_null_expr},
                {metric_negative_expr}
            FROM read_parquet('{p}')
            """
        ).fetchdf().iloc[0].to_dict()

        expected_start = pd.Timestamp(spec["expected_start"])
        expected_end = pd.Timestamp(spec["expected_end"])
        expected_hours = len(pd.date_range(expected_start, expected_end, freq="h"))

        source_failures: list[str] = []
        if pd.isna(stats["min_ts"]) or pd.Timestamp(stats["min_ts"]) != expected_start:
            source_failures.append(f"start={stats['min_ts']} (expected {expected_start})")
        if pd.isna(stats["max_ts"]) or pd.Timestamp(stats["max_ts"]) != expected_end:
            source_failures.append(f"end={stats['max_ts']} (expected {expected_end})")
        if int(stats["distinct_hours"]) != expected_hours:
            source_failures.append(f"distinct_hours={int(stats['distinct_hours'])} (expected {expected_hours})")
        if int(stats["null_ts"]) or int(stats["non_hourly"]):
            source_failures.append(f"null_ts={int(stats['null_ts'])}; non_hourly={int(stats['non_hourly'])}")

        if source_failures:
            add(report, "FAIL", section, "hourly coverage", "; ".join(source_failures))
        else:
            add(report, "PASS", section, "hourly coverage", "Complete expected hourly coverage.", expected_hours)

        for col in spec["metric_columns"]:
            nulls = int(stats[f"{col}_nulls"])
            negs = int(stats[f"{col}_negative"])
            if nulls:
                add(report, "FAIL", section, f"{col} missingness", "Stored metric has null values.", nulls)
            else:
                add(report, "PASS", section, f"{col} missingness", "No stored null values.", 0)

            if negs:
                add(report, "FAIL", section, f"{col} negatives", "Negative count values found.", negs)
            else:
                add(report, "PASS", section, f"{col} negatives", "No negative values.", 0)

        print(f"  {name}: {int(stats['row_count']):,} rows")

    con.close()


# ---------------------------------------------------------------------
# Output and CLI
# ---------------------------------------------------------------------
def write_outputs(report: list[dict[str, Any]], panel_path: Path) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(report)
    frame["severity_rank"] = frame["status"].map(SEVERITY_ORDER)
    frame = frame.sort_values(["severity_rank", "section", "check"], ascending=[False, True, True])
    frame = frame.drop(columns="severity_rank")

    csv_path = OUTPUT_DIR / "final_quality_check_report.csv"
    json_path = OUTPUT_DIR / "final_quality_check_summary.json"
    frame.to_csv(csv_path, index=False)

    counts = frame["status"].value_counts().to_dict()
    summary = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "panel_file": str(panel_path),
        "summary_counts": {key: int(counts.get(key, 0)) for key in ["PASS", "WARNING", "FAIL"]},
        "submission_guidance": "Do not submit with FAIL findings unresolved. Warnings require interpretation, not necessarily code changes.",
        "report_file": str(csv_path),
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return csv_path, json_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Final thesis data-quality check (read-only).")
    parser.add_argument("--panel", help="Optional path to a specific final panel parquet file.")
    parser.add_argument(
        "--no-source-checks",
        action="store_true",
        help="Check only the final panel; skip saved master-file coverage checks.",
    )
    args = parser.parse_args()

    report = new_report()
    panel_path = choose_panel(args.panel)
    check_panel(panel_path, report)

    if not args.no_source_checks:
        check_master_sources(report)

    csv_path, json_path = write_outputs(report, panel_path)
    frame = pd.DataFrame(report)
    counts = frame["status"].value_counts().to_dict()

    print("\n" + "=" * 72)
    print("FINAL QUALITY-CHECK SUMMARY")
    print("=" * 72)
    for status in ["FAIL", "WARNING", "PASS"]:
        print(f"{status}: {int(counts.get(status, 0))}")

    if not frame.loc[frame["status"] == "FAIL"].empty:
        print("\nFAIL findings:")
        for _, row in frame.loc[frame["status"] == "FAIL"].iterrows():
            print(f"- [{row['section']}] {row['check']}: {row['detail']}")

    print(f"\nSaved CSV report: {csv_path}")
    print(f"Saved JSON summary: {json_path}")
    return 1 if int(counts.get("FAIL", 0)) else 0


if __name__ == "__main__":
    sys.exit(main())
