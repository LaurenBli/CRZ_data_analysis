"""
13_peak_toll_period_analysis.py

Purpose
-------
Re-estimate the thesis's mode-specific Difference-in-Differences and event-study
models using only hours in which the CRZ peak toll applies:

    * Weekdays: 05:00–20:59
    * Weekends: 09:00–20:59

The script treats the input timestamps/hours as New York local time. Confirm that
all source panels use local time before running it.

This is a PEAK-PERIOD-ONLY robustness analysis. It estimates post-policy changes
within peak-toll observations; it does not itself test whether peak effects differ
statistically from overnight effects. That would require a separate peak × post ×
treated interaction model.

Expected inputs (run from repository root)
-------------------------------------------
    data/processed/taxi_did_panel.parquet
    data/processed/forhire_did_panel.parquet
    data/processed/subway_did_panel.parquet
    data/processed/bus_did_panel.parquet
    data/processed/bridge_all_directions_did_panel.parquet

Outputs
-------
    outputs/models/13_peak_toll_period_did_summary.csv
    outputs/models/13_peak_toll_period_event_study_coefficients.csv
    outputs/models/13_peak_toll_period_sample_counts.csv
    outputs/models/13_peak_toll_period_results.txt
    outputs/figures/13_peak_toll_period_event_studies.png

Dependencies
------------
    pandas, numpy, statsmodels, matplotlib, pyarrow
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
POLICY_DATE = pd.Timestamp("2025-01-05")
EVENT_MIN = -12
EVENT_MAX = 12
REFERENCE_EVENT_TIME = -1  # December 2024

ROOT = Path.cwd()
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "outputs" / "models"
FIGURE_DIR = ROOT / "outputs" / "figures"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

OUT_DID = MODEL_DIR / "13_peak_toll_period_did_summary.csv"
OUT_EVENT = MODEL_DIR / "13_peak_toll_period_event_study_coefficients.csv"
OUT_COUNTS = MODEL_DIR / "13_peak_toll_period_sample_counts.csv"
OUT_TXT = MODEL_DIR / "13_peak_toll_period_results.txt"
OUT_FIG = FIGURE_DIR / "13_peak_toll_period_event_studies.png"


@dataclass(frozen=True)
class ModeSpec:
    mode: str
    filename: str
    outcome: str
    group_fe_candidates: tuple[str, ...]
    interpretation_note: str = ""


SPECS = [
    ModeSpec(
        "Taxi",
        "taxi_did_panel.parquet",
        "taxi_trips",
        ("taxi_zone_group", "pickup_zone_group", "zone_group", "taxi_group"),
        "Treated group: core CRZ pickup zones; comparison group: outside-CRZ pickup zones.",
    ),
    ModeSpec(
        "For-hire vehicle",
        "forhire_did_panel.parquet",
        "forhire_trips",
        ("forhire_zone_group", "fhv_zone_group", "pickup_zone_group", "zone_group", "forhire_group"),
        "Treated group: core CRZ pickup zones; comparison group: outside-CRZ pickup zones.",
    ),
    ModeSpec(
        "Subway",
        "subway_did_panel.parquet",
        "subway_ridership",
        ("subway_station_group", "station_group", "subway_group"),
        "Treated group: CRZ/CBD stations; comparison group: outside-CRZ stations.",
    ),
    ModeSpec(
        "Bus",
        "bus_did_panel.parquet",
        "bus_ridership",
        ("bus_route_group", "route_group", "bus_group"),
        "Treated group: CRZ-serving routes; comparison group: routes with no CRZ-designated stops.",
    ),
    ModeSpec(
        "Bridge and tunnel",
        "bridge_all_directions_did_panel.parquet",
        "bridge_traffic",
        ("bridge_group", "facility_group", "facility_id", "facility"),
        "Aggregate facility-group comparison; spillover-prone comparison facilities require cautious interpretation.",
    ),
]


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------
def find_datetime_column(df: pd.DataFrame) -> str:
    """Return the most likely timestamp/date column."""
    for col in ("transit_timestamp", "timestamp", "datetime", "date"):
        if col in df.columns:
            return col
    raise KeyError("No usable datetime column found. Expected one of: transit_timestamp, timestamp, datetime, date.")


def prepare_panel(raw: pd.DataFrame) -> pd.DataFrame:
    """Create common time, treatment, and control columns without changing outcomes."""
    d = raw.copy()
    date_col = find_datetime_column(d)
    parsed = pd.to_datetime(d[date_col], errors="coerce")
    if parsed.isna().all():
        raise ValueError(f"Could not parse any values in datetime column '{date_col}'.")

    # Some panels store calendar date in `date` and hour separately. Reconstruct a
    # proper timestamp in that case. If the datetime already contains non-zero hours,
    # retain it and use the explicit hour column only as a consistency check.
    if "hour" in d.columns:
        hour_num = pd.to_numeric(d["hour"], errors="coerce")
        if hour_num.isna().any() or not hour_num.between(0, 23).all():
            raise ValueError("The 'hour' column contains missing or invalid values outside 0–23.")
        if parsed.dt.hour.eq(0).all():
            timestamp = parsed.dt.normalize() + pd.to_timedelta(hour_num, unit="h")
        else:
            timestamp = parsed
            d["hour"] = timestamp.dt.hour
    else:
        timestamp = parsed
        d["hour"] = timestamp.dt.hour

    d["timestamp"] = timestamp
    d["calendar_date"] = timestamp.dt.normalize()
    d["day_of_week"] = timestamp.dt.dayofweek  # Monday = 0, Sunday = 6
    d["year_month"] = timestamp.dt.to_period("M").astype(str)
    d["date_cluster"] = d["calendar_date"].astype(str)

    if "treated_group" not in d.columns:
        raise KeyError("Panel is missing 'treated_group', required for Difference-in-Differences estimation.")
    d["treated_group"] = pd.to_numeric(d["treated_group"], errors="coerce")
    if d["treated_group"].isna().any() or not set(d["treated_group"].dropna().unique()).issubset({0, 1}):
        raise ValueError("'treated_group' must be a complete binary 0/1 variable.")

    d["post_congestion_pricing"] = (d["calendar_date"] >= POLICY_DATE).astype(int)
    d["event_time"] = (
        (timestamp.dt.year - POLICY_DATE.year) * 12
        + (timestamp.dt.month - POLICY_DATE.month)
    ).astype(int)

    # Preserve an existing trend index where available. Otherwise use elapsed calendar
    # days, which keeps the trend tied to real time rather than the number of retained rows.
    if "time_index" not in d.columns:
        d["time_index"] = (timestamp - timestamp.min()).dt.total_seconds() / 86_400.0
    else:
        d["time_index"] = pd.to_numeric(d["time_index"], errors="coerce")
        if d["time_index"].isna().any():
            d["time_index"] = (timestamp - timestamp.min()).dt.total_seconds() / 86_400.0

    return d


def peak_toll_mask(df: pd.DataFrame) -> pd.Series:
    """Return True only for CRZ peak-toll hours.

    Weekday peak: 05:00–20:59 (Monday–Friday)
    Weekend peak: 09:00–20:59 (Saturday–Sunday)
    """
    weekday = df["day_of_week"].between(0, 4)
    weekend = df["day_of_week"].between(5, 6)
    weekday_peak = weekday & df["hour"].between(5, 20)
    weekend_peak = weekend & df["hour"].between(9, 20)
    return weekday_peak | weekend_peak


def select_group_fe(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """Return the available treatment-group label for reporting only.

    The processed panels are aggregate treated/control panels. Adding a group fixed
    effect together with treated_group creates a redundant design matrix, so this
    variable is deliberately *not* included as a fixed effect in the peak models.
    """
    for col in candidates:
        if col in df.columns and df[col].nunique(dropna=True) > 1:
            return col
    return None


def available_controls(df: pd.DataFrame, include_year_month: bool) -> list[str]:
    """Build the common fixed-effect/control block available in a panel."""
    terms: list[str] = ["C(hour)", "C(day_of_week)"]
    if include_year_month:
        terms.append("C(year_month)")
    for col in ("holiday_flag", "severe_weather_flag"):
        if col in df.columns and df[col].nunique(dropna=True) > 1:
            terms.append(col)
    return terms


def fit_clustered_ols(formula: str, df: pd.DataFrame):
    """Fit OLS with calendar-date clustered covariance and validate output."""
    model = smf.ols(formula=formula, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["date_cluster"]},
    )
    exog = model.model.exog
    if np.linalg.matrix_rank(exog) < exog.shape[1]:
        raise ValueError(
            "Rank-deficient design matrix. Do not interpret the model; check for redundant fixed effects or controls."
        )
    if not (np.isfinite(model.params).all() and np.isfinite(model.bse).all() and np.isfinite(model.pvalues).all()):
        raise ValueError(
            "Non-finite coefficient, standard error, or p-value detected. Do not interpret the model output."
        )
    return model


def interaction_name(params: pd.Index) -> str:
    """Locate the DiD post × treated coefficient robustly."""
    candidates = [
        name
        for name in params
        if "post_congestion_pricing" in name and "treated_group" in name
    ]
    if len(candidates) != 1:
        raise KeyError(
            "Could not uniquely identify the post × treated coefficient. "
            f"Candidates found: {candidates}"
        )
    return candidates[0]


def result_row(mode: str, model_label: str, model, term: str, is_log: bool) -> dict[str, object]:
    ci = model.conf_int().loc[term]
    coef = float(model.params[term])
    row: dict[str, object] = {
        "mode": mode,
        "model": model_label,
        "term": term,
        "coef": coef,
        "std_err": float(model.bse[term]),
        "p_value": float(model.pvalues[term]),
        "ci_low": float(ci.iloc[0]),
        "ci_high": float(ci.iloc[1]),
        "r_squared": float(model.rsquared),
        "n_obs": int(model.nobs),
        "clusters": int(pd.Series(model.model.data.frame["date_cluster"]).nunique()),
        "percent_effect": 100.0 * (np.exp(coef) - 1.0) if is_log else np.nan,
    }
    return row


# -----------------------------------------------------------------------------
# Difference-in-Differences
# -----------------------------------------------------------------------------
def run_did(mode: ModeSpec, df: pd.DataFrame, group_fe: str | None) -> list[dict[str, object]]:
    """Estimate baseline and trend-adjusted level/log DiD models.

    The aggregate panels contain treated and comparison group series. The group
    indicator is represented by treated_group, so no additional group fixed effect
    is included; doing so would duplicate the treatment-group main effect.
    """
    d = df.dropna(subset=[mode.outcome, "treated_group", "post_congestion_pricing"]).copy()
    d = d[d[mode.outcome] >= 0].copy()
    if d.empty:
        raise ValueError(f"No usable peak-toll observations remain for {mode.mode}.")

    d["log_outcome"] = np.log1p(pd.to_numeric(d[mode.outcome], errors="coerce"))
    d = d.dropna(subset=["log_outcome"])

    control_terms = available_controls(d, include_year_month=True)
    rhs_base = "post_congestion_pricing * treated_group"
    if control_terms:
        rhs_base += " + " + " + ".join(control_terms)
    rhs_trend = rhs_base + " + treated_group:time_index"

    models = [
        ("Level baseline", f"{mode.outcome} ~ {rhs_base}", False),
        ("Log baseline", f"log_outcome ~ {rhs_base}", True),
        ("Level trend-adjusted", f"{mode.outcome} ~ {rhs_trend}", False),
        ("Log trend-adjusted", f"log_outcome ~ {rhs_trend}", True),
    ]

    rows: list[dict[str, object]] = []
    for label, formula, is_log in models:
        fitted = fit_clustered_ols(formula, d)
        term = interaction_name(fitted.params.index)
        rows.append(result_row(mode.mode, label, fitted, term, is_log))
    return rows


# -----------------------------------------------------------------------------
# Event study
# -----------------------------------------------------------------------------
def find_event_term(params: pd.Index, event_time: int) -> str | None:
    event_token = f"[T.{event_time}]"
    candidates = [
        name for name in params
        if event_token in name and "treated_group" in name and "event_time_cat" in name
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def pretrend_test(model, pre_terms: list[str]) -> tuple[float | None, float | None]:
    """Joint Wald/F-test that all retained pre-policy interaction coefficients are zero."""
    if not pre_terms:
        return None, None
    r = np.zeros((len(pre_terms), len(model.params)))
    name_to_pos = {name: pos for pos, name in enumerate(model.params.index)}
    for row, term in enumerate(pre_terms):
        r[row, name_to_pos[term]] = 1.0
    test = model.f_test(r)
    f_value = float(np.asarray(test.fvalue).squeeze())
    p_value = float(np.asarray(test.pvalue).squeeze())
    return f_value, p_value


def run_event_study(mode: ModeSpec, df: pd.DataFrame, group_fe: str | None) -> tuple[pd.DataFrame, dict[str, object]]:
    """Estimate a log-outcome monthly event study for peak-toll observations only.

    The formula includes event-time main effects, the treatment-group main effect,
    and their interaction. December 2024 (event_time = -1) is the omitted
    reference month. No group fixed effect is added because the aggregate-panel
    group label is redundant with treated_group.
    """
    d = df.copy()
    d = d[d["event_time"].between(EVENT_MIN, EVENT_MAX)].copy()
    d = d.dropna(subset=[mode.outcome, "treated_group"])
    d = d[d[mode.outcome] >= 0].copy()
    d["log_outcome"] = np.log1p(pd.to_numeric(d[mode.outcome], errors="coerce"))
    d = d.dropna(subset=["log_outcome"])
    d["event_time_cat"] = d["event_time"].astype(str)

    if str(REFERENCE_EVENT_TIME) not in set(d["event_time_cat"]):
        raise ValueError(f"{mode.mode}: reference month {REFERENCE_EVENT_TIME} is absent after filtering.")

    control_terms = available_controls(d, include_year_month=False)
    rhs = "C(event_time_cat, Treatment(reference='-1')) * treated_group"
    if control_terms:
        rhs += " + " + " + ".join(control_terms)

    model = fit_clustered_ols(f"log_outcome ~ {rhs}", d)

    rows: list[dict[str, object]] = []
    pre_terms: list[str] = []
    post_effects: list[float] = []
    for event_time in range(EVENT_MIN, EVENT_MAX + 1):
        if event_time == REFERENCE_EVENT_TIME:
            continue
        term = find_event_term(model.params.index, event_time)
        if term is None:
            raise KeyError(f"{mode.mode}: missing event-time interaction term for {event_time}.")
        ci = model.conf_int().loc[term]
        beta = float(model.params[term])
        pct = 100.0 * (np.exp(beta) - 1.0)
        if event_time < 0:
            pre_terms.append(term)
        if event_time >= 0:
            post_effects.append(pct)
        rows.append(
            {
                "mode": mode.mode,
                "event_time": event_time,
                "term": term,
                "beta": beta,
                "std_err": float(model.bse[term]),
                "p_value": float(model.pvalues[term]),
                "ci_low": float(ci.iloc[0]),
                "ci_high": float(ci.iloc[1]),
                "percent_effect": pct,
                "percent_ci_low": 100.0 * (np.exp(float(ci.iloc[0])) - 1.0),
                "percent_ci_high": 100.0 * (np.exp(float(ci.iloc[1])) - 1.0),
                "n_obs": int(model.nobs),
                "clusters": int(d["date_cluster"].nunique()),
                "standard_error_method": "Clustered by calendar date",
            }
        )

    f_value, f_p = pretrend_test(model, pre_terms)
    summary = {
        "mode": mode.mode,
        "event_study_n": int(model.nobs),
        "event_study_clusters": int(d["date_cluster"].nunique()),
        "pretrend_f": f_value,
        "pretrend_p": f_p,
        "mean_post_policy_percent_effect": float(np.mean(post_effects)) if post_effects else np.nan,
    }
    return pd.DataFrame(rows), summary


# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------
def make_event_figure(event_df: pd.DataFrame) -> None:
    if event_df.empty:
        return
    modes = event_df["mode"].drop_duplicates().tolist()
    fig, axes = plt.subplots(len(modes), 1, figsize=(9, max(3.0, 2.6 * len(modes))), sharex=True)
    if len(modes) == 1:
        axes = [axes]

    for ax, mode in zip(axes, modes):
        sub = event_df[event_df["mode"] == mode].sort_values("event_time")
        ax.axhline(0, linewidth=1)
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.errorbar(
            sub["event_time"],
            sub["percent_effect"],
            yerr=[
                sub["percent_effect"] - sub["percent_ci_low"],
                sub["percent_ci_high"] - sub["percent_effect"],
            ],
            fmt="o-",
            capsize=3,
        )
        ax.set_title(mode, loc="left")
        ax.set_ylabel("Percent effect")
    axes[-1].set_xlabel("Months relative to CRZ implementation (December 2024 = reference)")
    fig.suptitle("Peak-toll-period event studies", y=1.01)
    fig.text(
        0.01,
        0.005,
        "Peak sample: weekdays 05:00–20:59; weekends 09:00–20:59. Calendar-date clustered 95% confidence intervals.",
        ha="left",
        fontsize=8,
    )
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close()


def write_report(
    did_df: pd.DataFrame,
    event_summary_df: pd.DataFrame,
    counts_df: pd.DataFrame,
    modes_run: list[ModeSpec],
) -> None:
    with OUT_TXT.open("w", encoding="utf-8") as f:
        f.write("=" * 92 + "\n")
        f.write("13 Peak-Toll-Period Mobility Analysis —  Specification\n")
        f.write("=" * 92 + "\n\n")
        f.write("Peak-toll sample definition\n")
        f.write("- Weekdays: 05:00–20:59\n")
        f.write("- Weekends: 09:00–20:59\n")
        f.write("- Overnight observations are excluded.\n")
        f.write("- Assumes source date/hour fields are New York local time.\n\n")
        f.write("Interpretation\n")
        f.write("This is a peak-period-only robustness analysis. It does not directly estimate\n")
        f.write("whether peak responses differ from overnight responses; that requires a\n")
        f.write("peak × post-policy × treated-group interaction model.\n\n")

        f.write("Sample counts\n")
        f.write("-" * 92 + "\n")
        f.write(counts_df.to_string(index=False))
        f.write("\n\n")

        f.write("Difference-in-Differences results\n")
        f.write("-" * 92 + "\n")
        f.write(did_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        f.write("\n\n")

        f.write("Event-study diagnostics\n")
        f.write("-" * 92 + "\n")
        f.write(event_summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        f.write("\n\n")

        f.write("Mode-specific interpretation notes\n")
        f.write("-" * 92 + "\n")
        for spec in modes_run:
            f.write(f"{spec.mode}: {spec.interpretation_note}\n")
        f.write("\nStandard errors: clustered by calendar date for all models, including bridge and tunnel.\n")
        f.write("Event-study reference month: December 2024 (event_time = -1).\n")
        f.write("January 2025 is a partial implementation month; February 2025 is the first full calendar month after implementation.\n")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    did_rows: list[dict[str, object]] = []
    event_frames: list[pd.DataFrame] = []
    event_summaries: list[dict[str, object]] = []
    count_rows: list[dict[str, object]] = []
    modes_run: list[ModeSpec] = []

    for spec in SPECS:
        path = DATA_DIR / spec.filename
        if not path.exists():
            print(f"Skipping {spec.mode}: missing input {path}")
            continue

        print(f"Loading {spec.mode}: {path}")
        raw = pd.read_parquet(path)
        if spec.outcome not in raw.columns:
            print(f"Skipping {spec.mode}: outcome column '{spec.outcome}' not found.")
            continue

        prepared = prepare_panel(raw)
        peak = prepared.loc[peak_toll_mask(prepared)].copy()
        if peak.empty:
            print(f"Skipping {spec.mode}: no peak-toll observations after filtering.")
            continue

        group_fe = select_group_fe(peak, spec.group_fe_candidates)
        count_rows.append(
            {
                "mode": spec.mode,
                "all_rows": int(len(prepared)),
                "peak_rows": int(len(peak)),
                "peak_share_percent": 100.0 * len(peak) / len(prepared),
                "first_peak_observation": peak["timestamp"].min(),
                "last_peak_observation": peak["timestamp"].max(),
                "treatment_group_label": group_fe if group_fe is not None else "Not available",
                "group_fixed_effect": "Not included: redundant with treated_group in aggregate panel",
            }
        )

        try:
            did_rows.extend(run_did(spec, peak, group_fe))
            event_df, event_summary = run_event_study(spec, peak, group_fe)
            event_frames.append(event_df)
            event_summaries.append(event_summary)
            modes_run.append(spec)
            print(f"Completed {spec.mode}")
        except Exception as exc:  # keeps other modes running if one panel differs
            print(f"{spec.mode} failed: {exc}")

    if not did_rows:
        raise RuntimeError(
            "No models were estimated. Confirm that the script is run from the repository root "
            "and that all expected processed panels exist."
        )

    did_df = pd.DataFrame(did_rows)
    event_df = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    event_summary_df = pd.DataFrame(event_summaries)
    counts_df = pd.DataFrame(count_rows)

    did_df.to_csv(OUT_DID, index=False)
    event_df.to_csv(OUT_EVENT, index=False)
    counts_df.to_csv(OUT_COUNTS, index=False)
    make_event_figure(event_df)
    write_report(did_df, event_summary_df, counts_df, modes_run)

    print("\nCompleted peak-toll-period analysis.")
    print(f"DiD summary: {OUT_DID}")
    print(f"Event-study coefficients: {OUT_EVENT}")
    print(f"Sample counts: {OUT_COUNTS}")
    print(f"Report: {OUT_TXT}")
    print(f"Figure: {OUT_FIG}")


if __name__ == "__main__":
    main()
