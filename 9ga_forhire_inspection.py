import duckdb
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# MAIN FOR-HIRE PANEL INSPECTION — DUCKDB SAFE
# ============================================================

path = r"data/processed/forhire_master_2024_01_2026_03.parquet"
p = path.replace("\\", "/")

con = duckdb.connect()

print("=" * 80)
print("SHAPE")
row_count = con.execute(
    f"SELECT COUNT(*) FROM read_parquet('{p}')"
).fetchone()[0]

cols = con.execute(
    f"DESCRIBE SELECT * FROM read_parquet('{p}')"
).fetchdf()

print((row_count, len(cols)))

print("\n" + "=" * 80)
print("COLUMNS")
for c in cols["column_name"]:
    print(c)

print("\n" + "=" * 80)
print("HEAD")
print(
    con.execute(
        f"""
        SELECT *
        FROM read_parquet('{p}')
        LIMIT 10
        """
    ).fetchdf()
)

print("\n" + "=" * 80)
print("DATE RANGE CHECK")
print(
    con.execute(
        f"""
        SELECT
            MIN(transit_timestamp) AS min_transit_timestamp,
            MAX(transit_timestamp) AS max_transit_timestamp,
            SUM(CASE WHEN transit_timestamp IS NULL THEN 1 ELSE 0 END) AS missing_transit_timestamp
        FROM read_parquet('{p}')
        """
    ).fetchdf()
)

print("\n" + "=" * 80)
print("ZONE COLUMNS CHECK")
zone_candidates = [
    c for c in cols["column_name"].tolist()
    if "location" in c.lower()
    or "zone" in c.lower()
    or "borough" in c.lower()
    or "pickup" in c.lower()
    or "dropoff" in c.lower()
]
print(zone_candidates)

print("\n" + "=" * 80)
print("NUMERIC SUMMARY")
print(
    con.execute(
        f"""
        SELECT
            SUM(forhire_trip_count) AS total_forhire_trips,
            SUM(total_trip_miles) AS total_trip_miles,
            AVG(avg_trip_miles) AS avg_trip_miles,
            SUM(total_base_passenger_fare) AS total_base_passenger_fare,
            SUM(driver_pay_sum) AS total_driver_pay,
            SUM(cbd_congestion_fee_sum) AS total_cbd_congestion_fee,
            SUM(congestion_surcharge_sum) AS total_congestion_surcharge
        FROM read_parquet('{p}')
        """
    ).fetchdf()
)

print("\n" + "=" * 80)
print("TRIP PROVIDER TOTALS")
print(
    con.execute(
        f"""
        SELECT
            SUM(uber_trip_count) AS uber_trips,
            SUM(lyft_trip_count) AS lyft_trips,
            SUM(via_trip_count) AS via_trips,
            SUM(juno_trip_count) AS juno_trips,
            SUM(forhire_trip_count) AS total_forhire_trips
        FROM read_parquet('{p}')
        """
    ).fetchdf()
)

print("\n" + "=" * 80)
print("TOP PICKUP LOCATION IDS")
print(
    con.execute(
        f"""
        SELECT
            pickup_location_id,
            SUM(forhire_trip_count) AS trips
        FROM read_parquet('{p}')
        GROUP BY pickup_location_id
        ORDER BY trips DESC
        LIMIT 25
        """
    ).fetchdf()
)

print("\n" + "=" * 80)
print("TOP DROPOFF LOCATION IDS")
print(
    con.execute(
        f"""
        SELECT
            dropoff_location_id,
            SUM(forhire_trip_count) AS trips
        FROM read_parquet('{p}')
        GROUP BY dropoff_location_id
        ORDER BY trips DESC
        LIMIT 25
        """
    ).fetchdf()
)

print("\n" + "=" * 80)
print("HOURLY COVERAGE CHECK")
print(
    con.execute(
        f"""
        SELECT
            COUNT(DISTINCT transit_timestamp) AS observed_hours,
            MIN(transit_timestamp) AS min_hour,
            MAX(transit_timestamp) AS max_hour
        FROM read_parquet('{p}')
        """
    ).fetchdf()
)

print("\n" + "=" * 80)
print("MONTHLY TRIP TOTALS")
print(
    con.execute(
        f"""
        SELECT
            strftime(transit_timestamp, '%Y-%m') AS year_month,
            SUM(forhire_trip_count) AS forhire_trips,
            SUM(uber_trip_count) AS uber_trips,
            SUM(lyft_trip_count) AS lyft_trips
        FROM read_parquet('{p}')
        GROUP BY year_month
        ORDER BY year_month
        """
    ).fetchdf()
)

# ============================================================
# TAXI ZONE SHAPEFILE MAPPING
# ============================================================

shapefile_path = r"data/raw/taxi_zones/taxi_zones.shp"

gdf = gpd.read_file(shapefile_path)

print("\n" + "=" * 80)
print("TAXI ZONE SHAPEFILE CHECK")

print("\nColumns:")
print(gdf.columns)

print("\nShape:")
print(gdf.shape)

print("\nHead:")
print(gdf.head())

gdf["centroid"] = gdf.geometry.centroid
gdf["x"] = gdf["centroid"].x
gdf["y"] = gdf["centroid"].y

print("\nCentroid preview:")
print(
    gdf[
        ["LocationID", "zone", "borough", "x", "y"]
    ].head(20)
)

fig, ax = plt.subplots(figsize=(12, 12))

gdf.plot(
    ax=ax,
    alpha=0.2,
    edgecolor="gray",
)

ax.scatter(
    gdf["x"],
    gdf["y"],
    s=20,
)

for _, row in gdf.iterrows():
    ax.text(
        row["x"],
        row["y"],
        str(row["LocationID"]),
        fontsize=6,
    )

plt.title("NYC TLC Taxi Zone Centroids")
plt.tight_layout()
plt.show()

# ============================================================
# FINAL FOR-HIRE DID PANEL INSPECTION
# ============================================================

print("\n" + "=" * 80)
print("FINAL FOR-HIRE DID PANEL INSPECTION")
print("=" * 80)

did_path = r"data/processed/forhire_did_panel.parquet"
did_p = did_path.replace("\\", "/")

try:
    did = pd.read_parquet(did_path)

    print("\n" + "=" * 80)
    print("DID PANEL SHAPE")
    print(did.shape)

    print("\n" + "=" * 80)
    print("DID PANEL COLUMNS")
    for c in did.columns:
        print(c)

    print("\n" + "=" * 80)
    print("DID PANEL HEAD")
    print(did.head(10))

    # --------------------------------------------------------
    # Basic required-column checks
    # --------------------------------------------------------

    required_did_cols = {
        "transit_timestamp",
        "forhire_trips",
        "treated_group",
        "forhire_zone_group",
    }

    missing_required = required_did_cols - set(did.columns)

    print("\n" + "=" * 80)
    print("REQUIRED COLUMN CHECK")
    if missing_required:
        print("Missing required columns:")
        print(missing_required)
    else:
        print("All required FHV DiD columns are present.")

    # --------------------------------------------------------
    # Date and hourly coverage
    # --------------------------------------------------------

    did["transit_timestamp"] = pd.to_datetime(did["transit_timestamp"])

    min_ts = did["transit_timestamp"].min()
    max_ts = did["transit_timestamp"].max()

    full_hours = pd.date_range(
        start=min_ts,
        end=max_ts,
        freq="h",
    )

    observed_hours = (
        did["transit_timestamp"]
        .drop_duplicates()
        .sort_values()
    )

    missing_hours = full_hours.difference(observed_hours)

    print("\n" + "=" * 80)
    print("DID PANEL DATE RANGE AND HOURLY COVERAGE")
    print("Minimum timestamp:", min_ts)
    print("Maximum timestamp:", max_ts)
    print("Expected hourly timestamps:", len(full_hours))
    print("Observed hourly timestamps:", observed_hours.nunique())
    print("Missing hourly timestamps:", len(missing_hours))

    if len(missing_hours) > 0:
        print("First missing hourly timestamps:")
        print(missing_hours[:20])

    # --------------------------------------------------------
    # Rows per hour
    # Expected: normally 2 rows per hour, one control and one treated
    # --------------------------------------------------------

    rows_per_hour = (
        did.groupby("transit_timestamp")
        .size()
        .reset_index(name="rows_per_hour")
    )

    print("\n" + "=" * 80)
    print("ROWS PER HOUR CHECK")
    print(rows_per_hour["rows_per_hour"].value_counts().sort_index())

    unusual_rows = rows_per_hour[rows_per_hour["rows_per_hour"] != 2]

    print("Hours with something other than 2 treated/control rows:")
    print(len(unusual_rows))

    if len(unusual_rows) > 0:
        print(unusual_rows.head(20))

    # --------------------------------------------------------
    # Treatment/control group structure
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("TREATED GROUP COUNTS")
    print(did["treated_group"].value_counts(dropna=False).sort_index())

    print("\n" + "=" * 80)
    print("FOR-HIRE ZONE GROUP COUNTS")
    print(did["forhire_zone_group"].value_counts(dropna=False))

    if {"treated_group", "forhire_zone_group"}.issubset(did.columns):
        print("\n" + "=" * 80)
        print("TREATED GROUP × ZONE GROUP CROSS-TAB")
        print(
            pd.crosstab(
                did["treated_group"],
                did["forhire_zone_group"],
                dropna=False,
            )
        )

    # --------------------------------------------------------
    # Duplicate key check
    # Expected: one row per timestamp × treated_group
    # --------------------------------------------------------

    duplicate_keys = did.duplicated(
        subset=["transit_timestamp", "treated_group"]
    ).sum()

    print("\n" + "=" * 80)
    print("DUPLICATE KEY CHECK")
    print("Duplicate timestamp × treated_group rows:", duplicate_keys)

    # --------------------------------------------------------
    # Missing values in key model columns
    # --------------------------------------------------------

    key_cols = [
        "transit_timestamp",
        "forhire_trips",
        "treated_group",
        "forhire_zone_group",
    ]

    optional_cols = [
        "uber_trips",
        "lyft_trips",
        "date",
        "hour",
        "day_of_week",
        "is_weekend",
        "post_congestion_pricing",
        "holiday_flag",
        "severe_weather_flag",
        "major_event_flag",
    ]

    key_cols = [c for c in key_cols + optional_cols if c in did.columns]

    print("\n" + "=" * 80)
    print("MISSING VALUES IN KEY DID COLUMNS")
    print(did[key_cols].isna().sum())

    # --------------------------------------------------------
    # Outcome validity
    # --------------------------------------------------------

    did["forhire_trips"] = pd.to_numeric(
        did["forhire_trips"],
        errors="coerce",
    )

    print("\n" + "=" * 80)
    print("OUTCOME VALIDITY")
    print("Missing forhire_trips:", did["forhire_trips"].isna().sum())
    print("Negative forhire_trips rows:", (did["forhire_trips"] < 0).sum())
    print("Zero forhire_trips rows:", (did["forhire_trips"] == 0).sum())

    print("\nFor-hire trip summary:")
    print(did["forhire_trips"].describe())

    # --------------------------------------------------------
    # Pre/post and monthly means
    # --------------------------------------------------------

    if "post_congestion_pricing" not in did.columns:
        policy_start = pd.Timestamp("2025-01-05 00:00:00")
        did["post_congestion_pricing"] = (
            did["transit_timestamp"] >= policy_start
        ).astype(int)

    if "year_month" not in did.columns:
        did["year_month"] = (
            did["transit_timestamp"]
            .dt.to_period("M")
            .astype(str)
        )

    print("\n" + "=" * 80)
    print("PRE/POST HOURLY MEAN FOR-HIRE TRIPS BY GROUP")
    prepost_means = (
        did.groupby(
            ["post_congestion_pricing", "forhire_zone_group"],
            observed=True,
        )["forhire_trips"]
        .mean()
        .reset_index()
    )
    print(prepost_means.to_string(index=False))

    print("\n" + "=" * 80)
    print("MONTHLY TREATED/CONTROL MEANS")
    monthly_means = (
        did.groupby(
            ["year_month", "forhire_zone_group"],
            observed=True,
        )["forhire_trips"]
        .mean()
        .reset_index()
    )

    monthly_pivot = (
        monthly_means
        .pivot(
            index="year_month",
            columns="forhire_zone_group",
            values="forhire_trips",
        )
        .reset_index()
    )

    if {"core_crz_pickup", "outside_crz_pickup"}.issubset(monthly_pivot.columns):
        monthly_pivot["treated_minus_control"] = (
            monthly_pivot["core_crz_pickup"]
            - monthly_pivot["outside_crz_pickup"]
        )

        monthly_pivot["treated_over_control"] = (
            monthly_pivot["core_crz_pickup"]
            / monthly_pivot["outside_crz_pickup"].replace(0, np.nan)
        )

    print(monthly_pivot.to_string(index=False))

except FileNotFoundError:
    print(f"FHV DiD panel not found: {did_path}")
    print("Run the upstream script that creates data/processed/forhire_did_panel.parquet first.")