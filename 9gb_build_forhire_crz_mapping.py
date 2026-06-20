# 9gb_build_forhire_treatment_groups.py

import os
from pathlib import Path

import duckdb
import pandas as pd

# ---------------------------------------------------------------------
# 9gb Build for-hire CRZ treatment groups
#
# Mirrors the taxi CRZ treatment-group script, but uses DuckDB because
# the for-hire master is too large to load into pandas.
#
# Input:
#   data/processed/forhire_master_2024_01_2026_03.parquet
#
# Output:
#   data/processed/forhire_master_with_crz_groups.parquet
#   data/mappings/forhire_crz_location_mapping.csv
# ---------------------------------------------------------------------

INPUT = Path("data/processed/forhire_master_2024_01_2026_03.parquet")

OUT_FORHIRE = Path("data/processed/forhire_master_with_crz_groups.parquet")
OUT_MAPPING = Path("data/mappings/forhire_crz_location_mapping.csv")

os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/mappings", exist_ok=True)

CORE_CRZ_IDS = {
    12, 88, 261, 87, 231, 45, 232, 148, 144, 211,
    125, 158, 249, 113, 114, 79, 4, 246, 68, 90,
    234, 107, 224, 137, 170, 164, 186, 50, 48,
    230, 100, 163, 161, 162, 233, 229,
    13,
}

BORDER_CRZ_IDS = {
    143, 142, 237, 141, 140,
}


def sql_id_list(values):
    return ", ".join(str(int(v)) for v in sorted(values))


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing file: {INPUT}")

    input_path = str(INPUT).replace("\\", "/")
    output_path = str(OUT_FORHIRE).replace("\\", "/")

    core_ids = sql_id_list(CORE_CRZ_IDS)
    border_ids = sql_id_list(BORDER_CRZ_IDS)

    con = duckdb.connect()

    print("=" * 90)
    print("Building for-hire CRZ treatment groups with DuckDB")
    print("=" * 90)
    print(f"Input:  {INPUT}")
    print(f"Output: {OUT_FORHIRE}")
    print()

    query = f"""
    COPY (
        WITH base AS (
            SELECT
                *,

                CASE
                    WHEN pickup_location_id IS NULL THEN 'missing'
                    WHEN CAST(pickup_location_id AS INTEGER) IN ({core_ids}) THEN 'core_crz'
                    WHEN CAST(pickup_location_id AS INTEGER) IN ({border_ids}) THEN 'border_crz'
                    ELSE 'outside_crz'
                END AS pickup_crz_group,

                CASE
                    WHEN dropoff_location_id IS NULL THEN 'missing'
                    WHEN CAST(dropoff_location_id AS INTEGER) IN ({core_ids}) THEN 'core_crz'
                    WHEN CAST(dropoff_location_id AS INTEGER) IN ({border_ids}) THEN 'border_crz'
                    ELSE 'outside_crz'
                END AS dropoff_crz_group

            FROM read_parquet('{input_path}')
        ),

        flags AS (
            SELECT
                *,

                CASE WHEN pickup_crz_group = 'core_crz' THEN 1 ELSE 0 END
                    AS pickup_in_core_crz,

                CASE WHEN dropoff_crz_group = 'core_crz' THEN 1 ELSE 0 END
                    AS dropoff_in_core_crz,

                CASE WHEN pickup_crz_group = 'border_crz' THEN 1 ELSE 0 END
                    AS pickup_in_border_crz,

                CASE WHEN dropoff_crz_group = 'border_crz' THEN 1 ELSE 0 END
                    AS dropoff_in_border_crz,

                CASE WHEN pickup_crz_group = 'outside_crz' THEN 1 ELSE 0 END
                    AS pickup_outside_crz,

                CASE WHEN dropoff_crz_group = 'outside_crz' THEN 1 ELSE 0 END
                    AS dropoff_outside_crz

            FROM base
        )

        SELECT
            *,

            pickup_in_core_crz AS treated_group,

            CASE
                WHEN pickup_crz_group != 'border_crz' THEN 1
                ELSE 0
            END AS main_did_sample,

            CASE
                WHEN pickup_crz_group = 'missing'
                  OR dropoff_crz_group = 'missing'
                THEN 'missing'
                ELSE pickup_crz_group || '_to_' || dropoff_crz_group
            END AS crz_flow_type,

            CASE
                WHEN pickup_in_border_crz = 1
                  OR dropoff_in_border_crz = 1
                THEN 'border_related'

                WHEN pickup_in_core_crz = 1
                  AND dropoff_in_core_crz = 1
                THEN 'in_to_in'

                WHEN pickup_in_core_crz = 0
                  AND dropoff_in_core_crz = 1
                THEN 'out_to_in'

                WHEN pickup_in_core_crz = 1
                  AND dropoff_in_core_crz = 0
                THEN 'in_to_out'

                ELSE 'out_to_out'
            END AS simple_crz_flow_type

        FROM flags
    )
    TO '{output_path}'
    (FORMAT PARQUET);
    """

    con.execute(query)

    mapping_query = f"""
    WITH ids AS (
        SELECT DISTINCT CAST(pickup_location_id AS INTEGER) AS location_id
        FROM read_parquet('{input_path}')
        WHERE pickup_location_id IS NOT NULL

        UNION

        SELECT DISTINCT CAST(dropoff_location_id AS INTEGER) AS location_id
        FROM read_parquet('{input_path}')
        WHERE dropoff_location_id IS NOT NULL
    )

    SELECT
        location_id,

        CASE
            WHEN location_id IN ({core_ids}) THEN 'core_crz'
            WHEN location_id IN ({border_ids}) THEN 'border_crz'
            ELSE 'outside_crz'
        END AS crz_group,

        CASE WHEN location_id IN ({core_ids}) THEN 1 ELSE 0 END
            AS in_core_crz,

        CASE WHEN location_id IN ({border_ids}) THEN 1 ELSE 0 END
            AS in_border_crz,

        CASE
            WHEN location_id NOT IN ({core_ids})
             AND location_id NOT IN ({border_ids})
            THEN 1 ELSE 0
        END AS outside_crz

    FROM ids
    ORDER BY location_id
    """

    mapping = con.execute(mapping_query).fetchdf()
    mapping.to_csv(OUT_MAPPING, index=False)

    out_path = str(OUT_FORHIRE).replace("\\", "/")

    row_count = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{out_path}')"
    ).fetchone()[0]

    trip_count = con.execute(
        f"SELECT SUM(forhire_trip_count) FROM read_parquet('{out_path}')"
    ).fetchone()[0]

    pickup_counts = con.execute(
        f"""
        SELECT
            pickup_crz_group,
            COUNT(*) AS rows,
            SUM(forhire_trip_count) AS trips
        FROM read_parquet('{out_path}')
        GROUP BY pickup_crz_group
        ORDER BY trips DESC
        """
    ).fetchdf()

    dropoff_counts = con.execute(
        f"""
        SELECT
            dropoff_crz_group,
            COUNT(*) AS rows,
            SUM(forhire_trip_count) AS trips
        FROM read_parquet('{out_path}')
        GROUP BY dropoff_crz_group
        ORDER BY trips DESC
        """
    ).fetchdf()

    sample_counts = con.execute(
        f"""
        SELECT
            main_did_sample,
            COUNT(*) AS rows,
            SUM(forhire_trip_count) AS trips
        FROM read_parquet('{out_path}')
        GROUP BY main_did_sample
        ORDER BY main_did_sample
        """
    ).fetchdf()

    treated_counts = con.execute(
        f"""
        SELECT
            treated_group,
            COUNT(*) AS rows,
            SUM(forhire_trip_count) AS trips
        FROM read_parquet('{out_path}')
        WHERE main_did_sample = 1
        GROUP BY treated_group
        ORDER BY treated_group
        """
    ).fetchdf()

    flow_counts = con.execute(
        f"""
        SELECT
            simple_crz_flow_type,
            COUNT(*) AS rows,
            SUM(forhire_trip_count) AS trips
        FROM read_parquet('{out_path}')
        GROUP BY simple_crz_flow_type
        ORDER BY trips DESC
        """
    ).fetchdf()

    print("=" * 90)
    print("9gb for-hire CRZ treatment groups complete")
    print("=" * 90)
    print(f"Saved for-hire file to: {OUT_FORHIRE}")
    print(f"Saved mapping file to:  {OUT_MAPPING}")
    print()

    print("Rows:")
    print(f"{row_count:,}")
    print()

    print("Trips:")
    print(f"{int(trip_count):,}")
    print()

    print("Pickup CRZ group counts:")
    print(pickup_counts)
    print()

    print("Dropoff CRZ group counts:")
    print(dropoff_counts)
    print()

    print("Main DiD sample counts:")
    print(sample_counts)
    print()

    print("Treated group counts inside main DiD sample:")
    print(treated_counts)
    print()

    print("Simple CRZ flow type counts:")
    print(flow_counts)
    print()

    print("Mapping counts:")
    print(mapping["crz_group"].value_counts(dropna=False))
    print()

    print("Core CRZ IDs:")
    print(sorted(CORE_CRZ_IDS))
    print()

    print("Border CRZ IDs:")
    print(sorted(BORDER_CRZ_IDS))


if __name__ == "__main__":
    main()
