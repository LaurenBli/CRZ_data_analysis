from pathlib import Path
import duckdb

RAW_DIR = Path("data/raw/for-hire-volume")
PROCESSED_DIR = Path("data/processed")

OUTPUT_FILE = PROCESSED_DIR / "forhire_master_2024_01_2026_03.parquet"
FILE_PATTERN = "fhvhv_tripdata_*.parquet"

OVERWRITE_EXISTING = False

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists() and not OVERWRITE_EXISTING:
        print(f"Skipping existing file: {OUTPUT_FILE}")
        return

    files = sorted(RAW_DIR.glob(FILE_PATTERN))

    if not files:
        raise FileNotFoundError(f"No files found in {RAW_DIR}")

    print(f"Files found: {len(files)}")

    file_list_sql = ", ".join("'" + str(p).replace("\\", "/") + "'" for p in files)

    con = duckdb.connect()

    query = f"""
    COPY (
        SELECT
            date_trunc('hour', pickup_datetime) AS transit_timestamp,
            CAST(PULocationID AS INTEGER) AS pickup_location_id,
            CAST(DOLocationID AS INTEGER) AS dropoff_location_id,

            COUNT(*) AS forhire_trip_count,

            SUM(trip_miles) AS total_trip_miles,
            AVG(trip_miles) AS avg_trip_miles,

            SUM(trip_time) AS total_trip_time_seconds,
            AVG(trip_time / 60.0) AS avg_duration_minutes,

            SUM(base_passenger_fare) AS total_base_passenger_fare,
            AVG(base_passenger_fare) AS avg_base_passenger_fare,

            SUM(COALESCE(tolls, 0)) AS tolls_sum,
            SUM(COALESCE(bcf, 0)) AS bcf_sum,
            SUM(COALESCE(sales_tax, 0)) AS sales_tax_sum,
            SUM(COALESCE(congestion_surcharge, 0)) AS congestion_surcharge_sum,
            SUM(COALESCE(airport_fee, 0)) AS airport_fee_sum,
            SUM(COALESCE(tips, 0)) AS tips_sum,
            SUM(COALESCE(driver_pay, 0)) AS driver_pay_sum,
            SUM(COALESCE(cbd_congestion_fee, 0)) AS cbd_congestion_fee_sum,

            SUM(CASE WHEN shared_request_flag = 'Y' THEN 1 ELSE 0 END) AS shared_request_count,
            SUM(CASE WHEN shared_match_flag = 'Y' THEN 1 ELSE 0 END) AS shared_match_count,
            SUM(CASE WHEN access_a_ride_flag = 'Y' THEN 1 ELSE 0 END) AS access_a_ride_count,
            SUM(CASE WHEN wav_request_flag = 'Y' THEN 1 ELSE 0 END) AS wav_request_count,
            SUM(CASE WHEN wav_match_flag = 'Y' THEN 1 ELSE 0 END) AS wav_match_count,

            SUM(CASE WHEN hvfhs_license_num = 'HV0003' THEN 1 ELSE 0 END) AS uber_trip_count,
            SUM(CASE WHEN hvfhs_license_num = 'HV0005' THEN 1 ELSE 0 END) AS lyft_trip_count,
            SUM(CASE WHEN hvfhs_license_num = 'HV0004' THEN 1 ELSE 0 END) AS via_trip_count,
            SUM(CASE WHEN hvfhs_license_num = 'HV0002' THEN 1 ELSE 0 END) AS juno_trip_count,

            CAST(transit_timestamp AS DATE) AS date,
            EXTRACT(hour FROM transit_timestamp) AS hour,
            strftime(transit_timestamp, '%A') AS day_of_week,
            EXTRACT(dow FROM transit_timestamp) IN (0, 6) AS is_weekend

        FROM read_parquet([{file_list_sql}], union_by_name=true)

        WHERE pickup_datetime >= TIMESTAMP '2024-01-01'
          AND pickup_datetime < TIMESTAMP '2026-04-01'
          AND pickup_datetime IS NOT NULL
          AND dropoff_datetime IS NOT NULL
          AND PULocationID IS NOT NULL
          AND DOLocationID IS NOT NULL
          AND trip_time > 0
          AND trip_time <= 14400
          AND trip_miles >= 0
          AND trip_miles <= 150

        GROUP BY
            transit_timestamp,
            pickup_location_id,
            dropoff_location_id

        ORDER BY
            transit_timestamp,
            pickup_location_id,
            dropoff_location_id
    )
    TO '{str(OUTPUT_FILE).replace("\\", "/")}'
    (FORMAT PARQUET);
    """

    print("Building for-hire master with DuckDB...")
    con.execute(query)

    print("\nDone.")
    print(f"Saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()