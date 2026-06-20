from pathlib import Path
import pandas as pd

# -----------------------------
# Settings
# -----------------------------

PROCESSED_DIR = Path("data/processed")
MAPPING_DIR = Path("data/mappings")

BRIDGES_FILE = PROCESSED_DIR / "bridges_master_2024_01_2026_03.parquet"
OUTPUT_FILE = MAPPING_DIR / "bridge_treatment_map.csv"


def assign_treatment_group(row):
    facility_id = str(row["facility_id"])
    direction = str(row["direction"]).lower()
    facility = str(row["facility"]).lower()

    # Direct CRZ-relevant facilities
    direct_facilities = {"21", "22", "24", "27", "28"}

    # Spillover / comparison facilities
    spillover_facilities = {"23", "29", "30"}

    # Usually less relevant for Manhattan CRZ analysis
    lower_priority_facilities = {"25", "26"}

    if facility_id in direct_facilities:
        # Treat Manhattan-bound directions as treated
        if "manhattan" in direction:
            return "treated"
        # Henry Hudson may not always say Manhattan clearly; southbound is likely toward Manhattan core
        if facility_id == "24" and "south" in direction:
            return "treated"
        return "control"

    if facility_id in spillover_facilities:
        return "spillover"

    if facility_id in lower_priority_facilities:
        return "exclude"

    return "review"


def assign_exposure_type(row):
    facility_id = str(row["facility_id"])

    if facility_id in {"27", "28"}:
        return "high"
    if facility_id in {"21", "22", "24"}:
        return "medium"
    if facility_id in {"23", "29", "30"}:
        return "spillover"
    return "low"


def main():
    MAPPING_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(
        BRIDGES_FILE,
        columns=[
            "facility_id",
            "facility",
            "direction",
        ],
    )

    bridge_map = (
        df.drop_duplicates()
        .sort_values(["facility_id", "facility", "direction"])
        .reset_index(drop=True)
    )

    bridge_map["treatment_group"] = bridge_map.apply(assign_treatment_group, axis=1)
    bridge_map["exposure_type"] = bridge_map.apply(assign_exposure_type, axis=1)
    bridge_map["keep_first_pass"] = bridge_map["treatment_group"].isin(
        ["treated", "control", "spillover"]
    )
    bridge_map["notes"] = "review label before final analysis"

    bridge_map.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Rows: {len(bridge_map):,}")
    print("\nTreatment group counts:")
    print(bridge_map["treatment_group"].value_counts(dropna=False))
    print("\nPreview:")
    print(bridge_map.to_string(index=False))

if __name__ == "__main__":
    main()
