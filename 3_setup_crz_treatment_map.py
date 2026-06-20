from pathlib import Path
import pandas as pd

# -----------------------------
# Settings
# -----------------------------

PROCESSED_DIR = Path("data/processed")
MAPPING_DIR = Path("data/mappings")

CRZ_FILE = PROCESSED_DIR / "crz_master_2025_01_2026_03.parquet"
OUTPUT_FILE = MAPPING_DIR / "crz_treatment_map.csv"

# -----------------------------
# Treatment logic
# -----------------------------
# This file creates a first-pass mapping for CRZ detection groups.
# You should review the CSV manually before using it in final models.


def assign_treatment_group(row):
    exposure = assign_exposure_type(row)

    if exposure == "high":
        return "treated"

    if exposure == "excluded_roadway":
        return "exclude"

    if exposure == "review":
        return "review"

    return "spillover"


def assign_entry_type(row):
    group = str(row["detection_group"]).lower()
    region = str(row["detection_region"]).lower()

    if "lincoln" in group or "holland" in group:
        return "new_jersey_crossing"

    if "brooklyn" in region or "brooklyn" in group:
        return "brooklyn_crossing"

    if "queens" in region or "queens" in group or "queensboro" in group:
        return "queens_crossing"

    if "60" in group or "manhattan" in region or "upper manhattan" in region:
        return "sixtieth_st_cordon"

    if "fdr" in group or "west side" in group or "westside" in group:
        return "excluded_roadway_access"

    return "review"


def assign_exposure_type(row):
    group = str(row["detection_group"]).lower()

    if "lincoln" in group or "holland" in group:
        return "high"

    if "queens midtown" in group or "hugh" in group or "carey" in group:
        return "high"

    if "brooklyn bridge" in group or "manhattan bridge" in group or "williamsburg" in group:
        return "high"

    if "queensboro" in group or "60" in group:
        return "high"

    if "fdr" in group or "west side" in group or "westside" in group:
        return "excluded_roadway"

    return "review"


def main():
    MAPPING_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(
        CRZ_FILE,
        columns=[
            "detection_group",
            "detection_region",
            "vehicle_class",
        ],
    )

    crz_map = (
        df[["detection_group", "detection_region"]]
        .drop_duplicates()
        .sort_values(["detection_region", "detection_group"])
        .reset_index(drop=True)
    )

    crz_map["treatment_group"] = crz_map.apply(assign_treatment_group, axis=1)
    crz_map["entry_type"] = crz_map.apply(assign_entry_type, axis=1)
    crz_map["exposure_type"] = crz_map.apply(assign_exposure_type, axis=1)
    crz_map["keep_first_pass"] = True
    crz_map["notes"] = "review label before final analysis"

    crz_map.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Rows: {len(crz_map):,}")

    print("\nTreatment group counts:")
    print(crz_map["treatment_group"].value_counts(dropna=False))

    print("\nEntry type counts:")
    print(crz_map["entry_type"].value_counts(dropna=False))

    print("\nExposure type counts:")
    print(crz_map["exposure_type"].value_counts(dropna=False))

    print("\nPreview:")
    print(crz_map.to_string(index=False))

    print("\nVehicle classes in CRZ file:")
    vehicle_classes = (
        df[["vehicle_class"]]
        .drop_duplicates()
        .sort_values("vehicle_class")
        .reset_index(drop=True)
    )
    print(vehicle_classes.to_string(index=False))


if __name__ == "__main__":
    main()
