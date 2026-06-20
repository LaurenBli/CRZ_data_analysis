
import os
import pandas as pd

# ---------------------------------------------------------------------
# 9db Build taxi CRZ treatment groups
#
# Purpose:
#   Map TLC pickup/dropoff LocationIDs into:
#     - core_crz
#     - border_crz
#     - outside_crz
#
# Input:
#   data/processed/taxi_master_2024_01_2026_03.parquet
#
# Output:
#   data/processed/taxi_master_with_crz_groups.parquet
#   data/mappings/taxi_crz_location_mapping.csv
# ---------------------------------------------------------------------

INPUT = "data/processed/taxi_master_2024_01_2026_03.parquet"

OUT_TAXI = "data/processed/taxi_master_with_crz_groups.parquet"
OUT_MAPPING = "data/mappings/taxi_crz_location_mapping.csv"

os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/mappings", exist_ok=True)

# ---------------------------------------------------------------------
# Manual CRZ classification from TLC taxi zone LocationIDs
# ---------------------------------------------------------------------

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


def classify_crz(location_id):
    if pd.isna(location_id):
        return "missing"

    location_id = int(location_id)

    if location_id in CORE_CRZ_IDS:
        return "core_crz"

    if location_id in BORDER_CRZ_IDS:
        return "border_crz"

    return "outside_crz"


def make_flow_type(row):
    pu = row["pickup_crz_group"]
    do = row["dropoff_crz_group"]

    if pu == "missing" or do == "missing":
        return "missing"

    return f"{pu}_to_{do}"


def make_simple_flow_type(row):
    pu_core = row["pickup_in_core_crz"] == 1
    do_core = row["dropoff_in_core_crz"] == 1

    pu_border = row["pickup_in_border_crz"] == 1
    do_border = row["dropoff_in_border_crz"] == 1

    if pu_border or do_border:
        return "border_related"

    if pu_core and do_core:
        return "in_to_in"

    if (not pu_core) and do_core:
        return "out_to_in"

    if pu_core and (not do_core):
        return "in_to_out"

    return "out_to_out"


# ---------------------------------------------------------------------
# Load taxi master
# ---------------------------------------------------------------------

df = pd.read_parquet(INPUT)

required = {
    "transit_timestamp",
    "pickup_location_id",
    "dropoff_location_id",
    "trip_count",
}

missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["pickup_location_id"] = pd.to_numeric(
    df["pickup_location_id"],
    errors="coerce",
)

df["dropoff_location_id"] = pd.to_numeric(
    df["dropoff_location_id"],
    errors="coerce",
)

# ---------------------------------------------------------------------
# Add CRZ group classifications
# ---------------------------------------------------------------------

df["pickup_crz_group"] = df["pickup_location_id"].apply(classify_crz)
df["dropoff_crz_group"] = df["dropoff_location_id"].apply(classify_crz)

df["pickup_in_core_crz"] = (
    df["pickup_crz_group"] == "core_crz"
).astype(int)

df["dropoff_in_core_crz"] = (
    df["dropoff_crz_group"] == "core_crz"
).astype(int)

df["pickup_in_border_crz"] = (
    df["pickup_crz_group"] == "border_crz"
).astype(int)

df["dropoff_in_border_crz"] = (
    df["dropoff_crz_group"] == "border_crz"
).astype(int)

df["pickup_outside_crz"] = (
    df["pickup_crz_group"] == "outside_crz"
).astype(int)

df["dropoff_outside_crz"] = (
    df["dropoff_crz_group"] == "outside_crz"
).astype(int)

# Main DiD treatment definition:
# treated = pickup starts in core CRZ.
# Border pickup zones are excluded from the clean main DiD sample.
df["treated_group"] = df["pickup_in_core_crz"]

df["main_did_sample"] = (
    df["pickup_crz_group"] != "border_crz"
).astype(int)

# ---------------------------------------------------------------------
# OD flow type
# ---------------------------------------------------------------------

df["crz_flow_type"] = df.apply(
    make_flow_type,
    axis=1,
)

df["simple_crz_flow_type"] = df.apply(
    make_simple_flow_type,
    axis=1,
)

# ---------------------------------------------------------------------
# Build location-level mapping output
# ---------------------------------------------------------------------

all_location_ids = sorted(
    set(df["pickup_location_id"].dropna().astype(int).unique())
    | set(df["dropoff_location_id"].dropna().astype(int).unique())
)

mapping = pd.DataFrame({
    "location_id": all_location_ids,
})

mapping["crz_group"] = mapping["location_id"].apply(classify_crz)

mapping["in_core_crz"] = (
    mapping["crz_group"] == "core_crz"
).astype(int)

mapping["in_border_crz"] = (
    mapping["crz_group"] == "border_crz"
).astype(int)

mapping["outside_crz"] = (
    mapping["crz_group"] == "outside_crz"
).astype(int)

# ---------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------

df.to_parquet(
    OUT_TAXI,
    index=False,
)

mapping.to_csv(
    OUT_MAPPING,
    index=False,
)

# ---------------------------------------------------------------------
# Print checks
# ---------------------------------------------------------------------

print("=" * 90)
print("9db taxi CRZ treatment groups complete")
print("=" * 90)
print(f"Saved taxi file to:    {OUT_TAXI}")
print(f"Saved mapping file to: {OUT_MAPPING}")
print()

print("Rows:")
print(f"{len(df):,}")
print()

print("Pickup CRZ group counts:")
print(df["pickup_crz_group"].value_counts(dropna=False))
print()

print("Dropoff CRZ group counts:")
print(df["dropoff_crz_group"].value_counts(dropna=False))
print()

print("Main DiD sample counts:")
print(df["main_did_sample"].value_counts(dropna=False))
print()

print("Treated group counts inside main DiD sample:")
print(
    df[df["main_did_sample"] == 1]["treated_group"]
    .value_counts(dropna=False)
)
print()

print("Simple CRZ flow type counts:")
print(df["simple_crz_flow_type"].value_counts(dropna=False))
print()

print("Mapping counts:")
print(mapping["crz_group"].value_counts(dropna=False))
print()

print("Core CRZ IDs:")
print(sorted(CORE_CRZ_IDS))
print()

print("Border CRZ IDs:")
print(sorted(BORDER_CRZ_IDS))
