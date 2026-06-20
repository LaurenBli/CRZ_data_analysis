# 9eb_build_bus_crz_mapping.py

from pathlib import Path
import requests
import pandas as pd

BUS_MASTER = Path("data/processed/bus_master_2024_01_2026_03.parquet")

DATASET_ID = "2ucp-7wg5"  # MTA Bus Stops
BASE_URL = f"https://data.ny.gov/resource/{DATASET_ID}.json"

OUT_BUS = Path("data/processed/bus_master_with_crz_groups.parquet")
OUT_MAPPING = Path("data/mappings/bus_crz_route_mapping.csv")

APP_TOKEN = None
LIMIT = 3_500_000

OUT_BUS.parent.mkdir(parents=True, exist_ok=True)
OUT_MAPPING.parent.mkdir(parents=True, exist_ok=True)


def get_headers():
    headers = {}
    if APP_TOKEN:
        headers["X-App-Token"] = APP_TOKEN
    return headers


def request_json(params):
    response = requests.get(
        BASE_URL,
        params=params,
        headers=get_headers(),
        timeout=300,
    )

    if not response.ok:
        print("REQUEST FAILED")
        print("URL:", response.url)
        print("Status:", response.status_code)
        print("Response text:")
        print(response.text[:2000])

    response.raise_for_status()
    return response.json()


def normalize_bool_series(s):
    return (
        s.astype(str)
        .str.strip()
        .str.upper()
        .isin(["TRUE", "1", "YES", "Y"])
    )


print("=" * 90)
print("Inspecting MTA Bus Stops reference file")
print("=" * 90)

sample = pd.DataFrame.from_records(request_json({"$limit": 5}))

print("Sample shape:")
print(sample.shape)
print()
print("Columns:")
print(list(sample.columns))
print()
print("Head:")
print(sample.head())
print()

cols = list(sample.columns)

route_col = "route_id" if "route_id" in cols else "route_short_name"
stop_col = "stop_id" if "stop_id" in cols else None

cbd_candidates = [c for c in cols if "cbd" in c.lower()]
if not cbd_candidates:
    raise ValueError(
        "Could not find a CBD column in the Bus Stops API. "
        f"Available columns: {cols}"
    )

cbd_col = cbd_candidates[0]

optional_cols = []
for c in ["route_short_name", "route_long_name", "revenue_stop", "boarding"]:
    if c in cols:
        optional_cols.append(c)

select_cols = [route_col]
if stop_col:
    select_cols.append(stop_col)

select_cols.append(cbd_col)
select_cols.extend(optional_cols)
select_cols = list(dict.fromkeys(select_cols))

print("=" * 90)
print("Using columns")
print("=" * 90)
print("route_col:", route_col)
print("stop_col:", stop_col)
print("cbd_col:", cbd_col)
print("selected:", select_cols)
print()

print("=" * 90)
print("Loading MTA Bus Stops reference file")
print("=" * 90)

params = {
    "$select": ", ".join(select_cols),
    "$limit": LIMIT,
}

stops = pd.DataFrame.from_records(request_json(params))

print("Pulled stops shape:")
print(stops.shape)
print()

if len(stops) >= LIMIT:
    print(f"WARNING: hit LIMIT={LIMIT:,}. Route mapping may be incomplete.")
    print("If this happens, increase LIMIT or add pagination.")
    print()

stops[route_col] = stops[route_col].astype(str).str.strip().str.upper()
stops[cbd_col] = stops[cbd_col].astype(str).str.strip().str.upper()

if "revenue_stop" in stops.columns:
    stops = stops[normalize_bool_series(stops["revenue_stop"])].copy()

stops["is_cbd_stop"] = normalize_bool_series(stops[cbd_col]).astype(int)

agg_dict = {
    "any_cbd_stop": ("is_cbd_stop", "max"),
    "share_cbd_stops": ("is_cbd_stop", "mean"),
}

if stop_col is not None:
    agg_dict["n_unique_stops"] = (stop_col, "nunique")
else:
    agg_dict["n_unique_stops"] = (route_col, "size")

if "route_long_name" in stops.columns:
    agg_dict["example_route_name"] = ("route_long_name", "first")

route_map = (
    stops.groupby(route_col, as_index=False)
    .agg(**agg_dict)
    .rename(columns={route_col: "route_id"})
)

route_map["treated_group"] = route_map["any_cbd_stop"].astype(int)

route_map["bus_route_group"] = route_map["treated_group"].map(
    {
        1: "core_crz",
        0: "outside_crz",
    }
)

route_map["main_did_sample"] = 1

print("=" * 90)
print("Loading bus master")
print("=" * 90)

bus = pd.read_parquet(BUS_MASTER)

required = {
    "transit_timestamp",
    "bus_route",
    "ridership",
    "transfers",
}

missing = required - set(bus.columns)
if missing:
    raise ValueError(f"Missing required columns in bus master: {missing}")

bus["bus_route"] = bus["bus_route"].astype(str).str.strip().str.upper()

bus = bus.merge(
    route_map[
        [
            "route_id",
            "treated_group",
            "bus_route_group",
            "main_did_sample",
            "n_unique_stops",
            "share_cbd_stops",
        ]
    ],
    left_on="bus_route",
    right_on="route_id",
    how="left",
)

bus["treated_group"] = bus["treated_group"].fillna(0).astype(int)
bus["bus_route_group"] = bus["bus_route_group"].fillna("outside_crz")
bus["main_did_sample"] = bus["main_did_sample"].fillna(1).astype(int)
bus["n_unique_stops"] = bus["n_unique_stops"].fillna(0)
bus["share_cbd_stops"] = bus["share_cbd_stops"].fillna(0)

MANUAL_TREATED_ROUTES = {
    "SIM5X", "SIM6X",
    "X1", "X2", "X3", "X4", "X5", "X7", "X8", "X9",
    "X10", "X10B", "X11", "X12", "X14", "X15", "X17",
    "X17A", "X17J", "X19", "X21", "X22", "X22A",
    "X30", "X31", "X42",
}

manual_mask = bus["bus_route"].isin(MANUAL_TREATED_ROUTES)

bus.loc[manual_mask, "treated_group"] = 1
bus.loc[manual_mask, "bus_route_group"] = "core_crz"

mapping = (
    bus[
        [
            "bus_route",
            "treated_group",
            "bus_route_group",
            "n_unique_stops",
            "share_cbd_stops",
        ]
    ]
    .drop_duplicates()
    .rename(columns={"bus_route": "route_id"})
    .copy()
)

bus.to_parquet(OUT_BUS, index=False)
mapping.to_csv(OUT_MAPPING, index=False)

print("=" * 90)
print("9eb Bus CRZ treatment groups complete")
print("=" * 90)
print(f"Saved bus file to:    {OUT_BUS}")
print(f"Saved mapping file to: {OUT_MAPPING}")
print()

print("Rows:")
print(f"{len(bus):,}")
print()

print("Treated group counts:")
print(bus["treated_group"].value_counts(dropna=False))
print()

print("Bus route groups:")
print(bus["bus_route_group"].value_counts(dropna=False))
print()

print("Unique bus routes in master:")
print(bus["bus_route"].nunique())
print()

print("Mapping group counts:")
print(mapping["bus_route_group"].value_counts(dropna=False))
print()

print("Manual treated route rows:")
print(f"{manual_mask.sum():,}")
print()

unmatched = (
    bus[bus["route_id"].isna()][["bus_route"]]
    .drop_duplicates()
    .sort_values("bus_route")
)

print("Unmatched bus routes:")
print(unmatched.head(50))
print(f"Total unmatched routes: {len(unmatched):,}")
