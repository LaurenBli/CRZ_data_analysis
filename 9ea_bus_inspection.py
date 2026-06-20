# 9da_bus_inspection.py

import pandas as pd

path = r"data/processed/bus_master_2024_01_2026_03.parquet"

df = pd.read_parquet(path)

print("=" * 80)
print("SHAPE")
print(df.shape)

print("\n" + "=" * 80)
print("COLUMNS")
for c in df.columns:
    print(c)

print("\n" + "=" * 80)
print("HEAD")
print(df.head())

print("\n" + "=" * 80)
print("DATE RANGE CHECK")

date_candidates = [
    c for c in df.columns
    if "date" in c.lower()
    or "time" in c.lower()
    or "transit" in c.lower()
]

print("Possible datetime columns:", date_candidates)

for c in date_candidates:
    try:
        x = pd.to_datetime(df[c], errors="coerce")
        print(f"\n{c}")
        print("min:", x.min())
        print("max:", x.max())
        print("missing:", x.isna().sum())
    except Exception as e:
        print(f"{c}: skipped because {e}")

print("\n" + "=" * 80)
print("ROUTE / GEO COLUMNS CHECK")

geo_candidates = [
    c for c in df.columns
    if "route" in c.lower()
    or "stop" in c.lower()
    or "borough" in c.lower()
    or "lat" in c.lower()
    or "lon" in c.lower()
    or "lng" in c.lower()
    or "direction" in c.lower()
    or "agency" in c.lower()
]

print(geo_candidates)

print("\n" + "=" * 80)
print("NUMERIC SUMMARY")
print(df.describe(include="all").T)

print("\n" + "=" * 80)
print("UNIQUE COUNTS FOR ROUTE/GEO CANDIDATES")

for c in geo_candidates:
    try:
        print(f"{c}: {df[c].nunique(dropna=True):,} unique")
    except:
        pass

print("\n" + "=" * 80)
print("TOP VALUES FOR ROUTE/GEO CANDIDATES")

for c in geo_candidates[:30]:
    print("\n" + "-" * 80)
    print(c)
    print(df[c].value_counts(dropna=False).head(20))