import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# ============================================================
# MAIN TAXI PANEL INSPECTION
# ============================================================

path = r"data/processed/taxi_master_2024_01_2026_03.parquet"

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
    or "pickup" in c.lower()
    or "dropoff" in c.lower()
]

print("Possible datetime columns:", date_candidates)

for c in date_candidates[:10]:
    try:
        x = pd.to_datetime(df[c], errors="coerce")
        print(f"\n{c}")
        print("min:", x.min())
        print("max:", x.max())
    except:
        pass

print("\n" + "=" * 80)
print("ZONE COLUMNS CHECK")

zone_candidates = [
    c for c in df.columns
    if "location" in c.lower()
    or "zone" in c.lower()
    or "borough" in c.lower()
    or "pickup" in c.lower()
    or "dropoff" in c.lower()
]

print(zone_candidates)


# ============================================================
# TAXI ZONE SHAPEFILE MAPPING
# ============================================================

# path to TLC taxi zone shapefile
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


# ------------------------------------------------------------
# Create centroid points for each taxi zone
# ------------------------------------------------------------

gdf["centroid"] = gdf.geometry.centroid
gdf["lon"] = gdf["centroid"].x
gdf["lat"] = gdf["centroid"].y

print("\nCentroid preview:")
print(
    gdf[
        ["LocationID", "zone", "borough", "lat", "lon"]
    ].head(20)
)


# ------------------------------------------------------------
# Plot all taxi zones as dots
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 12))

gdf.plot(
    ax=ax,
    alpha=0.2,
    edgecolor="gray"
)

ax.scatter(
    gdf["lon"],
    gdf["lat"],
    s=20
)

for _, row in gdf.iterrows():
    ax.text(
        row["lon"],
        row["lat"],
        str(row["LocationID"]),
        fontsize=6
    )

plt.title("NYC TLC Taxi Zone Centroids")
plt.tight_layout()
plt.show()