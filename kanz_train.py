"""
Kanz Train v1 - Real global deposit training data
Pulls real satellite features for 10 known major iron deposits worldwide
(positive examples) plus 10 nearby background points (negative examples),
builds an honest training dataset, trains a Random Forest, and reports
real precision/recall/AUC on held-out data.
"""

import ee
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import json

print("Kanz Train v1 - connecting to Earth Engine...")
ee.Initialize(project="veryla")

# Real, documented major iron ore deposits (lat, lon)
deposits = [
    ("Carajas_Brazil", -6.00, -50.16),
    ("Mt_Whaleback_Australia", -23.35, 119.73),
    ("Tom_Price_Australia", -22.70, 117.79),
    ("Kiruna_Sweden", 67.85, 20.22),
    ("Mesabi_USA", 47.53, -92.53),
    ("Kryvyi_Rih_Ukraine", 47.91, 33.39),
    ("Sishen_SouthAfrica", -27.79, 22.99),
    ("Bailadila_India", 18.65, 81.23),
    ("Kursk_Russia", 51.73, 36.19),
    ("Gara_Djebilet_Algeria", 26.88, -7.00),
]

# Background (negative) points - offset from each deposit, likely non-ore terrain
import random
random.seed(42)
backgrounds = []
for name, lat, lon in deposits:
    offset_lat = lat + random.uniform(0.5, 0.9) * random.choice([-1, 1])
    offset_lon = lon + random.uniform(0.5, 0.9) * random.choice([-1, 1])
    backgrounds.append((f"{name}_background", offset_lat, offset_lon))

all_points = [(n, la, lo, 1) for n, la, lo in deposits] + [(n, la, lo, 0) for n, la, lo in backgrounds]

def get_features(lat, lon):
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(1000).bounds()

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate("2023-01-01", "2026-06-01")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    )
    image = collection.median()

    red = image.select("B4").divide(10000)
    blue = image.select("B2").divide(10000)
    nir = image.select("B8").divide(10000)
    swir1 = image.select("B11").divide(10000)
    swir2 = image.select("B12").divide(10000)

    iron_oxide = red.divide(blue)
    ferrous = swir2.divide(nir)
    clay = swir1.divide(swir2)

    dem = ee.Image("USGS/SRTMGL1_003").select("elevation")
    slope = ee.Terrain.slope(dem)

    combined = (
        iron_oxide.rename("iron").toFloat()
        .addBands(ferrous.rename("ferrous").toFloat())
        .addBands(clay.rename("clay").toFloat())
        .addBands(dem.rename("elevation").toFloat())
        .addBands(slope.rename("slope").toFloat())
    )

    stats = combined.reduceRegion(reducer=ee.Reducer.mean(), geometry=region, scale=30, maxPixels=1e9)
    return stats.getInfo()

rows = []
for name, lat, lon, label in all_points:
    print(f"Pulling: {name} ({lat}, {lon})...")
    try:
        feats = get_features(lat, lon)
        row = {
            "name": name, "lat": lat, "lon": lon, "label": label,
            "iron": feats.get("iron"), "ferrous": feats.get("ferrous"),
            "clay": feats.get("clay"), "elevation": feats.get("elevation"),
            "slope": feats.get("slope"),
        }
        if None in row.values():
            print(f"  -> Skipped (missing data)")
            continue
        rows.append(row)
        print(f"  -> OK: iron={row['iron']:.2f}")
    except Exception as e:
        print(f"  -> Failed: {e}")

print(f"\nCollected {len(rows)} valid training rows out of {len(all_points)} attempted")

with open("kanz_training_data.json", "w") as f:
    json.dump(rows, f, indent=2)
print("Saved raw data to kanz_training_data.json")

# --- Train the model ---
X = np.array([[r["iron"], r["ferrous"], r["clay"], r["elevation"], r["slope"]] for r in rows])
y = np.array([r["label"] for r in rows])

print("\nFeature matrix shape:", X.shape, "| Positive examples:", y.sum(), "| Negative:", len(y)-y.sum())

if len(rows) < 8:
    print("\nNOT ENOUGH DATA to train/test split meaningfully yet - need more points.")
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n--- HONEST PERFORMANCE ON REAL GLOBAL DATA ---")
    print(classification_report(y_test, y_pred, target_names=["Background", "Ore"]))
    print("AUC:", round(roc_auc_score(y_test, y_proba), 3))

    print("\nFeature importances:")
    for name, imp in zip(["iron","ferrous","clay","elevation","slope"], model.feature_importances_):
        print(f"  {name}: {round(imp,3)}")
