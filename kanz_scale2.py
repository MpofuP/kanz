"""
Kanz Scale Step 2 - Batch feature extraction + training on 500 real points
(250 real iron deposits + 250 matched background points)
"""

import ee
import csv
import random
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

print("Kanz Scale v2 - connecting to Earth Engine...")
ee.Initialize(project="veryla")

# Load the real deposits we just downloaded
positives = []
with open("iron_deposits_selected.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        positives.append((row["name"], float(row["lat"]), float(row["lon"])))

print(f"Loaded {len(positives)} real positive deposit points")

# Generate matched background (negative) points
random.seed(42)
backgrounds = []
for name, lat, lon in positives:
    offset_lat = lat + random.uniform(0.4, 0.8) * random.choice([-1, 1])
    offset_lon = lon + random.uniform(0.4, 0.8) * random.choice([-1, 1])
    offset_lat = max(-59, min(59, offset_lat))
    backgrounds.append((f"{name}_bg", offset_lat, offset_lon))

all_points = [(n, la, lo, 1) for n, la, lo in positives] + [(n, la, lo, 0) for n, la, lo in backgrounds]
print(f"Total points to process: {len(all_points)}")

def get_features_batch(batch):
    features = []
    for name, lat, lon, label in batch:
        geom = ee.Geometry.Point([lon, lat])
        features.append(ee.Feature(geom, {"name": name, "label": label}))
    return ee.FeatureCollection(features)

def build_image():
    def per_point_image(region):
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region)
            .filterDate("2023-01-01", "2026-06-01")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 25))
        )
        return collection.median()
    return per_point_image

dem = ee.Image("USGS/SRTMGL1_003").select("elevation")
slope = ee.Terrain.slope(dem)

batch_size = 25
all_results = []
total_batches = (len(all_points) + batch_size - 1) // batch_size

for i in range(0, len(all_points), batch_size):
    batch = all_points[i:i+batch_size]
    batch_num = i // batch_size + 1
    print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} points)...")

    try:
        fc = get_features_batch(batch)
        region = fc.geometry().bounds().buffer(50000)

        s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region)
            .filterDate("2023-01-01", "2026-06-01")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
            .median()
        )

        red = s2.select("B4").divide(10000)
        blue = s2.select("B2").divide(10000)
        nir = s2.select("B8").divide(10000)
        swir1 = s2.select("B11").divide(10000)
        swir2 = s2.select("B12").divide(10000)

        combined = (
            red.divide(blue).rename("iron").toFloat()
            .addBands(swir2.divide(nir).rename("ferrous").toFloat())
            .addBands(swir1.divide(swir2).rename("clay").toFloat())
            .addBands(dem.rename("elevation").toFloat())
            .addBands(slope.rename("slope").toFloat())
        )

        sampled = combined.reduceRegions(collection=fc, reducer=ee.Reducer.first(), scale=30)
        result = sampled.getInfo()

        for feat in result["features"]:
            props = feat["properties"]
            if all(k in props and props[k] is not None for k in ["iron","ferrous","clay","elevation","slope"]):
                all_results.append(props)

        print(f"  -> {len(result['features'])} points processed, valid so far: {len(all_results)}")

    except Exception as e:
        print(f"  -> Batch failed: {e}")

print(f"\nTotal valid training rows collected: {len(all_results)}")

with open("kanz_training_data_v2.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("Saved to kanz_training_data_v2.json")

X = np.array([[r["iron"], r["ferrous"], r["clay"], r["elevation"], r["slope"]] for r in all_results])
y = np.array([r["label"] for r in all_results])

print(f"\nFeature matrix: {X.shape} | Positives: {y.sum()} | Negatives: {len(y)-y.sum()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n--- HONEST PERFORMANCE (real global data, held-out test set) ---")
print(classification_report(y_test, y_pred, target_names=["Background", "Ore"]))
print("AUC:", round(roc_auc_score(y_test, y_proba), 3))
print("Test set size:", len(y_test))

print("\nFeature importances:")
for name, imp in zip(["iron","ferrous","clay","elevation","slope"], model.feature_importances_):
    print(f"  {name}: {round(imp,3)}")

joblib.dump(model, "kanz_model_v2.joblib")
print("\nModel saved as kanz_model_v2.joblib")
