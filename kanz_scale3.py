"""
Kanz Scale Step 3 - 1000 points + proper cross-validation + Gara Djebilet sanity check
"""

import ee
import csv
import random
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

print("Kanz Scale v3 - connecting to Earth Engine...")
ee.Initialize(project="veryla")

positives = []
with open("iron_deposits_selected.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        positives.append((row["name"], float(row["lat"]), float(row["lon"])))

print(f"Loaded {len(positives)} real positive deposit points from previous step")

# We need MORE real deposits to reach 500 positives (we only had 250 before)
# Re-read the full MRDS filtered list and take the NEXT 250 we haven't used yet
import os
extra_positives = []
if os.path.exists("mrds_data/mrds.csv"):
    with open("mrds_data/mrds.csv", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        used_names = set(n for n, la, lo in positives)
        for row in reader:
            commodity = (row.get("commod1") or "").lower()
            if "iron" in commodity:
                try:
                    lat = float(row.get("latitude"))
                    lon = float(row.get("longitude"))
                    name = row.get("dep_name", "unnamed")
                    if -60 < lat < 60 and name not in used_names:
                        extra_positives.append((name, lat, lon))
                except (TypeError, ValueError):
                    continue

random.seed(7)
random.shuffle(extra_positives)
positives_extended = positives + extra_positives[:250]
print(f"Extended positive set: {len(positives_extended)} real deposits")

random.seed(42)
backgrounds = []
for name, lat, lon in positives_extended:
    offset_lat = lat + random.uniform(0.4, 0.8) * random.choice([-1, 1])
    offset_lon = lon + random.uniform(0.4, 0.8) * random.choice([-1, 1])
    offset_lat = max(-59, min(59, offset_lat))
    backgrounds.append((f"{name}_bg", offset_lat, offset_lon))

all_points = [(n, la, lo, 1) for n, la, lo in positives_extended] + [(n, la, lo, 0) for n, la, lo in backgrounds]
print(f"Total points to process: {len(all_points)}")

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
        features = [ee.Feature(ee.Geometry.Point([lo, la]), {"name": n, "label": lb}) for n, la, lo, lb in batch]
        fc = ee.FeatureCollection(features)
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

        print(f"  -> valid so far: {len(all_results)}")
    except Exception as e:
        print(f"  -> Batch failed: {e}")

print(f"\nTotal valid training rows: {len(all_results)}")

with open("kanz_training_data_v3.json", "w") as f:
    json.dump(all_results, f, indent=2)

X = np.array([[r["iron"], r["ferrous"], r["clay"], r["elevation"], r["slope"]] for r in all_results])
y = np.array([r["label"] for r in all_results])
print(f"Feature matrix: {X.shape} | Positives: {y.sum()} | Negatives: {len(y)-y.sum()}")

model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)

# --- REAL cross-validation - 5 different splits, not just one ---
print("\n--- 5-FOLD CROSS-VALIDATION (much more reliable than a single split) ---")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
print("AUC per fold:", [round(s,3) for s in scores])
print("Mean AUC:", round(scores.mean(),3), "| Std deviation:", round(scores.std(),3))
print("(Low std = stable, trustworthy result. High std = unstable, be cautious)")

# --- Final train on everything, one more held-out test for a full report ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
print("\n--- FINAL HELD-OUT TEST REPORT ---")
print(classification_report(y_test, y_pred, target_names=["Background","Ore"]))
print("Test AUC:", round(roc_auc_score(y_test, y_proba),3))

# --- Real sanity check: what does the model say about Gara Djebilet itself? ---
print("\n--- SANITY CHECK: Gara Djebilet (our actual target site) ---")
gara_point = ee.Geometry.Point([-7.00028, 26.88222])
gara_region = gara_point.buffer(50000)
s2_gara = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(gara_region).filterDate("2023-01-01","2026-06-01")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE",30)).median()
)
red = s2_gara.select("B4").divide(10000)
blue = s2_gara.select("B2").divide(10000)
nir = s2_gara.select("B8").divide(10000)
swir1 = s2_gara.select("B11").divide(10000)
swir2 = s2_gara.select("B12").divide(10000)
gara_combined = (
    red.divide(blue).rename("iron").toFloat()
    .addBands(swir2.divide(nir).rename("ferrous").toFloat())
    .addBands(swir1.divide(swir2).rename("clay").toFloat())
    .addBands(dem.rename("elevation").toFloat())
    .addBands(slope.rename("slope").toFloat())
)
gara_stats = gara_combined.reduceRegion(reducer=ee.Reducer.mean(), geometry=gara_point, scale=30, maxPixels=1e9).getInfo()
gara_X = np.array([[gara_stats["iron"], gara_stats["ferrous"], gara_stats["clay"], gara_stats["elevation"], gara_stats["slope"]]])
gara_prob = model.predict_proba(gara_X)[0][1]
print("Gara Djebilet features:", gara_stats)
print(f"Model's predicted 'ore probability' for Gara Djebilet: {round(gara_prob*100,1)}%")
print("(This is a KNOWN real iron deposit - a well-tuned model should score this reasonably high)")

joblib.dump(model, "kanz_model_v3.joblib")
print("\nModel saved as kanz_model_v3.joblib")
