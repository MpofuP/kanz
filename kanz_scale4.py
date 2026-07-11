"""
Kanz Scale Step 4 - Fixed dedup + desert-focused subset + 2 more features
Compares a GLOBAL model vs a DESERT-LATITUDE model to see which
generalizes better to Gara Djebilet specifically.
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

print("Kanz Scale v4 - connecting to Earth Engine...")
ee.Initialize(project="veryla")

# --- Re-read MRDS fresh, dedupe by COORDINATES this time, not name ---
print("Reading full MRDS data fresh...")
all_iron = []
seen_coords = set()
with open("mrds_data/mrds.csv", encoding="utf-8", errors="ignore") as f:
    reader = csv.DictReader(f)
    for row in reader:
        commodity = (row.get("commod1") or "").lower()
        if "iron" in commodity:
            try:
                lat = float(row.get("latitude"))
                lon = float(row.get("longitude"))
                if not (-60 < lat < 60):
                    continue
                # Dedupe key: round to ~1km precision, catches true duplicates
                key = (round(lat, 2), round(lon, 2))
                if key in seen_coords:
                    continue
                seen_coords.add(key)
                name = row.get("dep_name") or f"deposit_{len(all_iron)}"
                all_iron.append((name, lat, lon))
            except (TypeError, ValueError):
                continue

print(f"Total unique real iron deposit locations found: {len(all_iron)}")

random.seed(42)
random.shuffle(all_iron)
positives_global = all_iron[:500]
print(f"Selected {len(positives_global)} for the GLOBAL experiment")

# Desert-latitude subset: roughly 15-40 degrees N or S, where most deserts sit
desert_candidates = [d for d in all_iron if 15 <= abs(d[1]) <= 40]
random.shuffle(desert_candidates)
positives_desert = desert_candidates[:300]
print(f"Found {len(desert_candidates)} deposits in desert-latitude bands, using {len(positives_desert)}")

def make_backgrounds(positives, seed):
    random.seed(seed)
    bgs = []
    for name, lat, lon in positives:
        offset_lat = lat + random.uniform(0.4, 0.8) * random.choice([-1, 1])
        offset_lon = lon + random.uniform(0.4, 0.8) * random.choice([-1, 1])
        offset_lat = max(-59, min(59, offset_lat))
        bgs.append((f"{name}_bg", offset_lat, offset_lon))
    return bgs

def extract_features(point_list, dem, slope):
    all_points = point_list
    batch_size = 25
    results = []
    total_batches = (len(all_points) + batch_size - 1) // batch_size
    for i in range(0, len(all_points), batch_size):
        batch = all_points[i:i+batch_size]
        batch_num = i // batch_size + 1
        print(f"  Batch {batch_num}/{total_batches}...")
        try:
            features = [ee.Feature(ee.Geometry.Point([lo, la]), {"name": n, "label": lb}) for n, la, lo, lb in batch]
            fc = ee.FeatureCollection(features)
            region = fc.geometry().bounds().buffer(50000)

            s2 = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(region).filterDate("2023-01-01", "2026-06-01")
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30)).median()
            )
            red = s2.select("B4").divide(10000)
            blue = s2.select("B2").divide(10000)
            green = s2.select("B3").divide(10000)
            nir = s2.select("B8").divide(10000)
            swir1 = s2.select("B11").divide(10000)
            swir2 = s2.select("B12").divide(10000)

            combined = (
                red.divide(blue).rename("iron").toFloat()
                .addBands(swir2.divide(nir).rename("ferrous").toFloat())
                .addBands(swir1.divide(swir2).rename("clay").toFloat())
                .addBands(nir.subtract(red).divide(nir.add(red)).rename("ndvi").toFloat())
                .addBands(swir1.divide(nir).rename("carbonate").toFloat())
                .addBands(dem.rename("elevation").toFloat())
                .addBands(slope.rename("slope").toFloat())
            )

            sampled = combined.reduceRegions(collection=fc, reducer=ee.Reducer.first(), scale=30)
            result = sampled.getInfo()
            need = ["iron","ferrous","clay","ndvi","carbonate","elevation","slope"]
            for feat in result["features"]:
                props = feat["properties"]
                if all(k in props and props[k] is not None for k in need):
                    results.append(props)
            print(f"    valid so far: {len(results)}")
        except Exception as e:
            print(f"    Batch failed: {e}")
    return results

dem = ee.Image("USGS/SRTMGL1_003").select("elevation")
slope = ee.Terrain.slope(dem)

def run_experiment(name, positives):
    print(f"\n=== Experiment: {name} ({len(positives)} positive points) ===")
    backgrounds = make_backgrounds(positives, seed=42)
    all_points = [(n, la, lo, 1) for n, la, lo in positives] + [(n, la, lo, 0) for n, la, lo in backgrounds]
    results = extract_features(all_points, dem, slope)
    print(f"Total valid rows for {name}: {len(results)}")

    feat_names = ["iron","ferrous","clay","ndvi","carbonate","elevation","slope"]
    X = np.array([[r[k] for k in feat_names] for r in results])
    y = np.array([r["label"] for r in results])
    print(f"Feature matrix: {X.shape} | Positives: {y.sum()} | Negatives: {len(y)-y.sum()}")

    model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    print(f"AUC per fold: {[round(s,3) for s in scores]}")
    print(f"Mean AUC: {round(scores.mean(),3)} | Std: {round(scores.std(),3)}")

    model.fit(X, y)
    with open(f"kanz_training_{name}.json", "w") as f:
        json.dump(results, f)
    joblib.dump(model, f"kanz_model_{name}.joblib")

    for fname, imp in zip(feat_names, model.feature_importances_):
        print(f"  {fname}: {round(imp,3)}")

    return model, feat_names

model_global, feats_global = run_experiment("global", positives_global)
model_desert, feats_desert = run_experiment("desert", positives_desert)

# --- Sanity check both models on Gara Djebilet ---
print("\n=== SANITY CHECK: Gara Djebilet, both models ===")
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
    .addBands(nir.subtract(red).divide(nir.add(red)).rename("ndvi").toFloat())
    .addBands(swir1.divide(nir).rename("carbonate").toFloat())
    .addBands(dem.rename("elevation").toFloat())
    .addBands(slope.rename("slope").toFloat())
)
gara_stats = gara_combined.reduceRegion(reducer=ee.Reducer.mean(), geometry=gara_point, scale=30, maxPixels=1e9).getInfo()
gara_X = np.array([[gara_stats[k] for k in feats_global]])

prob_global = model_global.predict_proba(gara_X)[0][1]
prob_desert = model_desert.predict_proba(gara_X)[0][1]
print(f"Global model probability for Gara Djebilet: {round(prob_global*100,1)}%")
print(f"Desert-focused model probability for Gara Djebilet: {round(prob_desert*100,1)}%")
print("(If desert model scores meaningfully higher, narrowing scope genuinely helped)")
