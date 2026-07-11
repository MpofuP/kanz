"""
Kanz Backend - Flask API server (production-ready)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import ee
import joblib
import numpy as np
import os
import json

app = Flask(__name__)
CORS(app)

print("Starting Kanz backend - authenticating with Earth Engine...")

service_account_json = os.environ.get("EE_SERVICE_ACCOUNT_JSON")
if service_account_json:
    key_data = json.loads(service_account_json)
    credentials = ee.ServiceAccountCredentials(
        email=key_data["client_email"],
        key_data=service_account_json
    )
else:
    credentials = ee.ServiceAccountCredentials(email=None, key_file="kanz-service-account.json")

ee.Initialize(credentials)
print("Earth Engine ready.")

print("Loading trained model...")
model = joblib.load("kanz_model_global.joblib")
FEATURE_NAMES = ["iron", "ferrous", "clay", "ndvi", "carbonate", "elevation", "slope"]
print("Model loaded.")

dem = ee.Image("USGS/SRTMGL1_003").select("elevation")
slope_img = ee.Terrain.slope(dem)

def get_features(lat, lon):
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(5000)

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
        .addBands(nir.subtract(red).divide(nir.add(red)).rename("ndvi").toFloat())
        .addBands(swir1.divide(nir).rename("carbonate").toFloat())
        .addBands(dem.rename("elevation").toFloat())
        .addBands(slope_img.rename("slope").toFloat())
    )

    stats = combined.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=30, maxPixels=1e9)
    return stats.getInfo()

@app.route("/predict", methods=["GET"])
def predict():
    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lon"))
    except (TypeError, ValueError):
        return jsonify({"error": "Missing or invalid lat/lon parameters"}), 400

    try:
        features = get_features(lat, lon)

        if any(features.get(k) is None for k in FEATURE_NAMES):
            return jsonify({"error": "No valid satellite data for this location"}), 422

        X = np.array([[features[k] for k in FEATURE_NAMES]])
        probability = float(model.predict_proba(X)[0][1])

        return jsonify({
            "lat": lat, "lon": lon,
            "probability": round(probability, 4),
            "confidence_note": "Model AUC ~0.61-0.62 on held-out global test data - moderate, not high confidence",
            "features": {k: round(features[k], 4) for k in FEATURE_NAMES}
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
