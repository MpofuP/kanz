"""
Kanz - Export real data as JSON for the interactive app
"""

import ee
import numpy as np
import json

print("Exporting real Kanz data for the app...")

ee.Initialize(project="veryla")

region = ee.Geometry.Rectangle([-7.05, 26.83, -6.39, 26.93])

collection = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(region)
    .filterDate("2024-01-01", "2026-06-01")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 15))
)
image = collection.median()

red = image.select("B4").divide(10000)
blue = image.select("B2").divide(10000)
nir = image.select("B8").divide(10000)
swir1 = image.select("B11").divide(10000)
swir2 = image.select("B12").divide(10000)

iron_oxide_index = red.divide(blue).rename("iron_oxide").toFloat()
ferrous_index = swir2.divide(nir).rename("ferrous_index").toFloat()
clay_index = swir1.divide(swir2).rename("clay_index").toFloat()

combined = iron_oxide_index.addBands(ferrous_index).addBands(clay_index).reproject(crs="EPSG:4326", scale=150)
sample = combined.sampleRectangle(region=region, defaultValue=-999)

iron_array = np.array(sample.get("iron_oxide").getInfo())
ferrous_array = np.array(sample.get("ferrous_index").getInfo())
clay_array = np.array(sample.get("clay_index").getInfo())

# Downsample a bit more for smooth browser performance, and export as JSON
data = {
    "shape": list(iron_array.shape),
    "iron": np.round(iron_array, 3).tolist(),
    "ferrous": np.round(ferrous_array, 3).tolist(),
    "clay": np.round(clay_array, 3).tolist(),
}

with open("kanz_data.json", "w") as f:
    json.dump(data, f)

print("Exported kanz_data.json - shape:", iron_array.shape)
