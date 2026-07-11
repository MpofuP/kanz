"""
Test if the service account can authenticate with Earth Engine
"""
import ee

try:
    credentials = ee.ServiceAccountCredentials(
        email=None,
        key_file="kanz-service-account.json"
    )
    ee.Initialize(credentials)
    print("SUCCESS - service account authenticated with Earth Engine")

    # Try an actual real query to confirm full access, not just login
    point = ee.Geometry.Point([-7.00028, 26.88222])
    image = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(point).first()
    print("Test query result:", image.date().format().getInfo())
except Exception as e:
    print("FAILED:", e)
