"""
Kanz Scale Step 1b - Download real global iron deposits (with proper headers)
"""

import urllib.request
import zipfile
import csv
import os
import random

print("Downloading USGS MRDS database...")
url = "https://mrdata.usgs.gov/mrds/mrds-csv.zip"

req = urllib.request.Request(url, headers={
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
})

with urllib.request.urlopen(req) as response, open("mrds-csv.zip", "wb") as out_file:
    out_file.write(response.read())

print("Downloaded. Extracting...")

with zipfile.ZipFile("mrds-csv.zip", "r") as z:
    z.extractall("mrds_data")

print("Extracted. Searching for the main CSV file...")
csv_file = None
for f in os.listdir("mrds_data"):
    if f.lower().endswith(".csv"):
        csv_file = os.path.join("mrds_data", f)
        break

print("Reading:", csv_file)

iron_deposits = []
with open(csv_file, encoding="utf-8", errors="ignore") as f:
    reader = csv.DictReader(f)
    for row in reader:
        commodity = (row.get("commod1") or "").lower()
        if "iron" in commodity:
            try:
                lat = float(row.get("latitude"))
                lon = float(row.get("longitude"))
                if -60 < lat < 60:
                    iron_deposits.append({
                        "name": row.get("dep_name", "unnamed"),
                        "lat": lat, "lon": lon
                    })
            except (TypeError, ValueError):
                continue

print(f"Found {len(iron_deposits)} valid iron deposit records worldwide")

random.seed(42)
random.shuffle(iron_deposits)
selected = iron_deposits[:250]

with open("iron_deposits_selected.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["name", "lat", "lon"])
    writer.writeheader()
    writer.writerows(selected)

print(f"Saved {len(selected)} selected deposits to iron_deposits_selected.csv")
