#!/usr/bin/env python3
"""
Script to fetch USGS Delaware River gauge height data for topology inference testing.
This script downloads the data but does NOT save it to the repository (to avoid large files).
Instead, it demonstrates how to fetch the data and shows that the test data loading works.
The actual test data should be obtained by running this script separately and committing
only the small metadata files, not the large CSV.
"""

import requests
import datetime
import sys
import pandas as pd
import matplotlib.pyplot as plt
import os

# USGS sites for Delaware River basin
sites = [
    "01417500",  # E. Br. Delaware at Harvard
    "01426500",  # W. Br. Delaware at Hale Eddy
    "01427207",  # Delaware at Lordville
    "01427510",  # Delaware at Callicoon
    # "01436000",  # Neversink at Neversink
    "01437100",  # Gumaer Brook Near Wurtsboro
    # "01436500",  # Neversink at Woodbourne
    "01437500",  # Neversink at Godeffroy
    "01438500",  # Delaware at Montague
    "01440200",  # Delaware at Water Gap
    "01463500"   # Delaware at Trenton
]

BASE_URL = "https://waterservices.usgs.gov/nwis/iv/"

# Date range for testing (3 months of hourly data)
start = datetime.datetime(2024, 1, 1, 0, 0, 0, 0)
end = datetime.datetime(2024, 4, 1, 0, 0, 0, 0)

print(f"Fetching USGS gauge height data from {start.date()} to {end.date()}")
print(f"Sites: {sites}")

records = []

for site in sites:
    print(f"Fetching {site}...")
    
    params = {
        "format": "json",
        "sites": site,
        "startDT": start.isoformat(),
        "endDT": end.isoformat(),
        "parameterCd": "00065",  # gage height
        "siteStatus": "all"
    }
    
    try:
        r = requests.get(BASE_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        
        try:
            ts = data["value"]["timeSeries"][0]["values"][0]["value"]
        except (KeyError, IndexError):
            print(f"  No data for {site}.")
            continue
            
        for row in ts:
            # Keep full timestamp for instantaneous values
            records.append({
                "datetime": row["dateTime"],
                "site": site,
                "gage_ht_ft": float(row["value"]) if row.get("value") not in (None, "") else float('nan')
            })
            
    except Exception as e:
        print(f"  Error fetching {site}: {e}")
        continue

# Create dataframe
df = pd.DataFrame(records)

# If no records were fetched, exit early
if df.empty:
    print("No records fetched; exiting.")
    sys.exit(0)

# Parse datetimes robustly (coerce errors) and drop rows that failed to parse
# Specify utc=True so all datetimes are timezone-aware (avoids tz-naive vs tz-aware comparisons)
df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
df = df.dropna(subset=["datetime"])  # Remove rows where datetime couldn't be parsed

# Create timezone-aware timestamp bounds for filtering (inclusive)
start_ts = pd.to_datetime(start).tz_localize("UTC")
end_ts = (pd.to_datetime(end) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)).tz_localize("UTC")
df = df[(df["datetime"] >= start_ts) & (df["datetime"] <= end_ts)]

# Pivot wide: rows=datetimes, columns=sites
df_pivot = df.pivot(index="datetime", columns="site", values="gage_ht_ft")

# Sort by datetime
df_pivot = df_pivot.sort_index()

# Linearly interpolate missing values less than 6 hours (6 data points at 1-hr intervals)
# Raise an error if internal gaps bigger than 6 hours
df_pivot = df_pivot.interpolate(method='time', limit=6)

print(f"\nData shape: {df_pivot.shape}")
print(f"Date range: {df_pivot.index.min()} to {df_pivot.index.max()}")
print(f"Sites: {list(df_pivot.columns)}")
print("\nFirst 5 rows:")
print(df_pivot.head())

# Save to local file for testing (this would be committed to repo in real usage)
output_dir = "testing_data/usgs_delaware"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "usgs_gage_ht_3months_iv.csv")
df_pivot.to_csv(output_path)
print(f"\nSaved data to: {output_path}")

# Plot normalized data for visualization
df_pivot_normalized = df_pivot - df_pivot.min()
df_pivot_normalized = df_pivot_normalized.divide(df_pivot_normalized.max())
df_pivot_normalized.plot(figsize=(14, 8))
plt.title("USGS Instantaneous Gage Height for Selected Sites - Last 3 Months (Normalized)")
plt.xlabel("Datetime")
plt.ylabel("Normalized Gage Height")
plt.legend(title="Site")
plt.grid(True)
plt.tight_layout()

plot_path = os.path.join(output_dir, "usgs_gage_ht_normalized.png")
plt.savefig(plot_path, dpi=150)
print(f"Saved plot to: {plot_path}")
plt.show()

print("\nTo use this data in tests:")
print("1. Run this script to generate the CSV file")
print("2. The test will load: './testing_data/usgs_delaware/usgs_gage_ht_3months_iv.csv'")
print("3. Only commit the script and small metadata files, not the large CSV")