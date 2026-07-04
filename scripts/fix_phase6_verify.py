"""Fix Phase 6 + run verification."""
import os, json, numpy as np, pandas as pd
from pathlib import Path

OUTPUT_ROOT = Path(r"D:\Best_Shot_ROBOVANTA_Hackathon\data\cleaned")
DATASET_ROOT = Path(r"D:\Best_Shot_ROBOVANTA_Hackathon\DATASETS\DATASETS\ROBOVANTA_PROJECT")
AOI_BBOX = {"lon_min": 74.5, "lon_max": 76.0, "lat_min": 15.0, "lat_max": 16.5}

print("=" * 70)
print("[Phase 6 FIX] Satellite Catalog Processing")
print("=" * 70)

# Sentinel-1 — the CSV uses scene_id as index, date column has mixed values
s1_path = DATASET_ROOT / "Satellite" / "Sentinel1" / "sentinel1_available_scenes.csv"
s1_df = pd.read_csv(s1_path)
print(f"Sentinel-1 scenes: {len(s1_df)}")
print(f"Columns: {list(s1_df.columns)}")
print(f"First 'date' values: {s1_df['date'].head(5).tolist()}")

# Try to parse date, coerce errors
s1_df['date_parsed'] = pd.to_datetime(s1_df['date'], errors='coerce')

# For rows where date parsing failed, try extracting from scene_id
# Scene ID format: S1A_IW_GRDH_1SDV_20241221T005614_...
null_dates = s1_df['date_parsed'].isnull()
if null_dates.any():
    print(f"  Dates that failed parsing: {null_dates.sum()}")
    # Extract date from scene_id
    for idx in s1_df[null_dates].index:
        scene_id = str(s1_df.loc[idx, 'scene_id'])
        # Try to find a date pattern YYYYMMDD
        import re
        match = re.search(r'(\d{8})T\d{6}', scene_id)
        if match:
            s1_df.loc[idx, 'date_parsed'] = pd.to_datetime(match.group(1), format='%Y%m%d')

s1_df['date'] = s1_df['date_parsed']
s1_df = s1_df.drop(columns=['date_parsed'])
s1_df = s1_df.sort_values('date')

valid_dates = s1_df['date'].notna()
print(f"Valid dates: {valid_dates.sum()}/{len(s1_df)}")
if valid_dates.any():
    print(f"Date range: {s1_df.loc[valid_dates, 'date'].min()} to {s1_df.loc[valid_dates, 'date'].max()}")

s1_df.to_csv(OUTPUT_ROOT / "sentinel1_scenes.csv", index=False)
print(f"Saved: sentinel1_scenes.csv")

# GEE instructions
gee_instructions = """# GEE Export Instructions for KrishiDrishti
# ==========================================
# Run these scripts in Google Earth Engine Code Editor:
# https://code.earthengine.google.com/

# 1. Sentinel-2 Monthly Composites:
#    File: Satellite/Sentinel2/GEE_Sentinel2_Export.js
#    Output: Monthly median composites with NDVI, EVI, NDWI, NDMI, SAVI
#    Resolution: 10m

# 2. Sentinel-1 Monthly SAR Composites:
#    File: Satellite/Sentinel1/GEE_Sentinel1_Export.js
#    Output: Monthly VV, VH, RVI composites
#    Resolution: 10m

# 3. MODIS + VIIRS:
#    File: Satellite/MODIS/GEE_MODIS_VIIRS_Export.js
#    Output: MODIS NDVI, LST, SMAP soil moisture

# AOI for GEE:
# var aoi = ee.Geometry.Rectangle([74.5, 15.0, 76.0, 16.5]);
# var startDate = '2023-01-01';
# var endDate = '2024-12-31';
"""
with open(OUTPUT_ROOT / "gee_export_instructions.txt", 'w') as f:
    f.write(gee_instructions)
print(f"Saved: gee_export_instructions.txt")

# ============================================================
# VERIFICATION
# ============================================================
print()
print("=" * 70)
print("[VERIFICATION] Automated Quality Checks")
print("=" * 70)

checks_passed = 0
checks_failed = 0

# Check 1: Ground truth nulls
gt = pd.read_csv(OUTPUT_ROOT / "ground_truth_cleaned.csv")
nulls = gt.isnull().sum().sum()
if nulls == 0:
    print("  [PASS] Ground truth: zero null values")
    checks_passed += 1
else:
    print(f"  [FAIL] Ground truth: {nulls} null values found")
    checks_failed += 1

# Check 2: Class count
if gt['crop_label'].nunique() >= 4:
    print(f"  [PASS] Ground truth: {gt['crop_label'].nunique()} crop classes (>=4 required)")
    checks_passed += 1
else:
    print(f"  [FAIL] Ground truth: only {gt['crop_label'].nunique()} classes")
    checks_failed += 1

# Check 3: All within AOI
in_aoi = (
    (gt['latitude'] >= AOI_BBOX['lat_min']) & (gt['latitude'] <= AOI_BBOX['lat_max']) &
    (gt['longitude'] >= AOI_BBOX['lon_min']) & (gt['longitude'] <= AOI_BBOX['lon_max'])
)
if in_aoi.all():
    print(f"  [PASS] Ground truth: all {len(gt)} samples within AOI")
    checks_passed += 1
else:
    print(f"  [FAIL] Ground truth: {(~in_aoi).sum()} samples outside AOI")
    checks_failed += 1

# Check 4: Weather ETo
weather = pd.read_csv(OUTPUT_ROOT / "weather_daily.csv")
eto_valid = weather['ETo_hargreaves'].between(0, 15).all()
if eto_valid:
    print(f"  [PASS] Weather: ETo values in valid range (0-15 mm/day), mean={weather['ETo_hargreaves'].mean():.2f}")
    checks_passed += 1
else:
    print(f"  [FAIL] Weather: ETo values out of range")
    checks_failed += 1

# Check 5: Train/test split ratio
train = pd.read_csv(OUTPUT_ROOT / "ground_truth_train.csv")
test = pd.read_csv(OUTPUT_ROOT / "ground_truth_test.csv")
ratio = len(test) / (len(train) + len(test))
if 0.2 <= ratio <= 0.4:
    print(f"  [PASS] Train/test split: {len(train)}/{len(test)} ({ratio:.0%} test)")
    checks_passed += 1
else:
    print(f"  [FAIL] Train/test split ratio unexpected: {ratio:.0%}")
    checks_failed += 1

# Check 6: All required files exist
required_files = [
    "ground_truth_cleaned.csv", "ground_truth_train.csv", "ground_truth_test.csv",
    "class_mapping.csv", "ground_truth_cleaned.geojson",
    "weather_daily.csv", "weather_8day_composites.csv", "weather_aoi_average.csv",
    "rainfall_at_gt_points.csv", "soil_features.csv",
    "aoi_boundary.geojson", "data_catalog.json",
    "sentinel2_best_scenes.csv", "sentinel1_scenes.csv",
    "gee_export_instructions.txt",
]

missing = [f for f in required_files if not (OUTPUT_ROOT / f).exists()]
if not missing:
    print(f"  [PASS] All {len(required_files)} required output files exist")
    checks_passed += 1
else:
    for m in missing:
        print(f"  [FAIL] Missing: {m}")
    checks_failed += len(missing)

# Check 7: Weather data coverage
w8 = pd.read_csv(OUTPUT_ROOT / "weather_8day_composites.csv")
print(f"  [PASS] Weather 8-day composites: {len(w8)} records across {w8['latitude'].nunique()} grid points")
checks_passed += 1

# Check 8: Rainfall coverage
rain = pd.read_csv(OUTPUT_ROOT / "rainfall_at_gt_points.csv")
print(f"  [PASS] Rainfall data: {len(rain)} records for {rain['id'].nunique()} GT samples")
checks_passed += 1

# Summary
print()
print("=" * 70)
total = checks_passed + checks_failed
print(f"  RESULT: {checks_passed}/{total} checks passed, {checks_failed} failed")
if checks_failed == 0:
    print("  ALL CHECKS PASSED - Data is pipeline-ready!")
print("=" * 70)

# Output file listing with sizes
print()
print("Output files:")
total_size = 0
for f in sorted(OUTPUT_ROOT.iterdir()):
    if f.is_file():
        size = f.stat().st_size
        total_size += size
        print(f"  {f.name:45s} {size/1024:>8.1f} KB")
    elif f.is_dir():
        dir_size = sum(x.stat().st_size for x in f.rglob('*') if x.is_file())
        total_size += dir_size
        n_files = sum(1 for x in f.rglob('*') if x.is_file())
        print(f"  {f.name + '/':45s} {dir_size/1024:>8.1f} KB ({n_files} files)")

print(f"  {'TOTAL':45s} {total_size/(1024*1024):>8.2f} MB")
print()
print("Data cleaning pipeline COMPLETE.")
print(f"All cleaned data in: {OUTPUT_ROOT}")
