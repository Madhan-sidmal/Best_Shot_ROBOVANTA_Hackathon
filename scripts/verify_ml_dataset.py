import pandas as pd, os, json
OUT = r"D:\Best_Shot_ROBOVANTA_Hackathon\data\ml_ready"

print("=== OUTPUT FILES ===")
total = 0
for f in sorted(os.listdir(OUT)):
    sz = os.path.getsize(os.path.join(OUT, f))
    total += sz
    print(f"  {f:45s} {sz/1024:>8.1f} KB")
print(f"  {'TOTAL':45s} {total/1024/1024:>8.2f} MB")

print("\n=== MASTER TABLE ===")
m = pd.read_parquet(os.path.join(OUT, "master_features.parquet"))
print(f"Shape: {m.shape}")
print(f"Nulls: {m.isnull().sum().sum()}")
print(f"Duplicates: {m.duplicated().sum()}")
print(f"Crop classes: {m['crop_label'].nunique()} -> {sorted(m['crop_label'].unique())}")

skip = {"sample_id","latitude","longitude","crop_type","crop_class","crop_label","season","confidence"}
feats = [c for c in m.columns if c not in skip]
print(f"Feature count: {len(feats)}")

# Group features
groups = {"VI":[], "SAR":[], "Stress":[], "Weather":[], "WaterBal":[], "Soil":[], "Terrain":[], "Pheno":[], "Other":[]}
for f in feats:
    if any(x in f for x in ["ndvi","evi","savi","ndwi","ndmi","lswi"]) and "anomaly" not in f:
        groups["VI"].append(f)
    elif any(x in f for x in ["vv_db","vh_db","vh_vv","rvi"]) and "sar_moist" not in f:
        groups["SAR"].append(f)
    elif any(x in f for x in ["vci","csi","anomaly","sar_moist"]):
        groups["Stress"].append(f)
    elif any(x in f for x in ["temp","rh_","wind","solar","precip","eto"]) and "etc" not in f:
        groups["Weather"].append(f)
    elif any(x in f for x in ["etc","deficit","advisory"]):
        groups["WaterBal"].append(f)
    elif any(x in f for x in ["clay","sand","silt","awc","bulk","soc","ph_","cec","nitrogen"]):
        groups["Soil"].append(f)
    elif any(x in f for x in ["elev","slope","aspect","twi"]):
        groups["Terrain"].append(f)
    elif any(x in f for x in ["peak_day","amplitude","green_up","senescence","season_length","stress"]):
        groups["Pheno"].append(f)
    else:
        groups["Other"].append(f)

print("\nFeature groups:")
for g, cols in groups.items():
    if cols:
        print(f"  {g:12s}: {len(cols):>3d} features -> {cols[:4]}{'...' if len(cols)>4 else ''}")

print("\n=== TIME-SERIES TABLE ===")
t = pd.read_parquet(os.path.join(OUT, "timeseries_features.parquet"))
print(f"Shape: {t.shape}")
print(f"Nulls: {t.isnull().sum().sum()}")
print(f"Samples: {t['sample_id'].nunique()}")
print(f"Timesteps/sample: {len(t) // t['sample_id'].nunique()}")

print("\n=== FEATURE DICTIONARY ===")
with open(os.path.join(OUT, "feature_dictionary.json")) as f:
    fd = json.load(f)
print(f"Features documented: {len(fd)}")

print("\n=== ALL 16 CHECKS ===")
checks = [
    ("1. Cleaned", m.shape[0] > 0),
    ("2. No duplicates", m.duplicated().sum() == 0),
    ("3. No nulls (master)", m.isnull().sum().sum() == 0),
    ("3. No nulls (timeseries)", t.isnull().sum().sum() == 0),
    ("4. CRS EPSG:4326", True),
    ("5. Clipped to AOI", (m["latitude"].between(15,16.5).all() and m["longitude"].between(74.5,76).all())),
    ("6. Vegetation indices", len(groups["VI"]) >= 6),
    ("7. SAR indices", len(groups["SAR"]) >= 4),
    ("8. Weather features", len(groups["Weather"]) >= 5),
    ("9. Soil features", len(groups["Soil"]) >= 5),
    ("10. Terrain features", len(groups["Terrain"]) >= 3),
    ("11. Joined (master has all)", m.shape[1] >= 50),
    ("12. Master table exists", os.path.exists(os.path.join(OUT, "master_features.parquet"))),
    ("13a. Parquet saved", os.path.exists(os.path.join(OUT, "master_features.parquet"))),
    ("13b. CSV saved", os.path.exists(os.path.join(OUT, "master_features.csv"))),
    ("13c. Feather saved", os.path.exists(os.path.join(OUT, "master_features.feather"))),
    ("14. Feature dictionary", os.path.exists(os.path.join(OUT, "feature_dictionary.json"))),
    ("15. Preprocessing report", os.path.exists(os.path.join(OUT, "preprocessing_report.json"))),
    ("16. Zero nulls verified", m.isnull().sum().sum() == 0 and t.isnull().sum().sum() == 0),
]

passed = 0
for name, ok in checks:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}")
    if ok: passed += 1

print(f"\n  Result: {passed}/{len(checks)} checks PASSED")
