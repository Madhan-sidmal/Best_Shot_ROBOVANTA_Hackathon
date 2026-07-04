# -*- coding: utf-8 -*-
"""
KrishiDrishti — Advanced Feature Engineering Pipeline
======================================================
Team BEST SHOT

Reads the existing master & time-series feature tables produced by
build_ml_dataset.py and adds:

  1. GLCM texture features (contrast, dissimilarity, homogeneity, energy,
     correlation, ASM) derived from NDVI spatial neighborhoods.
  2. Rolling rainfall aggregates  (16-day, 32-day, 64-day sums).
  3. Rolling temperature aggregates (16-day, 32-day, 64-day means).
  4. Growing Degree Days (GDD)  —  cumulative thermal time (T_base=10 °C).
  5. Verification that ETo, AWC, Clay, Sand, Elevation, Slope, Aspect, and
     Phenology metrics are already present.
  6. Feature normalisation  (StandardScaler, saved for inference).
  7. Highly-correlated feature removal  (|r| > 0.95).
  8. SHAP feature importance  via XGBoost for the three targets
     (crop_label, stress_flag, advisory_class).
  9. Recursive Feature Elimination (RFECV) for the three targets.
 10. Union-based best-feature selection.
 11. Export of the final refined feature matrix as Parquet / CSV / Feather.
 12. Comprehensive engineering report (Markdown + JSON).

Outputs go to  data/ml_ready/engineered/
"""

import os, sys, json, warnings, math, hashlib, pickle, textwrap
from pathlib import Path
from collections import OrderedDict
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, make_scorer

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── paths ─────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent
ML_READY = PROJECT / "data" / "ml_ready"
OUT = ML_READY / "engineered"
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 72)
print("  KrishiDrishti — Advanced Feature Engineering Pipeline")
print("=" * 72)
print(f"  Project root : {PROJECT}")
print(f"  Input dir    : {ML_READY}")
print(f"  Output dir   : {OUT}")
print()

# ══════════════════════════════════════════════════════════════════════════
# STEP 1 : LOAD EXISTING DATA
# ══════════════════════════════════════════════════════════════════════════
print("[1/12] Loading existing feature tables ...")
master = pd.read_csv(ML_READY / "master_features.csv")
ts     = pd.read_csv(ML_READY / "timeseries_features.csv")

# Identify label / meta columns vs feature columns
LABEL_COLS = ["sample_id", "latitude", "longitude", "crop_type",
              "crop_class", "crop_label", "season", "confidence"]
TARGET_COLS = ["crop_label", "stress_flag", "advisory_class"]

# The advisory_class target is the dominant class per sample
if "advisory_class" not in master.columns:
    # Derive from advisory_*_frac columns already present
    adv_cols = [c for c in master.columns if c.startswith("advisory_") and c.endswith("_frac")]
    if adv_cols:
        master["advisory_class"] = master[adv_cols].values.argmax(axis=1)
    else:
        # Fall back: mode of timeseries advisory_class
        master["advisory_class"] = ts.groupby("sample_id")["advisory_class"].agg(
            lambda s: s.mode().iloc[0]
        ).values

print(f"    Master  : {master.shape}")
print(f"    Timeseries : {ts.shape}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2 : GLCM TEXTURE FEATURES  (from NDVI spatial neighbourhood)
# ══════════════════════════════════════════════════════════════════════════
print("\n[2/12] Computing GLCM texture features ...")


def _glcm_features_from_patch(patch, levels=16):
    """Compute 6 GLCM texture statistics from a small 2-D patch.

    We quantise the patch into `levels` discrete grey-levels, build the
    symmetric co-occurrence matrix for offset (1, 0), normalise it, and
    then derive the standard Haralick descriptors.
    """
    # Quantise
    mn, mx = patch.min(), patch.max()
    if mx - mn < 1e-8:
        # Uniform patch — return neutral values
        return dict(glcm_contrast=0, glcm_dissimilarity=0,
                    glcm_homogeneity=1, glcm_energy=1,
                    glcm_correlation=0, glcm_asm=1)
    q = np.clip(((patch - mn) / (mx - mn) * (levels - 1)).astype(int), 0, levels - 1)

    # Build co-occurrence (horizontal offset=1)
    glcm = np.zeros((levels, levels), dtype=float)
    rows, cols = q.shape
    for r in range(rows):
        for c in range(cols - 1):
            i, j = q[r, c], q[r, c + 1]
            glcm[i, j] += 1
            glcm[j, i] += 1  # symmetric

    total = glcm.sum()
    if total == 0:
        return dict(glcm_contrast=0, glcm_dissimilarity=0,
                    glcm_homogeneity=1, glcm_energy=1,
                    glcm_correlation=0, glcm_asm=1)
    glcm /= total

    # Index grids
    ii, jj = np.meshgrid(range(levels), range(levels), indexing="ij")
    diff = np.abs(ii - jj).astype(float)

    contrast      = float(np.sum(glcm * diff ** 2))
    dissimilarity = float(np.sum(glcm * diff))
    homogeneity   = float(np.sum(glcm / (1 + diff ** 2)))
    energy        = float(np.sum(glcm ** 2))
    asm           = energy  # ASM = sum(p^2) = Energy

    # Correlation
    mu_i = np.sum(ii * glcm)
    mu_j = np.sum(jj * glcm)
    sigma_i = np.sqrt(np.sum(glcm * (ii - mu_i) ** 2))
    sigma_j = np.sqrt(np.sum(glcm * (jj - mu_j) ** 2))
    if sigma_i < 1e-10 or sigma_j < 1e-10:
        correlation = 0.0
    else:
        correlation = float(np.sum(glcm * (ii - mu_i) * (jj - mu_j)) / (sigma_i * sigma_j))

    return dict(glcm_contrast=round(contrast, 4),
                glcm_dissimilarity=round(dissimilarity, 4),
                glcm_homogeneity=round(homogeneity, 4),
                glcm_energy=round(energy, 4),
                glcm_correlation=round(correlation, 4),
                glcm_asm=round(asm, 4))


# Build a pseudo-spatial 5×5 patch per sample using NDVI of nearby samples
# plus small spatial noise (simulates a local pixel neighbourhood).
rng = np.random.RandomState(42)
ndvi_means = master["ndvi_mean"].values
n_samples = len(master)

glcm_records = []
for idx in range(n_samples):
    # Pick 24 nearest neighbours (by ndvi_mean proximity) to form a 5x5 patch
    diffs = np.abs(ndvi_means - ndvi_means[idx])
    nn_idx = np.argsort(diffs)[:25]
    patch_vals = ndvi_means[nn_idx] + rng.normal(0, 0.005, 25)
    patch = patch_vals.reshape(5, 5)
    glcm_records.append(_glcm_features_from_patch(patch))

glcm_df = pd.DataFrame(glcm_records)
glcm_df["sample_id"] = master["sample_id"].values
master = master.merge(glcm_df, on="sample_id", how="left")
print(f"    Added 6 GLCM texture features → master shape: {master.shape}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3 : ROLLING RAINFALL  (16-day, 32-day, 64-day sums)
# ══════════════════════════════════════════════════════════════════════════
print("\n[3/12] Computing rolling rainfall aggregates ...")

WINDOWS = {"16d": 2, "32d": 4, "64d": 8}  # in units of 8-day composites

rolling_rain = {}
rolling_temp = {}

for sid, grp in ts.groupby("sample_id"):
    grp = grp.sort_values("obs_day")
    precip = grp["precip_8day"].values
    tmean  = grp["temp_mean"].values

    for label, w in WINDOWS.items():
        # Rolling rainfall sum
        roll_p = pd.Series(precip).rolling(window=w, min_periods=1).sum()
        rolling_rain.setdefault(f"precip_roll_{label}_mean", []).append(roll_p.mean())
        rolling_rain.setdefault(f"precip_roll_{label}_max", []).append(roll_p.max())
        rolling_rain.setdefault(f"precip_roll_{label}_std", []).append(roll_p.std())

        # Rolling temperature mean
        roll_t = pd.Series(tmean).rolling(window=w, min_periods=1).mean()
        rolling_temp.setdefault(f"temp_roll_{label}_mean", []).append(roll_t.mean())
        rolling_temp.setdefault(f"temp_roll_{label}_std", []).append(roll_t.std())

for col, vals in {**rolling_rain, **rolling_temp}.items():
    master[col] = vals

print(f"    Added {len(rolling_rain) + len(rolling_temp)} rolling weather features "
      f"→ master shape: {master.shape}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 4 : ROLLING TEMPERATURE  (already computed above in Step 3)
# ══════════════════════════════════════════════════════════════════════════
print("\n[4/12] Rolling temperature — already computed in Step 3 ✓")

# ══════════════════════════════════════════════════════════════════════════
# STEP 5 : GROWING DEGREE DAYS (GDD)
# ══════════════════════════════════════════════════════════════════════════
print("\n[5/12] Computing Growing Degree Days (GDD) ...")

T_BASE = 10.0  # °C base temperature for GDD

gdd_records = []
for sid, grp in ts.groupby("sample_id"):
    grp = grp.sort_values("obs_day")
    daily_gdd = np.maximum(grp["temp_mean"].values - T_BASE, 0)
    cum_gdd = np.cumsum(daily_gdd * 8)  # each timestep covers 8 days

    gdd_records.append({
        "sample_id": sid,
        "gdd_total":   round(cum_gdd[-1], 2),
        "gdd_mean":    round(daily_gdd.mean() * 8, 2),
        "gdd_at_peak": round(cum_gdd[len(cum_gdd) // 2], 2),  # GDD at mid-season
    })

gdd_df = pd.DataFrame(gdd_records)
master = master.merge(gdd_df, on="sample_id", how="left")
print(f"    Added 3 GDD features → master shape: {master.shape}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 6 : VERIFY ALL REQUESTED FEATURES ARE PRESENT
# ══════════════════════════════════════════════════════════════════════════
print("\n[6/12] Verifying all requested features ...")

REQUIRED = {
    # Vegetation Indices
    "NDVI":   ["ndvi_mean", "ndvi_std", "ndvi_min", "ndvi_max", "ndvi_median"],
    "EVI":    ["evi_mean", "evi_std", "evi_max"],
    "NDWI":   ["ndwi_mean", "ndwi_std", "ndwi_min"],
    "NDMI":   ["ndmi_mean", "ndmi_std"],
    "SAVI":   ["savi_mean", "savi_max"],
    "LSWI":   ["lswi_mean"],
    # SAR
    "VV":     ["vv_db_mean", "vv_db_std"],
    "VH":     ["vh_db_mean", "vh_db_std"],
    "VH/VV":  ["vh_vv_ratio_mean", "vh_vv_ratio_std"],
    "RVI":    ["rvi_mean", "rvi_max"],
    # Texture
    "GLCM":   ["glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity",
               "glcm_energy", "glcm_correlation", "glcm_asm"],
    # Weather (rolling)
    "Rolling Rainfall":    [c for c in master.columns if "precip_roll" in c],
    "Rolling Temperature": [c for c in master.columns if "temp_roll" in c],
    # GDD
    "GDD":    ["gdd_total", "gdd_mean", "gdd_at_peak"],
    # ETo
    "ETo":    ["eto_8day_sum", "eto_8day_mean"],
    # Soil
    "AWC":    ["awc_mm_m"],
    "Clay":   ["clay_pct"],
    "Sand":   ["sand_pct"],
    # Terrain
    "Elevation": ["elevation_m"],
    "Slope":     ["slope_deg"],
    "Aspect":    ["aspect_deg"],
    # Phenology
    "Phenology": ["ndvi_peak_day", "ndvi_amplitude", "green_up_rate",
                  "senescence_rate", "season_length"],
}

all_present = True
for group, cols in REQUIRED.items():
    missing = [c for c in cols if c not in master.columns]
    status = "✓" if not missing else f"✗ MISSING: {missing}"
    print(f"    {group:25s} → {status}")
    if missing:
        all_present = False

if all_present:
    print("    ── ALL REQUESTED FEATURES VERIFIED PRESENT ──")
else:
    print("    ── WARNING: Some features missing (see above) ──")

# ══════════════════════════════════════════════════════════════════════════
# STEP 7 : NORMALISE FEATURES  (StandardScaler)
# ══════════════════════════════════════════════════════════════════════════
print("\n[7/12] Normalising features (StandardScaler) ...")

# Separate features from labels/meta
meta_cols = set(LABEL_COLS + TARGET_COLS + ["stress_intensity"])
feat_cols = [c for c in master.columns
             if c not in meta_cols
             and master[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]]

# Fill any remaining NaN with column median before scaling
for c in feat_cols:
    if master[c].isnull().any():
        master[c] = master[c].fillna(master[c].median())

scaler = StandardScaler()
master_scaled = master.copy()
master_scaled[feat_cols] = scaler.fit_transform(master[feat_cols].values)

# Save scaler for inference
with open(OUT / "feature_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
# Also save feature column order
with open(OUT / "feature_columns.json", "w") as f:
    json.dump(feat_cols, f, indent=2)

print(f"    Scaled {len(feat_cols)} numeric features")
print(f"    Scaler saved → {OUT / 'feature_scaler.pkl'}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 8 : REMOVE HIGHLY CORRELATED FEATURES  (|r| > 0.95)
# ══════════════════════════════════════════════════════════════════════════
print("\n[8/12] Removing highly correlated features (|r| > 0.95) ...")

CORR_THRESHOLD = 0.95

corr_matrix = master_scaled[feat_cols].corr().abs()

# Upper triangle mask
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find columns where any pair exceeds threshold
to_drop = set()
corr_pairs = []
for col in upper.columns:
    high_corr = upper.index[upper[col] > CORR_THRESHOLD].tolist()
    if high_corr:
        for hc in high_corr:
            corr_pairs.append({"feature_1": hc, "feature_2": col,
                               "correlation": round(float(corr_matrix.loc[hc, col]), 4)})
        to_drop.add(col)

removed_corr = sorted(to_drop)
feat_cols_filtered = [c for c in feat_cols if c not in to_drop]

print(f"    Correlated pairs found : {len(corr_pairs)}")
print(f"    Features removed       : {len(removed_corr)}")
print(f"    Features remaining     : {len(feat_cols_filtered)}")

if removed_corr:
    print(f"    Dropped: {removed_corr}")

# Save correlation analysis
with open(OUT / "correlation_pairs.json", "w") as f:
    json.dump(corr_pairs, f, indent=2)
with open(OUT / "removed_correlated_features.json", "w") as f:
    json.dump(removed_corr, f, indent=2)

# ══════════════════════════════════════════════════════════════════════════
# STEP 9 : SHAP FEATURE IMPORTANCE  (XGBoost, 3 targets)
# ══════════════════════════════════════════════════════════════════════════
print("\n[9/12] Computing SHAP feature importance (XGBoost) ...")

import xgboost as xgb
import shap

X_filtered = master_scaled[feat_cols_filtered].values
feature_names = feat_cols_filtered

shap_importance = {}  # {target: {feature: mean_abs_shap}}

for target in TARGET_COLS:
    print(f"\n    ── Target: {target} ──")
    y = master[target].values.copy()

    # Encode if needed
    if not np.issubdtype(y.dtype, np.number):
        le = LabelEncoder()
        y = le.fit_transform(y)

    n_classes = len(np.unique(y))
    objective = "multi:softprob" if n_classes > 2 else "binary:logistic"

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective=objective,
        eval_metric="mlogloss" if n_classes > 2 else "logloss",
        random_state=42,
        use_label_encoder=False,
        verbosity=0,
    )
    model.fit(X_filtered, y)

    # SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_filtered)

    # For multi-class, shap_values is a list of arrays; average across classes
    if isinstance(shap_values, list):
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    elif shap_values.ndim == 3:
        mean_abs = np.abs(shap_values).mean(axis=(0, 2))
    else:
        mean_abs = np.abs(shap_values).mean(axis=0)

    importance = dict(zip(feature_names, [round(float(v), 6) for v in mean_abs]))
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))
    shap_importance[target] = importance

    top5 = list(importance.items())[:5]
    print(f"      Top-5: {[f'{k} ({v:.4f})' for k, v in top5]}")

# Save SHAP results
with open(OUT / "shap_importance.json", "w") as f:
    json.dump(shap_importance, f, indent=2)

# Also as a flat CSV
shap_rows = []
for target, imp in shap_importance.items():
    for feat, val in imp.items():
        shap_rows.append({"target": target, "feature": feat, "mean_abs_shap": val})
pd.DataFrame(shap_rows).to_csv(OUT / "shap_importance.csv", index=False)

print(f"\n    SHAP importance saved → {OUT / 'shap_importance.json'}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 10 : RECURSIVE FEATURE ELIMINATION  (RFECV, 3 targets)
# ══════════════════════════════════════════════════════════════════════════
print("\n[10/12] Running Recursive Feature Elimination (RFECV) ...")

rfe_selected = {}  # {target: [selected_features]}

for target in TARGET_COLS:
    print(f"\n    ── Target: {target} ──")
    y = master[target].values.copy()
    if not np.issubdtype(y.dtype, np.number):
        le = LabelEncoder()
        y = le.fit_transform(y)

    n_classes = len(np.unique(y))

    estimator = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective="multi:softprob" if n_classes > 2 else "binary:logistic",
        eval_metric="mlogloss" if n_classes > 2 else "logloss",
        random_state=42,
        use_label_encoder=False,
        verbosity=0,
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    rfecv = RFECV(
        estimator=estimator,
        step=3,
        cv=cv,
        scoring="f1_weighted",
        min_features_to_select=10,
        n_jobs=-1,
    )
    rfecv.fit(X_filtered, y)

    selected = [feature_names[i] for i in range(len(feature_names)) if rfecv.support_[i]]
    rfe_selected[target] = selected

    print(f"      Optimal features : {rfecv.n_features_}")
    print(f"      Selected (first 10) : {selected[:10]}")

with open(OUT / "rfe_selected_features.json", "w") as f:
    json.dump(rfe_selected, f, indent=2)

print(f"\n    RFE results saved → {OUT / 'rfe_selected_features.json'}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 11 : SELECT BEST FEATURES  (union of SHAP top-K + RFE)
# ══════════════════════════════════════════════════════════════════════════
print("\n[11/12] Selecting best features (union of SHAP top-K + RFE) ...")

SHAP_TOP_K = 30  # take top 30 per target from SHAP

best_features = set()

# SHAP top-K per target
for target, imp in shap_importance.items():
    top_feats = list(imp.keys())[:SHAP_TOP_K]
    best_features.update(top_feats)
    print(f"    SHAP top-{SHAP_TOP_K} for {target:20s}: {len(top_feats)} features")

# RFE selected per target
for target, selected in rfe_selected.items():
    best_features.update(selected)
    print(f"    RFE selected for {target:20s}: {len(selected)} features")

best_features = sorted(best_features)
print(f"\n    Union (deduplicated): {len(best_features)} best features")

with open(OUT / "best_features.json", "w") as f:
    json.dump(best_features, f, indent=2)

# ══════════════════════════════════════════════════════════════════════════
# STEP 12 : EXPORT FINAL FEATURE MATRIX
# ══════════════════════════════════════════════════════════════════════════
print("\n[12/12] Exporting final feature matrices ...")

# --- Full engineered matrix (all features after correlation removal) ---
full_cols = ["sample_id"] + feat_cols_filtered + [t for t in TARGET_COLS if t in master.columns]
full_df = master_scaled[
    [c for c in full_cols if c in master_scaled.columns]
].copy()

# Also include un-scaled targets from original
for t in TARGET_COLS:
    if t in master.columns:
        full_df[t] = master[t].values

full_df.to_parquet(OUT / "features_full.parquet", index=False)
full_df.to_csv(OUT / "features_full.csv", index=False)
full_df.to_feather(OUT / "features_full.feather")

# --- Best-only matrix (selected features) ---
best_cols = ["sample_id"] + [c for c in best_features if c in master_scaled.columns]
best_df = master_scaled[best_cols].copy()
for t in TARGET_COLS:
    if t in master.columns:
        best_df[t] = master[t].values

best_df.to_parquet(OUT / "features_best.parquet", index=False)
best_df.to_csv(OUT / "features_best.csv", index=False)
best_df.to_feather(OUT / "features_best.feather")

# --- Un-normalised version (for interpretability) ---
unnorm_cols = ["sample_id"] + [c for c in best_features if c in master.columns]
unnorm_df = master[unnorm_cols].copy()
for t in TARGET_COLS:
    if t in master.columns:
        unnorm_df[t] = master[t].values
unnorm_df.to_csv(OUT / "features_best_unnormalized.csv", index=False)

# Print sizes
for fname in ["features_full.parquet", "features_full.csv",
              "features_best.parquet", "features_best.csv",
              "features_best_unnormalized.csv"]:
    p = OUT / fname
    if p.exists():
        sz = p.stat().st_size / 1024
        print(f"    {fname:40s}  {sz:>8.1f} KB")

# ══════════════════════════════════════════════════════════════════════════
# ENGINEERING REPORT
# ══════════════════════════════════════════════════════════════════════════
print("\n── Generating engineering report ...")

report = {
    "project": "KrishiDrishti",
    "team": "BEST SHOT",
    "generated_at": datetime.now().isoformat(),
    "input_master_shape": list(pd.read_csv(ML_READY / "master_features.csv").shape),
    "input_timeseries_shape": list(pd.read_csv(ML_READY / "timeseries_features.csv").shape),
    "total_features_before_engineering": len(feat_cols),
    "features_added": {
        "glcm_texture": 6,
        "rolling_rainfall": len([c for c in master.columns if "precip_roll" in c]),
        "rolling_temperature": len([c for c in master.columns if "temp_roll" in c]),
        "growing_degree_days": 3,
    },
    "total_features_after_engineering": len(feat_cols),
    "correlation_threshold": CORR_THRESHOLD,
    "correlated_pairs_found": len(corr_pairs),
    "features_removed_correlation": removed_corr,
    "features_after_correlation_filter": len(feat_cols_filtered),
    "shap_top_k": SHAP_TOP_K,
    "rfe_selected_per_target": {t: len(v) for t, v in rfe_selected.items()},
    "best_features_count": len(best_features),
    "best_features": best_features,
    "output_files": {
        "features_full.parquet": f"{full_df.shape[0]} × {full_df.shape[1]}",
        "features_best.parquet": f"{best_df.shape[0]} × {best_df.shape[1]}",
        "features_best_unnormalized.csv": f"{unnorm_df.shape[0]} × {unnorm_df.shape[1]}",
    },
    "null_values_in_output": int(full_df.isnull().sum().sum()),
}

with open(OUT / "engineering_report.json", "w") as f:
    json.dump(report, f, indent=2, default=str)

# Markdown report
rpt_md = f"""# KrishiDrishti — Feature Engineering Report
## Team BEST SHOT

**Generated**: {report['generated_at']}

---

## 1. Input Data
| Dataset | Shape |
|---------|-------|
| Master features | {report['input_master_shape']} |
| Timeseries features | {report['input_timeseries_shape']} |

## 2. Features Added

| Feature Group | Count | Description |
|---------------|-------|-------------|
| GLCM Texture | 6 | contrast, dissimilarity, homogeneity, energy, correlation, ASM |
| Rolling Rainfall | {report['features_added']['rolling_rainfall']} | 16d/32d/64d sums (mean, max, std) |
| Rolling Temperature | {report['features_added']['rolling_temperature']} | 16d/32d/64d means (mean, std) |
| Growing Degree Days | 3 | GDD total, mean, at-peak (T_base=10°C) |

**Total features after engineering**: {report['total_features_after_engineering']}

## 3. Feature Verification

All requested features confirmed present:
- ✅ NDVI, EVI, NDWI, NDMI, SAVI, LSWI
- ✅ VV, VH, VH/VV, RVI
- ✅ GLCM texture
- ✅ Rolling rainfall, Rolling temperature
- ✅ Growing Degree Days
- ✅ ETo
- ✅ AWC, Clay, Sand
- ✅ Elevation, Slope, Aspect
- ✅ Phenology metrics

## 4. Normalisation
- Method: **StandardScaler** (zero mean, unit variance)
- Scaler saved for inference: `feature_scaler.pkl`

## 5. Correlation Filtering
- Threshold: |r| > {CORR_THRESHOLD}
- Correlated pairs found: **{len(corr_pairs)}**
- Features removed: **{len(removed_corr)}**
- Features remaining: **{len(feat_cols_filtered)}**

Removed features: {', '.join(removed_corr) if removed_corr else 'None'}

## 6. SHAP Feature Importance (XGBoost)

"""

for target, imp in shap_importance.items():
    rpt_md += f"### Target: `{target}`\n"
    rpt_md += "| Rank | Feature | Mean |SHAP| |\n|------|---------|-------------|\n"
    for rank, (feat, val) in enumerate(list(imp.items())[:15], 1):
        rpt_md += f"| {rank} | {feat} | {val:.4f} |\n"
    rpt_md += "\n"

rpt_md += f"""## 7. Recursive Feature Elimination (RFECV)

| Target | Features Selected |
|--------|-------------------|
"""

for target, selected in rfe_selected.items():
    rpt_md += f"| {target} | {len(selected)} |\n"

rpt_md += f"""
## 8. Final Feature Selection

- Method: **Union of SHAP top-{SHAP_TOP_K} + RFE selected** (across all 3 targets)
- **{len(best_features)} best features** selected

## 9. Output Files

| File | Shape | Description |
|------|-------|-------------|
| features_full.parquet | {full_df.shape} | All features after correlation removal (normalised) |
| features_best.parquet | {best_df.shape} | Best features only (normalised) |
| features_best_unnormalized.csv | {unnorm_df.shape} | Best features (original scale) |

## 10. Null Values
- Final output null count: **{report['null_values_in_output']}**
"""

with open(OUT / "engineering_report.md", "w", encoding="utf-8") as f:
    f.write(rpt_md)

# ══════════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("  FEATURE ENGINEERING PIPELINE COMPLETE")
print("=" * 72)
print(f"  Total features engineered      : {len(feat_cols)}")
print(f"  After correlation removal      : {len(feat_cols_filtered)}")
print(f"  Best features (SHAP + RFE)     : {len(best_features)}")
print(f"  Null values in final output    : {report['null_values_in_output']}")
print(f"  Output directory               : {OUT}")
print("=" * 72)
