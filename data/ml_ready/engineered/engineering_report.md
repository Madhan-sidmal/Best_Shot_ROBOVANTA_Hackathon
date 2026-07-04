# KrishiDrishti — Feature Engineering Report
## Team BEST SHOT

**Generated**: 2026-07-04T22:36:47.433023

---

## 1. Input Data
| Dataset | Shape |
|---------|-------|
| Master features | [400, 80] |
| Timeseries features | [18400, 53] |

## 2. Features Added

| Feature Group | Count | Description |
|---------------|-------|-------------|
| GLCM Texture | 6 | contrast, dissimilarity, homogeneity, energy, correlation, ASM |
| Rolling Rainfall | 9 | 16d/32d/64d sums (mean, max, std) |
| Rolling Temperature | 6 | 16d/32d/64d means (mean, std) |
| Growing Degree Days | 3 | GDD total, mean, at-peak (T_base=10°C) |

**Total features after engineering**: 94

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
- Threshold: |r| > 0.95
- Correlated pairs found: **116**
- Features removed: **36**
- Features remaining: **58**

Removed features: csi_mean, eto_8day_mean, eto_8day_sum, evi_max, evi_mean, gdd_at_peak, gdd_mean, gdd_total, glcm_asm, lswi_mean, ndmi_mean, ndwi_mean, precip_8day_mean, precip_8day_sum, precip_roll_16d_max, precip_roll_16d_mean, precip_roll_16d_std, precip_roll_32d_max, precip_roll_32d_mean, precip_roll_32d_std, precip_roll_64d_max, precip_roll_64d_mean, precip_roll_64d_std, rh_mean_mean, rvi_mean, savi_max, savi_mean, season_length, temp_roll_16d_mean, temp_roll_32d_mean, temp_roll_32d_std, temp_roll_64d_mean, temp_roll_64d_std, vh_db_mean, water_deficit_mean, water_deficit_sum

## 6. SHAP Feature Importance (XGBoost)

### Target: `crop_label`
| Rank | Feature | Mean |SHAP| |
|------|---------|-------------|
| 1 | vci_mean | 0.4283 |
| 2 | ndvi_max | 0.3846 |
| 3 | advisory_3_frac | 0.2051 |
| 4 | ndvi_median | 0.2010 |
| 5 | ndwi_anomaly_min | 0.1857 |
| 6 | ndvi_mean | 0.1772 |
| 7 | vv_db_mean | 0.1541 |
| 8 | etc_8day_sum | 0.1497 |
| 9 | ndvi_amplitude | 0.1478 |
| 10 | ndwi_anomaly_mean | 0.1414 |
| 11 | soc_g_kg | 0.1159 |
| 12 | ndmi_std | 0.1135 |
| 13 | ndwi_std | 0.1078 |
| 14 | vh_vv_ratio_mean | 0.1073 |
| 15 | silt_pct | 0.1023 |

### Target: `stress_flag`
| Rank | Feature | Mean |SHAP| |
|------|---------|-------------|
| 1 | evi_std | 1.0290 |
| 2 | ndvi_amplitude | 0.3724 |
| 3 | vci_mean | 0.3709 |
| 4 | senescence_rate | 0.3534 |
| 5 | ndvi_std | 0.3333 |
| 6 | awc_mm_m | 0.3273 |
| 7 | ndwi_anomaly_mean | 0.2891 |
| 8 | ndwi_std | 0.2447 |
| 9 | green_up_rate | 0.2433 |
| 10 | clay_pct | 0.2353 |
| 11 | ndmi_std | 0.2283 |
| 12 | aspect_deg | 0.2236 |
| 13 | sar_moisture_proxy_mean | 0.2234 |
| 14 | ndvi_median | 0.2144 |
| 15 | ndvi_min | 0.2075 |

### Target: `advisory_class`
| Rank | Feature | Mean |SHAP| |
|------|---------|-------------|
| 1 | advisory_2_frac | 0.7701 |
| 2 | advisory_0_frac | 0.5209 |
| 3 | etc_8day_sum | 0.2396 |
| 4 | temp_range_mean | 0.2073 |
| 5 | ndwi_anomaly_mean | 0.1489 |
| 6 | advisory_1_frac | 0.1122 |
| 7 | bulk_density | 0.1105 |
| 8 | rvi_max | 0.0995 |
| 9 | temp_min_min | 0.0912 |
| 10 | evi_std | 0.0554 |
| 11 | green_up_rate | 0.0552 |
| 12 | precip_8day_max | 0.0534 |
| 13 | vh_vv_ratio_mean | 0.0498 |
| 14 | elevation_m | 0.0401 |
| 15 | soc_g_kg | 0.0382 |

## 7. Recursive Feature Elimination (RFECV)

| Target | Features Selected |
|--------|-------------------|
| crop_label | 43 |
| stress_flag | 52 |
| advisory_class | 13 |

## 8. Final Feature Selection

- Method: **Union of SHAP top-30 + RFE selected** (across all 3 targets)
- **57 best features** selected

## 9. Output Files

| File | Shape | Description |
|------|-------|-------------|
| features_full.parquet | (400, 62) | All features after correlation removal (normalised) |
| features_best.parquet | (400, 61) | Best features only (normalised) |
| features_best_unnormalized.csv | (400, 61) | Best features (original scale) |

## 10. Null Values
- Final output null count: **0**
