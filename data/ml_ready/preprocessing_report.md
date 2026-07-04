# KrishiDrishti - Preprocessing Report

## Overview
- **Samples**: 400
- **Features**: 72
- **Timesteps/sample**: 46 (8-day composites over 365 days)
- **CRS**: EPSG:4326
- **AOI**: North Karnataka [74.5E-76.0E, 15.0N-16.5N]
- **Null values**: 0

## Feature Groups

| Group | Count | Examples |
|-------|-------|---------|
| Vegetation Indices | 15 | ndvi_mean, ndvi_std, evi_max, savi_mean |
| SAR Indices | 8 | vv_db_mean, vh_db_std, rvi_max |
| Stress Indices | 8 | vci_mean, csi_min, ndwi_anomaly_mean |
| Weather | 11 | temp_mean, precip_sum, eto_sum |
| Water Balance | 7 | etc_sum, deficit_max, advisory_fracs |
| Soil | 9 | clay_pct, awc_mm_m, soc_g_kg |
| Terrain | 4 | elevation_m, slope_deg, twi |
| Phenology | 7 | ndvi_peak_day, season_length |

**Total: 72 features**

## Target Variables
- `crop_label`: 15-class crop classification
- `stress_flag`: Binary stress detection
- `advisory_class`: 4-level irrigation advisory

## Output Files
| File | Format | Size |
|------|--------|------|
| master_features.parquet | PARQUET | 208.7 KB |
| timeseries_features.parquet | PARQUET | 1931.9 KB |
| master_features.csv | CSV | 399.3 KB |
| timeseries_features.csv | CSV | 6801.8 KB |
| master_features.feather | FEATHER | 202.3 KB |
| timeseries_features.feather | FEATHER | 2129.4 KB |

## Cleaning Steps Applied
- Removed duplicates by (lat, lon, crop, season)
- Filtered confidence >= 0.70
- Clipped to AOI bounding box
- Reprojected/verified EPSG:4326
- Filled all NaN with column median (numeric) or 'unknown' (string)
