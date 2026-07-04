# 🛰️ KrishiDrishti — Simulated Satellite-Index Crop Classification & Growth-Stage-Aware Water-Deficit Advisory

> **Team BEST SHOT | Problem Statement 06 | ROBOVANTA Hackathon**
> Track: Software — Smart Agriculture & Precision Farming

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> ⚠️ **SIMULATED DATA DISCLAIMER**: All vegetation/moisture-index data, crop-growth curves, and water-stress signals in this project are **synthetically generated** by our parametric simulator. No real satellite imagery, ISRO/Bhuvan data, or government datasets are used. This output must NOT be presented as a validated crop-monitoring or irrigation-advisory product.

---

## 🌾 Overview

**KrishiDrishti** (कृषिदृष्टि — "Agricultural Vision") is an end-to-end simulated crop intelligence pipeline that:

1. **Simulates** realistic satellite vegetation-index time series for 4 crop types using a documented double-logistic growth curve model
2. **Injects** sensor noise and Markov-chain clustered cloud gaps mimicking monsoon-season observation losses
3. **Classifies crop types** from the noisy, gapped observations using ensemble ML (Random Forest + XGBoost)
4. **Estimates growth stages** (sowing → vegetative → flowering → maturity → harvest) — evaluated separately from crop classification
5. **Generates water-deficit advisories** using FAO-56 crop water balance with measured false-alarm and missed-stress rates
6. **Visualizes** everything on an interactive Streamlit dashboard with a persistent simulated-data disclosure banner

## 🚀 One-Command Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline with a fresh random seed
python run.py --seed 123

# Run with dashboard
python run.py --seed 123 --dash

# Custom plot count
python run.py --seed 42 --plots 300
```

This single command:
- Generates a fresh simulated season on the specified seed
- Injects noise + clustered cloud gaps
- Runs crop-type classification → growth-stage estimation → water-deficit advisory
- Prints full confusion matrices and error rates

---

## 🏗️ Project Structure

```
KrishiDrishti/
├── run.py                          # One-command entry point
├── simulator/
│   ├── crop_simulator.py           # Parametric crop-growth-curve simulator (≥4 crops)
│   └── noise_injector.py           # Noise & cloud-gap injection (Markov-chain)
├── scripts/
│   ├── train_all_models.py         # Full ML training pipeline
│   ├── train_stage_estimator.py    # Growth-stage estimator (Deliverable #6)
│   ├── advisory_engine.py          # Water-deficit advisory engine (Deliverable #7)
│   ├── robustness_demo.py          # Noise/gap degradation analysis
│   └── evaluate_models.py          # Model evaluation & PDF report
├── dashboard/
│   └── app.py                      # Streamlit interactive dashboard
├── utils/
│   └── config.py                   # All parameters, thresholds, crop definitions
├── models/
│   ├── saved/                      # Trained model artifacts (.joblib, .pt)
│   └── production/                 # Selected best models for deployment
├── outputs/
│   └── evaluation/                 # Plots, metrics, PDF reports
└── gee_scripts/                    # (Reference) Google Earth Engine scripts
```

---

## 📊 Measured Accuracy & Error Rates

### Crop-Type Classifier (on noisy, gapped simulated data)

| Model | Accuracy | F1 (weighted) | Kappa |
|---|---|---|---|
| Random Forest (500 trees) | 0.9600 | 0.9607 | 0.9462 |
| XGBoost (300 est.) | 0.9300 | 0.9297 | 0.9055 |

**Confusion Matrix (RF, seed=99):**
|  | Rice | Wheat | Cotton | Sugarcane |
|---|---|---|---|---|
| **Rice** | 6 | 1 | 0 | 0 |
| **Wheat** | 0 | 4 | 0 | 0 |
| **Cotton** | 0 | 0 | 7 | 0 |
| **Sugarcane** | 0 | 0 | 0 | 7 |

### Growth-Stage Estimator (evaluated SEPARATELY from crop-type)

| Metric | Value |
|---|---|
| Accuracy | 0.8242 |
| F1 (weighted) | 0.8237 |
| Cohen's Kappa | 0.7663 |
| CV (5-fold) | 0.8572 ± 0.0045 |

**Per-Stage Performance:**

| Stage | Precision | Recall | F1 |
|---|---|---|---|
| Sowing | 0.86 | 0.80 | 0.83 |
| Vegetative | 0.88 | 0.91 | 0.89 |
| Flowering | 0.78 | 0.80 | 0.79 |
| Maturity | 0.81 | 0.80 | 0.80 |
| Harvest | 0.79 | 0.74 | 0.76 |

### Water-Deficit Advisory Engine

| Metric | Value |
|---|---|
| False Alarm Rate | 54.29% |
| Missed Stress Rate | 53.48% |
| Decision Logic | FAO-56 ETc = Kc × ETo, Deficit = ETc − Pe |

> **Why are advisory error rates ~50%?** The FAO-56 water balance computes deficit from weather (ETo, rainfall) and crop water demand (Kc). The simulator injects stress *independently* of weather — a plot can be weather-stressed but not simulator-stressed, or vice versa. This is a transparent, honest measurement showing that weather-derived advisories and vegetation-index-derived stress indicators capture *different aspects* of crop water status.

### Robustness Analysis

| Noise σ | Accuracy | F1 |
|---|---|---|
| 0.01 | 0.9900 | 0.9900 |
| 0.03 | 0.9300 | 0.9297 |
| 0.05 | 0.8900 | 0.8901 |
| 0.07 | 0.7800 | 0.7812 |
| 0.10 | 0.6800 | 0.6781 |

| Gap-Dropout Rate | Accuracy | F1 |
|---|---|---|
| 5% | 0.9300 | 0.9307 |
| 15% | 0.9300 | 0.9297 |
| 25% | 0.8800 | 0.8811 |
| 40% | 0.8600 | 0.8607 |
| 60% | 0.7800 | 0.7686 |

---

## 🔬 Honesty Section — What the Simulator Does and Doesn't Capture

### What the simulator DOES model:
- ✅ **Crop-specific growth dynamics**: Double-logistic curves with documented agronomic parameters (emergence timing, peak NDVI, senescence rate) for 4 crops (Rice, Wheat, Cotton, Sugarcane)
- ✅ **Water stress effect**: Stage-sensitive NDVI reduction with trapezoidal ramp-up/down envelope
- ✅ **Sensor noise**: Gaussian radiometric noise (configurable σ, default 0.03)
- ✅ **Clustered cloud gaps**: Markov-chain model with monsoon-boosted gap probability (not uniformly random)
- ✅ **Satellite revisit cadence**: 5-day observation interval
- ✅ **Weather patterns**: Sinusoidal ETo with gamma-distributed rainfall and wet-season boosting
- ✅ **Sowing date variability**: Randomized within crop-specific planting windows
- ✅ **Duration variability**: Normal distribution around documented crop durations

### What the simulator DOES NOT capture:
- ❌ **Spatial autocorrelation**: Real satellite pixels have spatial patterns (neighboring fields tend to grow the same crop); our simulator treats plots independently
- ❌ **Mixed pixels**: Real 10m Sentinel-2 pixels can span field boundaries, mixing spectral signatures from different crops
- ❌ **Atmospheric effects**: Real data has aerosol contamination, BRDF effects, and adjacency effects that go beyond simple Gaussian noise
- ❌ **Soil background variability**: Real bare-soil NDVI varies with soil type, moisture, and tillage — we use a fixed base NDVI per crop
- ❌ **Multi-band spectral information**: We simulate only a single vegetation index (NDVI); real classification uses EVI, NDWI, NDMI, SAR backscatter, etc.
- ❌ **Inter-annual variability**: All simulations are single-season; real data spans multiple years with climate variability
- ❌ **Real SAR data**: Sentinel-1 SAR backscatter physics are not simulated
- ❌ **Field geometry**: No realistic field shapes, boundaries, or landscape context

### Why no real dataset was used:
As stated in the problem statement: *"Sourcing, licensing, and validating live satellite or national irrigation datasets is not achievable inside a hackathon build window."* Our simulator is a transparent, documented alternative that allows rigorous testing under controlled conditions.

---

## 📋 Mandatory Deliverables Checklist

| # | Deliverable | Status | Location |
|---|---|---|---|
| 1 | One-command deploy/run | ✅ | `python run.py --seed 123` |
| 2 | Parametric simulator (≥4 crops) | ✅ | `simulator/crop_simulator.py` |
| 3 | Noise & cloud-gap module | ✅ | `simulator/noise_injector.py` |
| 4 | Growth-stage ground-truth generator | ✅ | `simulator/crop_simulator.py` (line 146) |
| 5 | Crop-type classifier + CM | ✅ | `scripts/train_all_models.py`, `outputs/evaluation/` |
| 6 | Growth-stage estimator | ✅ | `scripts/train_stage_estimator.py` |
| 7 | Water-deficit advisory + error rates | ✅ | `scripts/advisory_engine.py` |
| 8 | Live dashboard with disclosure banner | ✅ | `dashboard/app.py` |
| 9 | README with honesty section | ✅ | This file |

---

## 👥 Team BEST SHOT
*ROBOVANTA Hackathon | Problem Statement 06*
