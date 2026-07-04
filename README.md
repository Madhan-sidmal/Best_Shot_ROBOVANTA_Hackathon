# 🛰️ KrishiDrishti — AI-Driven Crop Intelligence Platform

> **Problem Statement 06**
> AI-Driven Automated Crop Type, Moisture Stress Detection & Irrigation Advisory

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![GEE](https://img.shields.io/badge/Google_Earth_Engine-Enabled-green.svg)](https://earthengine.google.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🌾 Overview

**KrishiDrishti** (कृषिदृष्टि — "Agricultural Vision") is an end-to-end AI pipeline that leverages multi-temporal optical and microwave satellite data to:

1. **Classify crop types** using ensemble ML (Random Forest + XGBoost)
2. **Detect phenology-aware moisture stress** using LSTM + Attention models
3. **Generate 8-day irrigation advisories** using FAO-56 crop water balance

## 🏗️ Architecture

```
KrishiDrishti/
├── gee_scripts/                 # Google Earth Engine scripts
│   ├── 01_data_ingestion.py     # Satellite data loading & compositing
│   ├── 02_feature_engineering.py # Index computation & feature extraction
│   └── 03_export_features.py    # Export processed data to Drive/Colab
├── models/                      # ML/DL model training
│   ├── crop_classifier.py       # RF + XGBoost ensemble classification
│   ├── phenology_extractor.py   # Growth stage detection from NDVI
│   ├── stress_detector.py       # LSTM + Attention for stress detection
│   └── irrigation_advisory.py   # FAO-56 water balance & advisory
├── dashboard/                   # Streamlit dashboard
│   ├── app.py                   # Main dashboard application
│   ├── components/              # UI components
│   └── assets/                  # Static assets, colormaps
├── utils/                       # Shared utilities
│   ├── config.py                # Configuration & constants
│   ├── geo_utils.py             # Geospatial utilities
│   └── alert_service.py         # SMS/WhatsApp notification service
├── data/                        # Data directory (gitignored)
│   ├── ground_truth/            # Ground truth labels
│   ├── rasters/                 # Exported rasters
│   └── ancillary/               # Meteorological, soil, command area
├── notebooks/                   # Jupyter/Colab notebooks
│   └── KrishiDrishti_Pipeline.ipynb
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Google Earth Engine account (authenticated)
- Google Colab (recommended for GPU training)

### Installation
```bash
pip install -r requirements.txt
earthengine authenticate
```

### Run Dashboard
```bash
streamlit run dashboard/app.py
```

## 📊 Key Features
- 🌾 Multi-temporal crop classification (>85% accuracy target)
- 💧 Stage-wise moisture stress detection
- 🚿 8-day irrigation advisory maps for canal command areas
- 📱 SMS/WhatsApp irrigation alerts
- 🧠 Explainable AI (SHAP) for model transparency
- 🛰️ NISAR-ready architecture
- 📈 Interactive dashboard with time-series visualization

## 👥 Team BEST SHOT

