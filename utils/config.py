"""
KrishiDrishti Configuration & Constants
=========================================
Central configuration for the SIMULATED satellite-index
crop classification & growth-stage-aware water-deficit advisory.

NOTE: All data in this project is synthetically generated.
No real satellite imagery or government datasets are used.
"""

import os
import numpy as np

# ============================================================
# CROP DEFINITIONS — Parametric Growth Curve Parameters
# ============================================================
# Each crop has a characteristic NDVI growth curve modeled as a
# double-logistic function with documented agronomic parameters.
#
# Growth curve: NDVI(t) = base + amplitude * f(t)
# where f(t) is shaped by emergence, peak timing, senescence rate.
#
# References for phenological durations:
#   - FAO Crop Calendar (approximate durations)
#   - Allen et al., 1998 (FAO-56) for Kc stage lengths
CROP_PARAMS = {
    "rice": {
        "label": 0,
        "display_name": "Rice (Paddy)",
        "color": "#228B22",
        # NDVI curve parameters
        "base_ndvi": 0.12,           # bare/flooded soil NDVI
        "peak_ndvi": 0.82,           # peak canopy NDVI
        "emergence_day": 10,          # days after sowing to emergence
        "green_up_rate": 0.06,        # NDVI increase per day during green-up
        "peak_day_frac": 0.55,        # fraction of total duration at peak
        "senescence_rate": 0.04,      # NDVI decrease per day during senescence
        "total_duration_days": 135,   # total crop duration
        "duration_std": 10,           # variation in duration (days)
        "sowing_window": (0, 30),     # sowing day range (relative to season start)
        # Growth stage boundaries (fraction of total duration)
        "stages": {
            "sowing":     (0.00, 0.07),
            "vegetative": (0.07, 0.40),
            "flowering":  (0.40, 0.65),
            "maturity":   (0.65, 0.90),
            "harvest":    (0.90, 1.00),
        },
        # Water stress sensitivity by stage (higher = more yield impact)
        "stress_sensitivity": {
            "sowing": 0.3, "vegetative": 0.5,
            "flowering": 1.0, "maturity": 0.6, "harvest": 0.2
        },
        # FAO-56 Kc values
        "kc": {"sowing": 1.05, "vegetative": 1.10, "flowering": 1.20,
               "maturity": 0.90, "harvest": 0.60},
    },
    "wheat": {
        "label": 1,
        "display_name": "Wheat",
        "color": "#DAA520",
        "base_ndvi": 0.10,
        "peak_ndvi": 0.75,
        "emergence_day": 8,
        "green_up_rate": 0.05,
        "peak_day_frac": 0.50,
        "senescence_rate": 0.045,
        "total_duration_days": 145,
        "duration_std": 12,
        "sowing_window": (0, 25),
        "stages": {
            "sowing":     (0.00, 0.06),
            "vegetative": (0.06, 0.38),
            "flowering":  (0.38, 0.60),
            "maturity":   (0.60, 0.88),
            "harvest":    (0.88, 1.00),
        },
        "stress_sensitivity": {
            "sowing": 0.2, "vegetative": 0.6,
            "flowering": 1.0, "maturity": 0.5, "harvest": 0.1
        },
        "kc": {"sowing": 0.40, "vegetative": 0.80, "flowering": 1.15,
               "maturity": 0.70, "harvest": 0.25},
    },
    "cotton": {
        "label": 2,
        "display_name": "Cotton",
        "color": "#FF8C00",
        "base_ndvi": 0.08,
        "peak_ndvi": 0.70,
        "emergence_day": 12,
        "green_up_rate": 0.035,
        "peak_day_frac": 0.50,
        "senescence_rate": 0.025,
        "total_duration_days": 180,
        "duration_std": 15,
        "sowing_window": (5, 35),
        "stages": {
            "sowing":     (0.00, 0.06),
            "vegetative": (0.06, 0.35),
            "flowering":  (0.35, 0.60),
            "maturity":   (0.60, 0.85),
            "harvest":    (0.85, 1.00),
        },
        "stress_sensitivity": {
            "sowing": 0.3, "vegetative": 0.7,
            "flowering": 1.0, "maturity": 0.4, "harvest": 0.1
        },
        "kc": {"sowing": 0.35, "vegetative": 0.70, "flowering": 1.15,
               "maturity": 0.70, "harvest": 0.40},
    },
    "sugarcane": {
        "label": 3,
        "display_name": "Sugarcane",
        "color": "#9370DB",
        "base_ndvi": 0.14,
        "peak_ndvi": 0.85,
        "emergence_day": 15,
        "green_up_rate": 0.03,
        "peak_day_frac": 0.55,
        "senescence_rate": 0.015,
        "total_duration_days": 300,
        "duration_std": 20,
        "sowing_window": (0, 40),
        "stages": {
            "sowing":     (0.00, 0.05),
            "vegetative": (0.05, 0.35),
            "flowering":  (0.35, 0.60),
            "maturity":   (0.60, 0.90),
            "harvest":    (0.90, 1.00),
        },
        "stress_sensitivity": {
            "sowing": 0.4, "vegetative": 0.8,
            "flowering": 0.9, "maturity": 0.5, "harvest": 0.1
        },
        "kc": {"sowing": 0.40, "vegetative": 0.75, "flowering": 1.25,
               "maturity": 0.75, "harvest": 0.50},
    },
}

CROP_NAMES = {v["label"]: v["display_name"] for v in CROP_PARAMS.values()}
NUM_CROPS = len(CROP_PARAMS)
STAGE_NAMES = ["sowing", "vegetative", "flowering", "maturity", "harvest"]
NUM_STAGES = len(STAGE_NAMES)

# ============================================================
# SIMULATION PARAMETERS
# ============================================================
SIMULATION = {
    "season_length_days": 365,          # full season to simulate
    "observation_interval_days": 5,     # simulated satellite revisit (days)
    "num_plots": 200,                   # number of simulated plots/fields
    "default_seed": 42,                 # default random seed

    # Water stress simulation
    "stress_probability": 0.35,         # fraction of plots experiencing stress
    "stress_onset_range": (0.25, 0.70), # when stress starts (fraction of season)
    "stress_duration_range": (15, 60),  # stress duration in days
    "stress_intensity_range": (0.15, 0.45),  # NDVI reduction factor

    # Reference ET (mm/day) — sinusoidal seasonal pattern
    "eto_mean": 4.5,
    "eto_amplitude": 1.5,
    "eto_peak_day_frac": 0.4,          # peak ETo at 40% through season (summer)

    # Rainfall (mm/day) — gamma-distributed with seasonal pattern
    "rainfall_mean": 3.0,
    "rainfall_shape": 0.8,             # gamma shape (< 1 = heavy-tailed)
    "rainfall_wet_season_frac": (0.2, 0.6),  # wet season window
}

# ============================================================
# NOISE & CLOUD-GAP INJECTION
# ============================================================
NOISE_CONFIG = {
    # Sensor noise
    "noise_std": 0.03,                 # Gaussian noise σ for NDVI (default)
    "noise_std_range": (0.01, 0.08),   # configurable range

    # Cloud-gap injection (clustered, not uniform)
    "gap_probability": 0.25,           # base probability of a gap at any time step
    "gap_cluster_size_range": (2, 6),  # consecutive missing observations in a cluster
    "gap_cluster_probability": 0.15,   # probability of starting a new gap cluster
    "monsoon_gap_boost": 2.5,          # multiplier during wet season
    "monsoon_window_frac": (0.15, 0.55),  # wet season window as fraction of season
}

# ============================================================
# CLASSIFIER PARAMETERS
# ============================================================
CLASSIFIER_CONFIG = {
    "random_forest": {
        "n_estimators": 500,
        "max_depth": 20,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    },
    "xgboost": {
        "n_estimators": 300,
        "max_depth": 8,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0.1,
        "random_state": 42,
        "eval_metric": "mlogloss",
    },
    "test_size": 0.3,
    "cv_folds": 5,
}

# ============================================================
# GROWTH-STAGE ESTIMATOR PARAMETERS
# ============================================================
STAGE_ESTIMATOR_CONFIG = {
    "model_type": "random_forest",  # or "gradient_boosting"
    "n_estimators": 300,
    "max_depth": 15,
    "random_state": 42,
}

# ============================================================
# WATER-DEFICIT ADVISORY
# ============================================================
ADVISORY_CONFIG = {
    "effective_rainfall_factor": 0.8,
    "advisory_period_days": 8,
    "thresholds": {
        # deficit in mm / advisory_period_days
        "adequate": 5,
        "irrigate_soon": 15,
        "stress_detected": 30,
    },
    "classes": {
        0: {"name": "Adequate",        "color": "#00AA00", "emoji": "🟢",
            "message": "No irrigation needed. Soil moisture is sufficient."},
        1: {"name": "Irrigate Soon",   "color": "#FFDD00", "emoji": "🟡",
            "message": "Schedule irrigation within {days} days."},
        2: {"name": "Stress Detected", "color": "#FF0000", "emoji": "🔴",
            "message": "Water stress detected. Immediate irrigation recommended."},
    },
}

# ============================================================
# DASHBOARD
# ============================================================
DASHBOARD = {
    "title": "🛰️ KrishiDrishti — Simulated Crop Intelligence Dashboard",
    "subtitle": "⚠️ ALL DATA IS SYNTHETICALLY GENERATED — NOT FROM REAL SATELLITES",
    "page_icon": "🛰️",
    "layout": "wide",
    "disclosure_banner": (
        "⚠️ SIMULATED DATA DISCLAIMER: All vegetation/moisture-index data, "
        "crop-growth curves, and water-stress signals displayed in this dashboard "
        "are synthetically generated by the team's parametric simulator. "
        "No real satellite imagery, ISRO/Bhuvan data, or government datasets are used. "
        "This output must NOT be presented as a validated crop-monitoring or "
        "irrigation-advisory product."
    ),
}

# ============================================================
# FILE PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models", "saved")

for d in [OUTPUT_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)
