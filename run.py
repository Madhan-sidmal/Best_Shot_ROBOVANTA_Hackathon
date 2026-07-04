"""
KrishiDrishti — One-Command Run
==================================
Generates a fresh simulated season on any random seed, runs the
full pipeline (simulator -> noise -> crop classifier -> stage
estimator -> advisory engine), prints all results, and optionally
launches the Streamlit dashboard.

Usage:
    python run.py                    # Random seed, no dashboard
    python run.py --seed 123         # Fixed seed
    python run.py --seed 123 --dash  # Fixed seed + launch dashboard
    python run.py --plots 200        # Custom plot count

DISCLAIMER: All data is synthetically generated.
"""

import os
import sys
import json
import argparse
import time
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, cohen_kappa_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from simulator.crop_simulator import CropGrowthSimulator
from simulator.noise_injector import NoiseInjector
from utils.config import (
    CROP_PARAMS, STAGE_NAMES, ADVISORY_CONFIG,
    NOISE_CONFIG, SIMULATION
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================
def extract_crop_features(plots):
    """Extract per-plot features for crop classification."""
    records = []
    for plot in plots:
        obs = plot.observed_ndvi.copy()
        mask = plot.valid_mask
        days = plot.days

        # Interpolate gaps
        nan_idx = np.isnan(obs)
        if nan_idx.any() and (~nan_idx).any():
            obs[nan_idx] = np.interp(days[nan_idx], days[~nan_idx], obs[~nan_idx])

        valid_obs = obs[~np.isnan(obs)]
        if len(valid_obs) == 0:
            valid_obs = np.array([0.0])

        records.append({
            'plot_id': plot.plot_id,
            'crop_label': plot.crop_label,
            'crop_type': plot.crop_type,
            'ndvi_mean': np.mean(valid_obs),
            'ndvi_max': np.max(valid_obs),
            'ndvi_min': np.min(valid_obs),
            'ndvi_std': np.std(valid_obs),
            'ndvi_range': np.max(valid_obs) - np.min(valid_obs),
            'ndvi_median': np.median(valid_obs),
            'ndvi_q25': np.percentile(valid_obs, 25),
            'ndvi_q75': np.percentile(valid_obs, 75),
            'green_up_rate': np.max(np.diff(valid_obs)) if len(valid_obs) > 1 else 0,
            'senescence_rate': np.min(np.diff(valid_obs)) if len(valid_obs) > 1 else 0,
            'gap_fraction': float((~mask).sum()) / len(mask),
        })
    return pd.DataFrame(records)


def extract_stage_features(plots, window=5):
    """Extract per-timestep features for stage estimation."""
    records = []
    for plot in plots:
        if not hasattr(plot, 'observed_ndvi'):
            continue
        obs = plot.observed_ndvi.copy()
        days = plot.days
        mask = plot.valid_mask

        obs_interp = obs.copy()
        nan_idx = np.isnan(obs_interp)
        if nan_idx.any() and (~nan_idx).any():
            obs_interp[nan_idx] = np.interp(
                days[nan_idx], days[~nan_idx], obs_interp[~nan_idx])

        season_len = days[-1] - days[0] + 1

        for i in range(len(days)):
            stage_id = plot.stage_labels[i]
            if stage_id < 0:
                continue
            ndvi_val = obs_interp[i]
            day_frac = days[i] / season_len
            start = max(0, i - window)
            window_vals = obs_interp[start:i+1]
            records.append({
                'plot_id': plot.plot_id,
                'ndvi': ndvi_val,
                'day_frac': day_frac,
                'roll_mean': np.nanmean(window_vals),
                'roll_std': np.nanstd(window_vals) if len(window_vals) > 1 else 0.0,
                'derivative': (obs_interp[i] - obs_interp[i-1]) / max(days[i] - days[i-1], 1) if i > 0 else 0.0,
                'cum_sum': np.nansum(obs_interp[:i+1]),
                'max_so_far': np.nanmax(obs_interp[:i+1]),
                'days_since_valid': 0 if mask[i] else (days[i] - days[np.where(mask[:i])[0][-1]] if np.any(mask[:i]) else days[i]),
                'is_gap': 0 if mask[i] else 1,
                'stage_id': stage_id,
            })
    return pd.DataFrame(records)


def compute_advisory(plots, advisory_period_days=8):
    """Compute FAO-56 water balance advisory."""
    cfg = ADVISORY_CONFIG
    thresholds = cfg['thresholds']
    pe_factor = cfg['effective_rainfall_factor']
    records = []

    for plot in plots:
        crop = plot.crop_type
        p = CROP_PARAMS[crop]
        kc_values = p['kc']
        stage_keys = list(p['stages'].keys())
        days = plot.days
        n_steps = len(days)

        period_start = 0
        while period_start < n_steps:
            period_end = min(period_start + advisory_period_days, n_steps)
            stages_in_period = plot.stage_labels[period_start:period_end]
            valid_stages = stages_in_period[stages_in_period >= 0]
            if len(valid_stages) == 0:
                period_start = period_end
                continue

            dominant_stage_id = min(int(np.median(valid_stages)), len(stage_keys) - 1)
            stage_name = stage_keys[dominant_stage_id]
            kc = kc_values.get(stage_name, 0.5)

            eto_sum = np.sum(plot.eto[period_start:period_end])
            rain_sum = np.sum(plot.rainfall[period_start:period_end])
            etc = kc * eto_sum
            pe = pe_factor * rain_sum
            deficit = etc - pe

            if deficit <= thresholds['adequate']:
                advisory = 0
            elif deficit <= thresholds['irrigate_soon']:
                advisory = 1
            else:
                advisory = 2

            stress_signal = plot.stress_signal[period_start:period_end]
            gt_stressed = bool(np.any(stress_signal > 0.1))

            records.append({
                'plot_id': plot.plot_id,
                'crop': CROP_PARAMS[crop]['display_name'],
                'stage': stage_name,
                'deficit_mm': float(deficit),
                'advisory': advisory,
                'gt_stressed': gt_stressed,
            })
            period_start = period_end

    return pd.DataFrame(records)


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='KrishiDrishti — Full Pipeline Run')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (default: random)')
    parser.add_argument('--plots', type=int, default=200,
                        help='Number of simulated plots (default: 200)')
    parser.add_argument('--dash', action='store_true',
                        help='Launch Streamlit dashboard after pipeline')
    args = parser.parse_args()

    seed = args.seed or np.random.randint(1, 100000)

    print()
    print("=" * 70)
    print("  🛰️  KrishiDrishti — Full End-to-End Pipeline")
    print("=" * 70)
    print(f"  Seed: {seed} | Plots: {args.plots}")
    print(f"  ⚠️  ALL DATA IS SYNTHETICALLY GENERATED")
    print("=" * 70)

    t0 = time.time()

    # ---- STEP 1: Simulate ----
    print(f"\n{'='*50}")
    print("  STEP 1 — Parametric Crop Growth Simulation")
    print(f"{'='*50}")
    sim = CropGrowthSimulator(seed=seed)
    plots = sim.simulate_season(num_plots=args.plots, seed=seed)

    # ---- STEP 2: Noise + Gaps ----
    print(f"\n{'='*50}")
    print("  STEP 2 — Noise & Cloud-Gap Injection")
    print(f"{'='*50}")
    injector = NoiseInjector(seed=seed)
    plots = injector.process_all_plots(plots)
    gap_stats = injector.get_gap_statistics(plots)
    print(f"   Overall gap rate: {gap_stats['gap_percentage']:.1f}%")
    print(f"   Monsoon gap rate: {gap_stats['monsoon_gap_percentage']:.1f}%")

    # ---- STEP 3: Crop Classification ----
    print(f"\n{'='*50}")
    print("  STEP 3 — Crop-Type Classification")
    print(f"{'='*50}")
    model_path = os.path.join(BASE_DIR, 'models', 'saved', 'crop_rf.joblib')
    if os.path.exists(model_path):
        data = joblib.load(model_path)
        crop_model = data['model']
        crop_scaler = data['scaler']
        crop_features = data['features']
        print("   Loaded pre-trained model: crop_rf.joblib")

        df_crop = extract_crop_features(plots)
        available_features = [f for f in crop_features if f in df_crop.columns]
        n_expected = crop_scaler.n_features_in_

        if len(available_features) == n_expected:
            X = df_crop[available_features].values.astype(np.float32)
            X = np.nan_to_num(X)
            X_s = crop_scaler.transform(X)
            pred = crop_model.predict(X_s)
            gt = df_crop['crop_label'].values
            acc = accuracy_score(gt, pred)
            f1 = f1_score(gt, pred, average='weighted')
            kappa = cohen_kappa_score(gt, pred)
        else:
            # Fallback: train a quick RF from scratch
            print("   Feature mismatch — training fresh RF on this seed's data...")
            feature_cols = ['ndvi_mean', 'ndvi_max', 'ndvi_min', 'ndvi_std',
                            'ndvi_range', 'ndvi_median', 'ndvi_q25', 'ndvi_q75',
                            'green_up_rate', 'senescence_rate', 'gap_fraction']
            X = df_crop[feature_cols].values.astype(np.float32)
            y = df_crop['crop_label'].values
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_te_s = sc.transform(X_te)
            rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=seed, n_jobs=-1)
            rf.fit(X_tr_s, y_tr)
            pred = rf.predict(X_te_s)
            gt = y_te
            acc = accuracy_score(gt, pred)
            f1 = f1_score(gt, pred, average='weighted')
            kappa = cohen_kappa_score(gt, pred)
    else:
        print("   No pre-trained model found — training fresh RF...")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        df_crop = extract_crop_features(plots)
        feature_cols = ['ndvi_mean', 'ndvi_max', 'ndvi_min', 'ndvi_std',
                        'ndvi_range', 'ndvi_median', 'ndvi_q25', 'ndvi_q75',
                        'green_up_rate', 'senescence_rate', 'gap_fraction']
        X = df_crop[feature_cols].values.astype(np.float32)
        y = df_crop['crop_label'].values
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)
        rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=seed, n_jobs=-1)
        rf.fit(X_tr_s, y_tr)
        pred = rf.predict(X_te_s)
        gt = y_te
        acc = accuracy_score(gt, pred)
        f1 = f1_score(gt, pred, average='weighted')
        kappa = cohen_kappa_score(gt, pred)

    crop_names = [CROP_PARAMS[k]['display_name'] for k in sorted(CROP_PARAMS.keys(), key=lambda x: CROP_PARAMS[x]['label'])]
    cm = confusion_matrix(gt, pred)
    print(f"\n   Crop Classification Results:")
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   F1 (wt):   {f1:.4f}")
    print(f"   Kappa:     {kappa:.4f}")
    print(f"\n   Confusion Matrix:")
    print(f"   {'':>15s}  " + "  ".join(f"{n[:6]:>6s}" for n in crop_names))
    for i, row in enumerate(cm):
        print(f"   {crop_names[i]:>15s}  " + "  ".join(f"{v:6d}" for v in row))

    # ---- STEP 4: Growth-Stage Estimation ----
    print(f"\n{'='*50}")
    print("  STEP 4 — Growth-Stage Estimation")
    print(f"{'='*50}")
    stage_model_path = os.path.join(BASE_DIR, 'models', 'saved', 'stage_estimator.joblib')
    if os.path.exists(stage_model_path):
        data = joblib.load(stage_model_path)
        stage_model = data['model']
        stage_scaler = data['scaler']
        stage_features = data['features']
        print("   Loaded pre-trained model: stage_estimator.joblib")

        df_stage = extract_stage_features(plots)
        X_stage = df_stage[stage_features].values.astype(np.float32)
        X_stage = np.nan_to_num(X_stage)
        y_stage_gt = df_stage['stage_id'].values
        X_stage_s = stage_scaler.transform(X_stage)
        y_stage_pred = stage_model.predict(X_stage_s)

        stage_acc = accuracy_score(y_stage_gt, y_stage_pred)
        stage_f1 = f1_score(y_stage_gt, y_stage_pred, average='weighted')
        stage_kappa = cohen_kappa_score(y_stage_gt, y_stage_pred)
    else:
        print("   No stage model found — skipping (run train_stage_estimator.py first)")
        stage_acc = stage_f1 = stage_kappa = 0.0

    print(f"\n   Growth-Stage Results:")
    print(f"   Accuracy:  {stage_acc:.4f}")
    print(f"   F1 (wt):   {stage_f1:.4f}")
    print(f"   Kappa:     {stage_kappa:.4f}")

    if stage_acc > 0:
        stage_cm = confusion_matrix(y_stage_gt, y_stage_pred)
        print(f"\n   Confusion Matrix:")
        print(f"   {'':>12s}  " + "  ".join(f"{n[:5]:>5s}" for n in STAGE_NAMES))
        for i, row in enumerate(stage_cm):
            print(f"   {STAGE_NAMES[i]:>12s}  " + "  ".join(f"{v:5d}" for v in row))

    # ---- STEP 5: Water-Deficit Advisory ----
    print(f"\n{'='*50}")
    print("  STEP 5 — Water-Deficit Advisory (FAO-56)")
    print(f"{'='*50}")
    df_adv = compute_advisory(plots)
    advisory_names = ['Adequate', 'Irrigate Soon', 'Stress Detected']

    pred_adv = df_adv['advisory'].values
    gt_stressed = df_adv['gt_stressed'].values.astype(int)
    pred_stressed = (pred_adv > 0).astype(int)

    fa_mask = gt_stressed == 0
    fa_rate = np.sum(pred_stressed[fa_mask] == 1) / max(fa_mask.sum(), 1)
    ms_mask = gt_stressed == 1
    ms_rate = np.sum(pred_stressed[ms_mask] == 0) / max(ms_mask.sum(), 1)

    print(f"\n   Advisory Distribution:")
    for i, name in enumerate(advisory_names):
        count = (pred_adv == i).sum()
        print(f"     {name}: {count}")
    print(f"\n   False Alarm Rate:    {fa_rate:.2%}")
    print(f"   Missed Stress Rate:  {ms_rate:.2%}")

    elapsed = time.time() - t0

    # ---- SUMMARY ----
    print(f"\n{'='*70}")
    print("  🏆 PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"  Seed: {seed} | Plots: {args.plots} | Time: {elapsed:.1f}s")
    print(f"  Crop Classification:    Acc={acc:.4f}, F1={f1:.4f}, Kappa={kappa:.4f}")
    print(f"  Growth-Stage Estimation: Acc={stage_acc:.4f}, F1={stage_f1:.4f}, Kappa={stage_kappa:.4f}")
    print(f"  Advisory: FA={fa_rate:.2%}, MS={ms_rate:.2%}")
    print(f"  ⚠️  ALL DATA IS SYNTHETICALLY GENERATED")
    print(f"{'='*70}")

    # Launch dashboard if requested
    if args.dash:
        print("\n  Launching Streamlit dashboard...")
        os.system(f'streamlit run dashboard/app.py')


if __name__ == "__main__":
    main()
