"""
Robustness Demo — Expected by Judges
=========================================
Runs the crop classifier at multiple noise levels and gap-dropout
rates, then plots accuracy degradation curves.

Proves the system isn't overfit to a clean synthetic case.

DISCLAIMER: All data is synthetically generated.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from simulator.crop_simulator import CropGrowthSimulator
from simulator.noise_injector import NoiseInjector
from utils.config import CROP_PARAMS

EVAL_DIR = os.path.join(BASE_DIR, "outputs", "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


def extract_features_from_plots(plots, noise_std, gap_rate, seed=42):
    """
    Given raw plots, inject noise at specified levels and extract
    classification features from the observed (noisy+gapped) data.
    """
    injector = NoiseInjector(seed=seed)
    plots = injector.process_all_plots(plots, noise_std=noise_std,
                                        gap_dropout_rate=gap_rate)

    records = []
    for plot in plots:
        obs = plot.observed_ndvi.copy()
        mask = plot.valid_mask
        days = plot.days

        # Interpolate gaps
        nan_idx = np.isnan(obs)
        if nan_idx.any() and (~nan_idx).any():
            obs[nan_idx] = np.interp(days[nan_idx], days[~nan_idx], obs[~nan_idx])

        # Aggregate features per plot (what the classifier uses)
        valid_obs = obs[~np.isnan(obs)]
        if len(valid_obs) == 0:
            valid_obs = np.array([0.0])

        records.append({
            'crop_label': plot.crop_label,
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
            'n_valid_obs': int(mask.sum()),
        })

    return pd.DataFrame(records)


def main():
    print("=" * 60)
    print("  KrishiDrishti — Robustness Demo")
    print("  Classifier degradation under noise & gaps")
    print("=" * 60)

    # Generate base plots (clean, no noise yet)
    print("\n[1] Generating 400 clean simulated plots...")
    sim = CropGrowthSimulator(seed=42)
    plots = sim.simulate_season(num_plots=400, seed=42)

    # Test configurations
    noise_levels = [0.01, 0.03, 0.05, 0.07, 0.10]
    gap_rates = [0.05, 0.15, 0.25, 0.40, 0.60]

    feature_cols = ['ndvi_mean', 'ndvi_max', 'ndvi_min', 'ndvi_std',
                    'ndvi_range', 'ndvi_median', 'ndvi_q25', 'ndvi_q75',
                    'green_up_rate', 'senescence_rate', 'gap_fraction', 'n_valid_obs']

    # --- Experiment 1: Vary noise, fixed gap rate ---
    print("\n[2] Experiment 1: Varying noise sigma (fixed gap=0.15)...")
    noise_results = []
    for sigma in noise_levels:
        # Re-create plots each time (they get modified by noise injection)
        fresh_plots = sim.simulate_season(num_plots=400, seed=42)
        df = extract_features_from_plots(fresh_plots, noise_std=sigma, gap_rate=0.15)

        X = df[feature_cols].values.astype(np.float32)
        y = df['crop_label'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=42)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = RandomForestClassifier(n_estimators=200, max_depth=15,
                                        random_state=42, n_jobs=-1)
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)

        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average='weighted')
        noise_results.append({'sigma': sigma, 'accuracy': acc, 'f1': f1})
        print(f"    sigma={sigma:.2f}: Acc={acc:.4f}, F1={f1:.4f}")

    # --- Experiment 2: Vary gap rate, fixed noise ---
    print("\n[3] Experiment 2: Varying gap-dropout rate (fixed noise=0.03)...")
    gap_results = []
    for gap in gap_rates:
        fresh_plots = sim.simulate_season(num_plots=400, seed=42)
        df = extract_features_from_plots(fresh_plots, noise_std=0.03, gap_rate=gap)

        X = df[feature_cols].values.astype(np.float32)
        y = df['crop_label'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=42)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = RandomForestClassifier(n_estimators=200, max_depth=15,
                                        random_state=42, n_jobs=-1)
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)

        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average='weighted')
        gap_results.append({'gap_rate': gap, 'accuracy': acc, 'f1': f1})
        print(f"    gap_rate={gap:.2f}: Acc={acc:.4f}, F1={f1:.4f}")

    # --- Plot degradation curves ---
    print("\n[4] Plotting degradation curves (300 DPI)...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Noise degradation
    nr = pd.DataFrame(noise_results)
    axes[0].plot(nr['sigma'], nr['accuracy'], 'o-', color='#2196F3',
                 linewidth=2, markersize=8, label='Accuracy')
    axes[0].plot(nr['sigma'], nr['f1'], 's--', color='#FF9800',
                 linewidth=2, markersize=8, label='F1 (weighted)')
    axes[0].set_xlabel('Noise σ (Gaussian)', fontsize=12)
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title('Classifier Degradation vs Noise Level', fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.05)

    # Gap degradation
    gr = pd.DataFrame(gap_results)
    axes[1].plot(gr['gap_rate'] * 100, gr['accuracy'], 'o-', color='#4CAF50',
                 linewidth=2, markersize=8, label='Accuracy')
    axes[1].plot(gr['gap_rate'] * 100, gr['f1'], 's--', color='#E91E63',
                 linewidth=2, markersize=8, label='F1 (weighted)')
    axes[1].set_xlabel('Gap-Dropout Rate (%)', fontsize=12)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('Classifier Degradation vs Cloud-Gap Rate', fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)

    plt.suptitle('KrishiDrishti — Robustness Analysis\n'
                 'Proving the classifier is not overfit to clean synthetic data',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(EVAL_DIR, 'robustness_degradation.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"    [SAVED] {path}")

    # Save results
    nr.to_csv(os.path.join(EVAL_DIR, 'robustness_noise.csv'), index=False)
    gr.to_csv(os.path.join(EVAL_DIR, 'robustness_gaps.csv'), index=False)

    print("\n" + "=" * 60)
    print("  ROBUSTNESS DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
