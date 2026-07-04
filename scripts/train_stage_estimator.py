"""
Growth-Stage Estimator — Mandatory Deliverable #6
=====================================================
Trains a classifier to predict growth stage (sowing/vegetative/
flowering/maturity/harvest) from the NOISY, GAPPED simulated
index stream — never from the clean ground-truth curve.

Evaluated SEPARATELY from crop-type accuracy.

DISCLAIMER: All data is synthetically generated.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, cohen_kappa_score
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from simulator.crop_simulator import CropGrowthSimulator
from simulator.noise_injector import NoiseInjector
from utils.config import STAGE_NAMES, STAGE_ESTIMATOR_CONFIG

MODEL_DIR = os.path.join(BASE_DIR, "models", "saved")
EVAL_DIR = os.path.join(BASE_DIR, "outputs", "evaluation")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


def extract_stage_features(plots, window=5):
    """
    Extract features from the NOISY, GAPPED observed NDVI for
    growth-stage prediction.

    Features per observation:
      - observed NDVI (with NaN interpolated)
      - day of season (normalized 0-1)
      - fraction of crop duration elapsed (approximate)
      - rolling mean NDVI (window)
      - rolling std NDVI (window)
      - NDVI derivative (rate of change)
      - cumulative NDVI sum
      - max NDVI seen so far
      - days since last valid observation
    """
    records = []

    for plot in plots:
        if not hasattr(plot, 'observed_ndvi'):
            continue

        obs = plot.observed_ndvi.copy()
        days = plot.days
        mask = plot.valid_mask

        # Interpolate NaN gaps for feature extraction
        # (the classifier still doesn't see the clean curve)
        obs_interp = obs.copy()
        nan_idx = np.isnan(obs_interp)
        if nan_idx.any() and (~nan_idx).any():
            obs_interp[nan_idx] = np.interp(
                days[nan_idx], days[~nan_idx], obs_interp[~nan_idx]
            )

        season_len = days[-1] - days[0] + 1

        for i in range(len(days)):
            day = days[i]
            stage_id = plot.stage_labels[i]

            # Skip no-crop time steps
            if stage_id < 0:
                continue

            ndvi_val = obs_interp[i]
            day_frac = day / season_len

            # Rolling features
            start = max(0, i - window)
            window_vals = obs_interp[start:i+1]

            roll_mean = np.nanmean(window_vals)
            roll_std = np.nanstd(window_vals) if len(window_vals) > 1 else 0.0

            # Derivative
            if i > 0:
                deriv = (obs_interp[i] - obs_interp[i-1]) / max(days[i] - days[i-1], 1)
            else:
                deriv = 0.0

            # Cumulative features
            cum_sum = np.nansum(obs_interp[:i+1])
            max_so_far = np.nanmax(obs_interp[:i+1])

            # Gap feature
            if mask[i]:
                days_since_valid = 0
            else:
                valid_before = np.where(mask[:i])[0]
                if len(valid_before) > 0:
                    days_since_valid = day - days[valid_before[-1]]
                else:
                    days_since_valid = day

            # Is this observation a gap?
            is_gap = 0 if mask[i] else 1

            records.append({
                'plot_id': plot.plot_id,
                'crop_type': plot.crop_type,
                'day': day,
                'ndvi': ndvi_val,
                'day_frac': day_frac,
                'roll_mean': roll_mean,
                'roll_std': roll_std,
                'derivative': deriv,
                'cum_sum': cum_sum,
                'max_so_far': max_so_far,
                'days_since_valid': days_since_valid,
                'is_gap': is_gap,
                'stage_id': stage_id,
            })

    return pd.DataFrame(records)


def main():
    print("=" * 60)
    print("  KrishiDrishti — Growth-Stage Estimator")
    print("  Mandatory Deliverable #6")
    print("=" * 60)

    # --- Step 1: Generate fresh simulated data ---
    print("\n[1] Generating simulated season (seed=42, 400 plots)...")
    sim = CropGrowthSimulator(seed=42)
    plots = sim.simulate_season(num_plots=400, seed=42)

    # --- Step 2: Inject noise + cloud gaps ---
    print("\n[2] Injecting noise and cloud gaps...")
    injector = NoiseInjector(seed=42)
    plots = injector.process_all_plots(plots)

    # --- Step 3: Extract features from NOISY data ---
    print("\n[3] Extracting features from noisy/gapped observations...")
    df = extract_stage_features(plots)
    print(f"    Total samples: {len(df)}")
    print(f"    Stage distribution:\n{df['stage_id'].value_counts().sort_index().to_string()}")

    feature_cols = ['ndvi', 'day_frac', 'roll_mean', 'roll_std',
                    'derivative', 'cum_sum', 'max_so_far',
                    'days_since_valid', 'is_gap']

    X = df[feature_cols].values.astype(np.float32)
    y = df['stage_id'].values.astype(int)
    X = np.nan_to_num(X, nan=0.0)

    stage_names = STAGE_NAMES  # ['sowing', 'vegetative', 'flowering', 'maturity', 'harvest']

    # --- Step 4: Train/test split ---
    # Split by PLOT to avoid data leakage (all time steps of a plot go together)
    plot_ids = df['plot_id'].unique()
    train_plots, test_plots = train_test_split(plot_ids, test_size=0.25, random_state=42)

    train_mask = df['plot_id'].isin(train_plots)
    test_mask = df['plot_id'].isin(test_plots)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"\n[4] Train: {len(X_train)}, Test: {len(X_test)}")

    # --- Step 5: Train Random Forest ---
    print("\n[5] Training Random Forest stage estimator...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    cfg = STAGE_ESTIMATOR_CONFIG
    model = RandomForestClassifier(
        n_estimators=cfg.get('n_estimators', 300),
        max_depth=cfg.get('max_depth', 15),
        class_weight='balanced',
        random_state=cfg.get('random_state', 42),
        n_jobs=-1
    )
    model.fit(X_train_s, y_train)

    # --- Step 6: 5-fold cross-validation ---
    print("\n[6] Running 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_s, y_train, cv=cv, scoring='accuracy')
    print(f"    CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # --- Step 7: Evaluate on test set ---
    print("\n[7] Evaluating on test set...")
    y_pred = model.predict(X_test_s)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_test, y_pred)

    print(f"    Accuracy:     {acc:.4f}")
    print(f"    F1 (weighted): {f1:.4f}")
    print(f"    Cohen's Kappa: {kappa:.4f}")

    report = classification_report(y_test, y_pred, target_names=stage_names, output_dict=True)
    print(f"\n    Classification Report:")
    print(classification_report(y_test, y_pred, target_names=stage_names))

    # --- Step 8: Confusion Matrix ---
    print("[8] Generating confusion matrix (300 DPI)...")
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=stage_names, yticklabels=stage_names, ax=axes[0])
    axes[0].set_title('Growth Stage — Counts', fontweight='bold')
    axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=stage_names, yticklabels=stage_names, ax=axes[1])
    axes[1].set_title('Growth Stage — Normalized', fontweight='bold')
    axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')

    plt.suptitle('Growth-Stage Estimator — Confusion Matrix\n'
                 '(Evaluated on noisy/gapped data, separate from crop-type accuracy)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    cm_path = os.path.join(EVAL_DIR, 'stage_confusion_matrix.png')
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    print(f"    [SAVED] {cm_path}")

    # --- Step 9: Feature importance ---
    fi = model.feature_importances_
    idx = np.argsort(fi)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(feature_cols)), fi[idx][::-1],
            color=plt.cm.viridis(np.linspace(0.2, 0.8, len(feature_cols))))
    ax.set_yticks(range(len(feature_cols)))
    ax.set_yticklabels([feature_cols[i] for i in idx[::-1]])
    ax.set_xlabel('Importance')
    ax.set_title('Growth-Stage Estimator — Feature Importance', fontweight='bold')
    plt.tight_layout()
    fi_path = os.path.join(EVAL_DIR, 'stage_feature_importance.png')
    plt.savefig(fi_path, bbox_inches='tight')
    plt.close()
    print(f"    [SAVED] {fi_path}")

    # --- Step 10: Save model ---
    metrics = {
        'accuracy': float(acc),
        'f1_weighted': float(f1),
        'kappa': float(kappa),
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'per_stage': {name: report[name] for name in stage_names if name in report}
    }

    model_path = os.path.join(MODEL_DIR, 'stage_estimator.joblib')
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'features': feature_cols,
        'stage_names': stage_names,
        'metrics': metrics
    }, model_path)
    print(f"\n    [SAVED] {model_path}")

    # Save metrics JSON
    metrics_path = os.path.join(EVAL_DIR, 'stage_estimator_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"    [SAVED] {metrics_path}")

    print("\n" + "=" * 60)
    print("  GROWTH-STAGE ESTIMATOR COMPLETE")
    print(f"  Accuracy: {acc:.4f} | F1: {f1:.4f} | Kappa: {kappa:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
