"""
Water-Deficit Advisory Engine — Mandatory Deliverable #7
==========================================================
Computes FAO-56 water balance and classifies advisory levels.
Measures false-alarm and missed-stress rates against the
simulator's own known synthetic stress driver.

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

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from simulator.crop_simulator import CropGrowthSimulator
from simulator.noise_injector import NoiseInjector
from utils.config import CROP_PARAMS, ADVISORY_CONFIG, STAGE_NAMES

EVAL_DIR = os.path.join(BASE_DIR, "outputs", "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


def compute_advisory(plots, advisory_period_days=8):
    """
    Compute FAO-56 water balance and advisory classification for each plot.

    For each 8-day advisory period:
      ETc = Kc(crop, stage) * ETo
      Pe  = 0.8 * Rainfall
      Deficit = ETc - Pe  (positive = needs water)

    Advisory levels (from config):
      0 = Adequate   (deficit <= 5 mm)
      1 = Irrigate Soon (5 < deficit <= 15 mm)
      2 = Stress Detected (deficit > 15 mm)
    """
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

        # Group into advisory periods
        period_start = 0
        while period_start < n_steps:
            period_end = min(period_start + advisory_period_days, n_steps)
            period_days = days[period_start:period_end]

            # Get dominant stage in this period
            stages_in_period = plot.stage_labels[period_start:period_end]
            valid_stages = stages_in_period[stages_in_period >= 0]
            if len(valid_stages) == 0:
                period_start = period_end
                continue

            dominant_stage_id = int(np.median(valid_stages))
            dominant_stage_id = min(dominant_stage_id, len(stage_keys) - 1)
            stage_name = stage_keys[dominant_stage_id]

            # Kc for this crop x stage
            kc = kc_values.get(stage_name, 0.5)

            # Sum ETo and rainfall over the period
            eto_sum = np.sum(plot.eto[period_start:period_end])
            rain_sum = np.sum(plot.rainfall[period_start:period_end])

            # FAO-56 water balance
            etc = kc * eto_sum
            pe = pe_factor * rain_sum
            deficit = etc - pe

            # Advisory classification
            if deficit <= thresholds['adequate']:
                advisory = 0  # Adequate
            elif deficit <= thresholds['irrigate_soon']:
                advisory = 1  # Irrigate Soon
            else:
                advisory = 2  # Stress Detected

            # Ground truth: did the simulator apply stress during this period?
            stress_signal = plot.stress_signal[period_start:period_end]
            gt_stressed = np.any(stress_signal > 0.1)

            # Ground truth advisory (based on simulator's known stress)
            if not gt_stressed:
                gt_advisory = 0  # No stress applied
            else:
                max_stress = np.max(stress_signal)
                if max_stress < 0.15:
                    gt_advisory = 0
                elif max_stress < 0.30:
                    gt_advisory = 1
                else:
                    gt_advisory = 2

            records.append({
                'plot_id': plot.plot_id,
                'crop': crop,
                'crop_display': CROP_PARAMS[crop]['display_name'],
                'period_start_day': int(period_days[0]),
                'stage': stage_name,
                'kc': kc,
                'eto_sum': float(eto_sum),
                'rainfall_sum': float(rain_sum),
                'etc': float(etc),
                'pe': float(pe),
                'deficit_mm': float(deficit),
                'advisory_pred': advisory,
                'gt_stressed': gt_stressed,
                'gt_advisory': gt_advisory,
                'stress_intensity': float(np.max(stress_signal)) if gt_stressed else 0.0,
            })

            period_start = period_end

    return pd.DataFrame(records)


def main():
    print("=" * 60)
    print("  KrishiDrishti — Water-Deficit Advisory Engine")
    print("  Mandatory Deliverable #7")
    print("=" * 60)

    # --- Step 1: Generate data ---
    print("\n[1] Generating simulated season (seed=42, 400 plots)...")
    sim = CropGrowthSimulator(seed=42)
    plots = sim.simulate_season(num_plots=400, seed=42)

    # Inject noise (needed for observed_ndvi attribute)
    injector = NoiseInjector(seed=42)
    plots = injector.process_all_plots(plots)

    # --- Step 2: Compute advisory ---
    print("\n[2] Computing FAO-56 water balance and advisory...")
    df = compute_advisory(plots)
    print(f"    Total advisory periods: {len(df)}")

    advisory_names = ['Adequate', 'Irrigate Soon', 'Stress Detected']

    # --- Step 3: Error rates ---
    print("\n[3] Computing error rates against simulator ground truth...")

    y_true = df['gt_advisory'].values
    y_pred = df['advisory_pred'].values

    # Binary: stressed vs not-stressed
    gt_binary = (y_true > 0).astype(int)
    pred_binary = (y_pred > 0).astype(int)

    # False alarm = predicted stress when ground truth = no stress
    no_stress_mask = gt_binary == 0
    false_alarms = np.sum(pred_binary[no_stress_mask] == 1)
    total_no_stress = np.sum(no_stress_mask)
    false_alarm_rate = false_alarms / max(total_no_stress, 1)

    # Missed stress = predicted no stress when ground truth = stressed
    stress_mask = gt_binary == 1
    missed_stress = np.sum(pred_binary[stress_mask] == 0)
    total_stress = np.sum(stress_mask)
    missed_stress_rate = missed_stress / max(total_stress, 1)

    print(f"    False Alarm Rate:  {false_alarm_rate:.4f} ({false_alarms}/{total_no_stress})")
    print(f"    Missed Stress Rate: {missed_stress_rate:.4f} ({missed_stress}/{total_stress})")

    # Multi-class accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"    Advisory Accuracy: {acc:.4f}")
    print(f"\n    Classification Report (3-class):")
    print(classification_report(y_true, y_pred, target_names=advisory_names))

    # --- Step 4: Confusion Matrix ---
    print("[4] Generating advisory confusion matrix (300 DPI)...")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=advisory_names, yticklabels=advisory_names, ax=axes[0])
    axes[0].set_title('Advisory — Counts', fontweight='bold')
    axes[0].set_xlabel('Predicted Advisory'); axes[0].set_ylabel('Ground Truth')

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=advisory_names, yticklabels=advisory_names, ax=axes[1])
    axes[1].set_title('Advisory — Normalized', fontweight='bold')
    axes[1].set_xlabel('Predicted Advisory'); axes[1].set_ylabel('Ground Truth')

    plt.suptitle('Water-Deficit Advisory vs Simulator Ground Truth\n'
                 f'False Alarm Rate: {false_alarm_rate:.2%} | '
                 f'Missed Stress Rate: {missed_stress_rate:.2%}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    cm_path = os.path.join(EVAL_DIR, 'advisory_confusion_matrix.png')
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    print(f"    [SAVED] {cm_path}")

    # --- Step 5: Deficit distribution plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'Adequate': '#00AA00', 'Irrigate Soon': '#FFDD00', 'Stress Detected': '#FF0000'}
    for adv_id, name in enumerate(advisory_names):
        subset = df[df['advisory_pred'] == adv_id]['deficit_mm']
        if len(subset) > 0:
            ax.hist(subset, bins=30, alpha=0.6, color=list(colors.values())[adv_id],
                    label=f'{name} (n={len(subset)})', edgecolor='white')
    ax.axvline(5, color='gray', linestyle='--', alpha=0.7, label='Threshold: 5mm')
    ax.axvline(15, color='gray', linestyle='-.', alpha=0.7, label='Threshold: 15mm')
    ax.set_xlabel('Water Deficit (mm / 8-day period)')
    ax.set_ylabel('Count')
    ax.set_title('Water Deficit Distribution by Advisory Class', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    dist_path = os.path.join(EVAL_DIR, 'advisory_deficit_distribution.png')
    plt.savefig(dist_path, bbox_inches='tight')
    plt.close()
    print(f"    [SAVED] {dist_path}")

    # --- Step 6: Save metrics ---
    metrics = {
        'false_alarm_rate': float(false_alarm_rate),
        'missed_stress_rate': float(missed_stress_rate),
        'advisory_accuracy': float(acc),
        'total_periods': int(len(df)),
        'periods_adequate': int((y_pred == 0).sum()),
        'periods_irrigate_soon': int((y_pred == 1).sum()),
        'periods_stress_detected': int((y_pred == 2).sum()),
        'decision_logic': {
            'method': 'FAO-56 Single Crop Coefficient',
            'ETc': 'Kc(crop, stage) * ETo',
            'Pe': '0.8 * Rainfall',
            'Deficit': 'ETc - Pe',
            'thresholds': ADVISORY_CONFIG['thresholds'],
        }
    }

    metrics_path = os.path.join(EVAL_DIR, 'advisory_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"    [SAVED] {metrics_path}")

    # Save advisory results CSV
    csv_path = os.path.join(EVAL_DIR, 'advisory_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"    [SAVED] {csv_path}")

    print("\n" + "=" * 60)
    print("  ADVISORY ENGINE COMPLETE")
    print(f"  False Alarm: {false_alarm_rate:.2%} | Missed Stress: {missed_stress_rate:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
