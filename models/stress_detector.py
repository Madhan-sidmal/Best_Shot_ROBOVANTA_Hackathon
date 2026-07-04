"""
Phenology Extractor & Stress Detector
========================================
Extracts crop phenological stages from NDVI time series and
detects moisture stress using multi-indicator fusion.

Components:
    1. Phenology Extraction — SOS, Peak, EOS, growth stages
    2. Stress Index Computation — VCI, NDWI anomaly, SAR moisture proxy
    3. Stage-wise Stress Classification — Stress levels per growth stage
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import PHENOLOGY, STRESS_THRESHOLDS, OUTPUT_DIR


class PhenologyExtractor:
    """
    Extracts crop phenological stages from NDVI time series.
    
    Uses Savitzky-Golay smoothing + threshold-based stage detection.
    
    Growth Stages:
        1. Germination  — NDVI starts rising from base
        2. Vegetative   — Rapid NDVI increase
        3. Reproductive — NDVI at/near peak
        4. Maturity     — NDVI declining (senescence)
    """
    
    def __init__(self, config=None):
        self.config = config or PHENOLOGY
    
    def smooth_ndvi(self, ndvi_series, window=None, order=None):
        """
        Apply Savitzky-Golay filter to smooth noisy NDVI time series.
        
        Args:
            ndvi_series: np.array — Raw NDVI values
            window: int — Smoothing window (must be odd)
            order: int — Polynomial order
        
        Returns:
            np.array — Smoothed NDVI
        """
        window = window or self.config['smoothing_window']
        order = order or self.config['smoothing_order']
        
        # Ensure window is odd
        if window % 2 == 0:
            window += 1
        
        # Handle NaN values — interpolate first
        valid_mask = ~np.isnan(ndvi_series)
        if valid_mask.sum() < window:
            return ndvi_series
        
        x = np.arange(len(ndvi_series))
        if not valid_mask.all():
            interp_func = interp1d(x[valid_mask], ndvi_series[valid_mask], 
                                    kind='linear', fill_value='extrapolate')
            ndvi_filled = interp_func(x)
        else:
            ndvi_filled = ndvi_series.copy()
        
        # Apply Savitzky-Golay filter
        smoothed = savgol_filter(ndvi_filled, window, order)
        
        # Clip to valid NDVI range
        smoothed = np.clip(smoothed, -1.0, 1.0)
        
        return smoothed
    
    def detect_phenology(self, ndvi_series, dates, threshold_sos=None, threshold_eos=None):
        """
        Detect phenological stages from smoothed NDVI time series.
        
        Args:
            ndvi_series: np.array — NDVI values (raw or smoothed)
            dates: np.array — Corresponding dates (ordinal or datetime)
            threshold_sos: float — NDVI threshold for Start of Season
            threshold_eos: float — NDVI threshold for End of Season
        
        Returns:
            dict with phenological metrics
        """
        threshold_sos = threshold_sos or self.config['ndvi_threshold_sos']
        threshold_eos = threshold_eos or self.config['ndvi_threshold_eos']
        
        # Smooth the series
        smoothed = self.smooth_ndvi(ndvi_series)
        
        # Find Start of Season (SOS)
        # First time NDVI crosses threshold from below
        sos_idx = None
        for i in range(1, len(smoothed)):
            if smoothed[i-1] < threshold_sos and smoothed[i] >= threshold_sos:
                sos_idx = i
                break
        
        # Find Peak (maximum NDVI)
        peak_idx = np.argmax(smoothed)
        
        # Find End of Season (EOS)
        # After peak, first time NDVI drops below threshold
        eos_idx = None
        for i in range(peak_idx + 1, len(smoothed)):
            if smoothed[i] < threshold_eos:
                eos_idx = i
                break
        
        # Default values if not detected
        if sos_idx is None:
            sos_idx = 0
        if eos_idx is None:
            eos_idx = len(smoothed) - 1
        
        # Season length
        season_length = eos_idx - sos_idx
        
        # Growth rate (green-up speed)
        if peak_idx > sos_idx:
            green_up_rate = (smoothed[peak_idx] - smoothed[sos_idx]) / (peak_idx - sos_idx)
        else:
            green_up_rate = 0
        
        # Senescence rate
        if eos_idx > peak_idx:
            senescence_rate = (smoothed[peak_idx] - smoothed[eos_idx]) / (eos_idx - peak_idx)
        else:
            senescence_rate = 0
        
        return {
            'sos_idx': sos_idx,
            'peak_idx': peak_idx,
            'eos_idx': eos_idx,
            'sos_date': dates[sos_idx] if dates is not None else sos_idx,
            'peak_date': dates[peak_idx] if dates is not None else peak_idx,
            'eos_date': dates[eos_idx] if dates is not None else eos_idx,
            'season_length': season_length,
            'peak_ndvi': smoothed[peak_idx],
            'green_up_rate': green_up_rate,
            'senescence_rate': senescence_rate,
            'smoothed_ndvi': smoothed
        }
    
    def classify_growth_stage(self, ndvi_series, dates, current_date_idx):
        """
        Classify the current growth stage of a crop pixel.
        
        Args:
            ndvi_series: np.array — NDVI time series
            dates: np.array — Date array
            current_date_idx: int — Index of current date
        
        Returns:
            str — Growth stage name
            int — Stage number (0-3)
        """
        pheno = self.detect_phenology(ndvi_series, dates)
        
        sos = pheno['sos_idx']
        peak = pheno['peak_idx']
        eos = pheno['eos_idx']
        season_length = max(eos - sos, 1)
        
        if current_date_idx < sos:
            return "Pre-season", -1
        
        # Fraction of season elapsed
        frac = (current_date_idx - sos) / season_length
        
        stages = self.config['stages']
        if frac <= stages['germination']['duration_pct'][1]:
            return "Germination", 0
        elif frac <= stages['vegetative']['duration_pct'][1]:
            return "Vegetative", 1
        elif frac <= stages['reproductive']['duration_pct'][1]:
            return "Reproductive", 2
        elif frac <= stages['maturity']['duration_pct'][1]:
            return "Maturity", 3
        else:
            return "Post-harvest", 4


class StressDetector:
    """
    Phenology-aware moisture stress detection.
    
    Computes stress indices and classifies stress levels
    at each crop growth stage.
    
    Stress Indices:
        - VCI (Vegetation Condition Index)
        - NDWI anomaly
        - SAR backscatter moisture proxy
        - Combined Stress Index (CSI)
    """
    
    def __init__(self, config=None):
        self.config = config or STRESS_THRESHOLDS
        self.phenology_extractor = PhenologyExtractor()
    
    def compute_vci(self, ndvi_current, ndvi_min, ndvi_max):
        """
        Vegetation Condition Index (VCI).
        
        VCI = (NDVI_current - NDVI_min) / (NDVI_max - NDVI_min) × 100
        
        Interpretation:
            VCI > 60%: No drought / Good condition
            40-60%: Mild drought
            20-40%: Moderate drought
            < 20%: Severe drought
        """
        denominator = ndvi_max - ndvi_min
        denominator = np.where(denominator == 0, 1e-6, denominator)
        vci = (ndvi_current - ndvi_min) / denominator
        return np.clip(vci, 0, 1)
    
    def compute_ndwi_anomaly(self, ndwi_current, ndwi_historical_mean, ndwi_historical_std):
        """
        NDWI anomaly (z-score relative to historical baseline).
        
        Negative anomaly = drier than normal = stress
        """
        std = np.where(ndwi_historical_std == 0, 1e-6, ndwi_historical_std)
        anomaly = (ndwi_current - ndwi_historical_mean) / std
        return anomaly
    
    def compute_sar_moisture_proxy(self, vh_current, vh_historical_mean, vh_historical_std):
        """
        SAR-based soil/vegetation moisture proxy.
        
        VH backscatter is sensitive to vegetation water content.
        Lower VH relative to historical = drier conditions.
        """
        std = np.where(vh_historical_std == 0, 1e-6, vh_historical_std)
        sar_anomaly = (vh_current - vh_historical_mean) / std
        return sar_anomaly
    
    def compute_combined_stress_index(self, vci, ndwi_anomaly, sar_anomaly, 
                                       weights=None):
        """
        Combined Stress Index (CSI) — Weighted fusion of all indicators.
        
        Args:
            vci: VCI values [0, 1]
            ndwi_anomaly: NDWI z-scores
            sar_anomaly: SAR z-scores
            weights: dict — Weights for each component
        
        Returns:
            Combined stress index [0, 1] where lower = more stressed
        """
        if weights is None:
            weights = {'vci': 0.4, 'ndwi': 0.3, 'sar': 0.3}
        
        # Normalize anomalies to [0, 1] range using sigmoid
        ndwi_norm = 1 / (1 + np.exp(-ndwi_anomaly))
        sar_norm = 1 / (1 + np.exp(-sar_anomaly))
        
        csi = (weights['vci'] * vci + 
               weights['ndwi'] * ndwi_norm + 
               weights['sar'] * sar_norm)
        
        return np.clip(csi, 0, 1)
    
    def classify_stress(self, stress_index):
        """
        Classify stress level from a stress index value.
        
        Args:
            stress_index: float or np.array — Stress index [0, 1]
        
        Returns:
            int or np.array — Stress class (0=None, 1=Mild, 2=Moderate, 3=Severe)
        """
        thresholds = self.config['vci']
        
        stress_class = np.zeros_like(stress_index, dtype=int)
        stress_class = np.where(stress_index >= thresholds['no_stress'], 0, stress_class)
        stress_class = np.where(
            (stress_index >= thresholds['mild']) & (stress_index < thresholds['no_stress']),
            1, stress_class
        )
        stress_class = np.where(
            (stress_index >= thresholds['moderate']) & (stress_index < thresholds['mild']),
            2, stress_class
        )
        stress_class = np.where(stress_index < thresholds['moderate'], 3, stress_class)
        
        return stress_class
    
    def stagewise_stress_analysis(self, ndvi_series, ndwi_series, vh_series,
                                   dates, ndvi_historical_range,
                                   ndwi_historical_stats, vh_historical_stats):
        """
        Full stage-wise stress analysis for a single pixel/field.
        
        Args:
            ndvi_series: np.array — NDVI time series
            ndwi_series: np.array — NDWI time series
            vh_series: np.array — SAR VH time series
            dates: np.array — Date labels
            ndvi_historical_range: tuple — (min, max) from historical data
            ndwi_historical_stats: tuple — (mean, std) from historical data
            vh_historical_stats: tuple — (mean, std) from historical data
        
        Returns:
            pd.DataFrame — Stage-wise stress analysis results
        """
        # Extract phenology
        pheno = self.phenology_extractor.detect_phenology(ndvi_series, dates)
        smoothed_ndvi = pheno['smoothed_ndvi']
        
        results = []
        
        for i in range(len(ndvi_series)):
            # Determine growth stage
            stage_name, stage_num = self.phenology_extractor.classify_growth_stage(
                ndvi_series, dates, i
            )
            
            # Compute stress indices
            vci = self.compute_vci(
                smoothed_ndvi[i],
                ndvi_historical_range[0],
                ndvi_historical_range[1]
            )
            
            ndwi_anom = self.compute_ndwi_anomaly(
                ndwi_series[i],
                ndwi_historical_stats[0],
                ndwi_historical_stats[1]
            )
            
            sar_anom = self.compute_sar_moisture_proxy(
                vh_series[i],
                vh_historical_stats[0],
                vh_historical_stats[1]
            )
            
            # Combined stress
            csi = self.compute_combined_stress_index(
                np.array([vci]),
                np.array([ndwi_anom]),
                np.array([sar_anom])
            )[0]
            
            # Classify
            stress_class = self.classify_stress(np.array([csi]))[0]
            stress_name = self.config['classes'][stress_class]['name']
            
            results.append({
                'date': dates[i] if dates is not None else i,
                'ndvi': smoothed_ndvi[i],
                'ndwi': ndwi_series[i],
                'vh': vh_series[i],
                'growth_stage': stage_name,
                'stage_num': stage_num,
                'vci': vci,
                'ndwi_anomaly': ndwi_anom,
                'sar_anomaly': sar_anom,
                'combined_stress_index': csi,
                'stress_class': stress_class,
                'stress_level': stress_name
            })
        
        return pd.DataFrame(results)
    
    def plot_stress_timeline(self, analysis_df, title="Crop Stress Timeline"):
        """
        Create a comprehensive visualization of stress over time.
        """
        fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
        
        dates = range(len(analysis_df))
        
        # ---- NDVI + Growth Stages ----
        ax1 = axes[0]
        ax1.plot(dates, analysis_df['ndvi'], 'g-', linewidth=2, label='NDVI')
        ax1.fill_between(dates, 0, analysis_df['ndvi'], alpha=0.1, color='green')
        ax1.set_ylabel('NDVI', fontsize=12)
        ax1.set_title('NDVI & Growth Stages', fontsize=13, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Color background by growth stage
        stage_colors = {
            'Pre-season': '#E0E0E0', 'Germination': '#FFF9C4',
            'Vegetative': '#C8E6C9', 'Reproductive': '#BBDEFB',
            'Maturity': '#FFE0B2', 'Post-harvest': '#E0E0E0'
        }
        for i in range(len(analysis_df)):
            stage = analysis_df.iloc[i]['growth_stage']
            color = stage_colors.get(stage, '#FFFFFF')
            ax1.axvspan(i - 0.5, i + 0.5, alpha=0.3, color=color)
        
        # ---- Stress Indices ----
        ax2 = axes[1]
        ax2.plot(dates, analysis_df['vci'], 'b-', linewidth=1.5, label='VCI')
        ax2.axhline(y=0.6, color='green', linestyle='--', alpha=0.5, label='No stress threshold')
        ax2.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, label='Mild threshold')
        ax2.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Severe threshold')
        ax2.set_ylabel('VCI', fontsize=12)
        ax2.set_title('Vegetation Condition Index (VCI)', fontsize=13, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # ---- Combined Stress Index ----
        ax3 = axes[2]
        stress_colors_map = {0: '#00AA00', 1: '#FFDD00', 2: '#FF8800', 3: '#FF0000'}
        bar_colors = [stress_colors_map[c] for c in analysis_df['stress_class']]
        ax3.bar(dates, analysis_df['combined_stress_index'], color=bar_colors, width=0.8)
        ax3.set_ylabel('Combined Stress Index', fontsize=12)
        ax3.set_title('Combined Stress Index (CSI) — Stage-wise', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # ---- SAR Backscatter ----
        ax4 = axes[3]
        ax4.plot(dates, analysis_df['vh'], 'purple', linewidth=1.5, label='VH Backscatter')
        ax4.plot(dates, analysis_df['ndwi'], 'cyan', linewidth=1.5, label='NDWI')
        ax4.set_ylabel('Value', fontsize=12)
        ax4.set_xlabel('Time Step', fontsize=12)
        ax4.set_title('SAR Backscatter & NDWI', fontsize=13, fontweight='bold')
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        
        # Legend for stress levels
        legend_patches = [
            mpatches.Patch(color='#00AA00', label='No Stress'),
            mpatches.Patch(color='#FFDD00', label='Mild Stress'),
            mpatches.Patch(color='#FF8800', label='Moderate Stress'),
            mpatches.Patch(color='#FF0000', label='Severe Stress'),
        ]
        fig.legend(handles=legend_patches, loc='lower center', ncol=4, fontsize=11,
                   bbox_to_anchor=(0.5, -0.02))
        
        plt.suptitle(f'🛰️ KrishiDrishti — {title}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(OUTPUT_DIR, 'stress_timeline.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"📊 Stress timeline saved to: {save_path}")


# ============================================================
# DEMO
# ============================================================
def demo_stress_detection():
    """Demonstrate stress detection with synthetic time series."""
    print("💧 KrishiDrishti — Stress Detection Demo")
    print("=" * 60)
    
    np.random.seed(42)
    n_timesteps = 24  # ~6 months of 8-day composites
    
    # Simulate a rice crop NDVI profile with stress mid-season
    t = np.linspace(0, 2 * np.pi, n_timesteps)
    ndvi = 0.15 + 0.55 * np.sin(t - np.pi / 4)
    ndvi = np.clip(ndvi + np.random.normal(0, 0.03, n_timesteps), 0, 1)
    
    # Introduce a stress period (dip around time step 12-16)
    ndvi[12:16] -= 0.15
    
    # Simulated NDWI and VH
    ndwi = ndvi * 0.6 + np.random.normal(0, 0.02, n_timesteps)
    vh = -15 + ndvi * 5 + np.random.normal(0, 0.5, n_timesteps)
    vh[12:16] -= 2  # SAR also shows stress
    
    dates = [f"Day_{i*8}" for i in range(n_timesteps)]
    
    # Historical baselines
    ndvi_hist_range = (0.1, 0.8)
    ndwi_hist_stats = (0.3, 0.1)
    vh_hist_stats = (-12, 1.5)
    
    # Run analysis
    detector = StressDetector()
    analysis = detector.stagewise_stress_analysis(
        ndvi, ndwi, vh, dates,
        ndvi_hist_range, ndwi_hist_stats, vh_hist_stats
    )
    
    print("\n📋 Stage-wise Analysis Summary:")
    print(analysis[['date', 'growth_stage', 'vci', 'combined_stress_index', 'stress_level']].to_string(index=False))
    
    # Visualize
    detector.plot_stress_timeline(analysis, title="Rice Crop — Kharif 2025")
    
    return analysis


if __name__ == '__main__':
    analysis = demo_stress_detection()
