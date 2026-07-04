"""
Noise & Cloud-Gap Injection Module
====================================
Adds realistic sensor-level noise and clustered observation gaps
to clean simulated NDVI curves, mimicking:
    - Sensor radiometric noise (Gaussian)
    - Monsoon-season cloud cover (clustered gaps, not uniform)
    - Satellite revisit cadence limitations

Gap clustering follows a Markov-chain model: once a gap starts,
consecutive observations are likely to also be missing (sticky gaps),
with higher probability during the configured monsoon window.

Configurable parameters:
    - noise_std: σ of Gaussian noise
    - gap_cluster_probability: P(starting a new gap cluster)
    - gap_cluster_size_range: (min, max) consecutive missing obs
    - monsoon_gap_boost: multiplier for gap probability during wet season
    
DISCLAIMER: All data is synthetically generated.
"""

import numpy as np
from typing import List, Tuple, Optional
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import NOISE_CONFIG, SIMULATION


class NoiseInjector:
    """
    Adds sensor noise and clustered cloud gaps to clean NDVI series.
    
    Noise Model:
        noisy_ndvi(t) = clean_ndvi(t) + ε(t)
        where ε(t) ~ N(0, σ²) and σ is configurable.
    
    Gap Model (Markov-chain clustered gaps):
        At each time step t:
            if in_gap_cluster:
                P(gap) = 0.85 (high — sticky)
                cluster_remaining -= 1
            else:
                P(start_new_cluster) = gap_cluster_probability
                    × monsoon_boost (if in monsoon window)
                if new cluster starts:
                    cluster_size ~ Uniform(min_size, max_size)
        
        Gap = observation is set to NaN (missing).
    """
    
    def __init__(self, seed: int = None, config: dict = None):
        self.config = config or NOISE_CONFIG
        self.sim_config = SIMULATION
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def add_noise(self, clean_ndvi: np.ndarray,
                   noise_std: float = None) -> np.ndarray:
        """
        Add Gaussian sensor noise to clean NDVI values.
        
        Args:
            clean_ndvi: noise-free NDVI array
            noise_std: standard deviation of noise (default from config)
        
        Returns:
            noisy_ndvi: NDVI with added noise (clipped to [0, 1])
        """
        sigma = noise_std or self.config["noise_std"]
        noise = self.rng.normal(0, sigma, size=len(clean_ndvi))
        noisy = clean_ndvi + noise
        return np.clip(noisy, 0.0, 1.0)
    
    def inject_cloud_gaps(self, ndvi: np.ndarray,
                           days: np.ndarray,
                           gap_probability: float = None,
                           cluster_prob: float = None,
                           cluster_size_range: Tuple[int, int] = None,
                           monsoon_boost: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject clustered cloud-gap missing observations.
        
        Gaps are NOT uniformly random — they cluster together,
        especially during the monsoon/wet season window.
        
        Args:
            ndvi: NDVI array (will be modified with NaN for gaps)
            days: day indices
            gap_probability: base P(gap) at any step
            cluster_prob: P(starting a new gap cluster)
            cluster_size_range: (min, max) cluster length
            monsoon_boost: multiplier during monsoon
        
        Returns:
            gapped_ndvi: NDVI with NaN gaps
            gap_mask: boolean array (True = observation present, False = gap)
        """
        cfg = self.config
        cluster_prob = cluster_prob or cfg["gap_cluster_probability"]
        cluster_size_range = cluster_size_range or tuple(cfg["gap_cluster_size_range"])
        monsoon_boost = monsoon_boost or cfg["monsoon_gap_boost"]
        
        season_len = self.sim_config["season_length_days"]
        monsoon_start = cfg["monsoon_window_frac"][0] * season_len
        monsoon_end = cfg["monsoon_window_frac"][1] * season_len
        
        gapped = ndvi.copy()
        gap_mask = np.ones(len(ndvi), dtype=bool)  # True = valid
        
        cluster_remaining = 0
        
        for i, day in enumerate(days):
            # Are we in the monsoon window?
            in_monsoon = monsoon_start <= day <= monsoon_end
            boost = monsoon_boost if in_monsoon else 1.0
            
            if cluster_remaining > 0:
                # Inside an active gap cluster — high probability of gap
                if self.rng.random() < 0.85:
                    gapped[i] = np.nan
                    gap_mask[i] = False
                cluster_remaining -= 1
            else:
                # Check if a new gap cluster starts
                effective_prob = cluster_prob * boost
                if self.rng.random() < effective_prob:
                    # Start a new cluster
                    cluster_size = self.rng.randint(
                        cluster_size_range[0], cluster_size_range[1] + 1
                    )
                    cluster_remaining = cluster_size
                    gapped[i] = np.nan
                    gap_mask[i] = False
        
        return gapped, gap_mask
    
    def process_plot(self, clean_ndvi: np.ndarray,
                      days: np.ndarray,
                      noise_std: float = None,
                      gap_dropout_rate: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full noise + gap injection pipeline for a single plot.
        
        Args:
            clean_ndvi: clean NDVI from simulator
            days: observation day indices
            noise_std: override noise σ
            gap_dropout_rate: override gap cluster probability
                              (useful for live robustness demo)
        
        Returns:
            observed_ndvi: noisy + gapped NDVI (what the classifier sees)
            valid_mask: boolean mask (True = observation exists)
        """
        # Step 1: Add sensor noise
        noisy = self.add_noise(clean_ndvi, noise_std)
        
        # Step 2: Inject cloud gaps
        observed, valid_mask = self.inject_cloud_gaps(
            noisy, days,
            cluster_prob=gap_dropout_rate
        )
        
        return observed, valid_mask
    
    def process_all_plots(self, plots, noise_std: float = None,
                           gap_dropout_rate: float = None):
        """
        Apply noise + gaps to all simulated plots.
        
        Adds 'observed_ndvi' and 'valid_mask' attributes to each plot.
        
        Args:
            plots: list of PlotSimulation objects
            noise_std: override noise σ (for robustness testing)
            gap_dropout_rate: override gap rate (for robustness testing)
        
        Returns:
            plots: same list with new attributes added
        """
        total_gaps = 0
        total_obs = 0
        
        for plot in plots:
            observed, mask = self.process_plot(
                plot.clean_ndvi, plot.days,
                noise_std=noise_std,
                gap_dropout_rate=gap_dropout_rate
            )
            plot.observed_ndvi = observed
            plot.valid_mask = mask
            
            total_obs += len(mask)
            total_gaps += (~mask).sum()
        
        gap_pct = 100 * total_gaps / total_obs if total_obs > 0 else 0
        print(f"✅ Noise & gap injection complete:")
        print(f"   Noise σ: {noise_std or self.config['noise_std']:.3f}")
        print(f"   Gap rate: {gap_dropout_rate or self.config['gap_cluster_probability']:.2f}")
        print(f"   Total gaps: {total_gaps}/{total_obs} ({gap_pct:.1f}%)")
        
        return plots
    
    def get_gap_statistics(self, plots) -> dict:
        """Compute gap statistics for reporting."""
        total_obs = 0
        total_gaps = 0
        monsoon_gaps = 0
        monsoon_total = 0
        
        season_len = self.sim_config["season_length_days"]
        m_start = self.config["monsoon_window_frac"][0] * season_len
        m_end = self.config["monsoon_window_frac"][1] * season_len
        
        for plot in plots:
            if not hasattr(plot, 'valid_mask'):
                continue
            for i, day in enumerate(plot.days):
                total_obs += 1
                if not plot.valid_mask[i]:
                    total_gaps += 1
                in_monsoon = m_start <= day <= m_end
                if in_monsoon:
                    monsoon_total += 1
                    if not plot.valid_mask[i]:
                        monsoon_gaps += 1
        
        return {
            "total_observations": total_obs,
            "total_gaps": total_gaps,
            "gap_percentage": 100 * total_gaps / max(total_obs, 1),
            "monsoon_gap_percentage": 100 * monsoon_gaps / max(monsoon_total, 1),
            "non_monsoon_gap_percentage": 100 * (total_gaps - monsoon_gaps) / max(total_obs - monsoon_total, 1),
        }


# ============================================================
# INIT FILE
# ============================================================
# Write __init__.py for the simulator package
_init_path = os.path.join(os.path.dirname(__file__), "__init__.py")
if not os.path.exists(_init_path):
    with open(_init_path, 'w') as f:
        f.write("# KrishiDrishti — Simulator Package\n")


# ============================================================
# CLI DEMO
# ============================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from crop_simulator import CropGrowthSimulator
    
    # Generate clean data
    sim = CropGrowthSimulator(seed=42)
    plots = sim.simulate_season(num_plots=4, seed=42)
    
    # Inject noise and gaps
    injector = NoiseInjector(seed=42)
    plots = injector.process_all_plots(plots)
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, plot in enumerate(plots[:4]):
        ax = axes[idx]
        p = sim.crop_params[plot.crop_type]
        
        # Clean curve
        ax.plot(plot.days, plot.clean_ndvi, '--', color='gray',
                linewidth=1, alpha=0.5, label='Clean (ground truth)')
        
        # Observed (noisy + gaps)
        valid = plot.valid_mask
        ax.scatter(plot.days[valid], plot.observed_ndvi[valid],
                   color=p["color"], s=20, zorder=5, label='Observed')
        
        # Gap indicators
        gap_days = plot.days[~valid]
        ax.scatter(gap_days, np.zeros(len(gap_days)) + 0.05,
                   marker='x', color='red', s=15, alpha=0.5, label='Cloud gap')
        
        if plot.stress_applied:
            ax.axvspan(plot.stress_onset_day,
                       plot.stress_onset_day + plot.stress_duration,
                       alpha=0.1, color='red')
        
        ax.set_title(f"Plot {plot.plot_id}: {p['display_name']}", fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("KrishiDrishti — Clean vs. Observed (Noisy + Gapped)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "outputs", "noise_injection.png"), dpi=150)
    plt.show()
    
    stats = injector.get_gap_statistics(plots)
    print(f"\n📊 Gap Statistics: {stats}")
