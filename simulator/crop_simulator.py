"""
Parametric Crop-Growth-Curve Simulator
========================================
Generates realistic synthetic vegetation/moisture-index trajectories
for ≥4 crop types using documented parametric equations.

Model: Double-logistic growth curve with water-stress modulation.

NDVI(t) = base + amplitude × [σ(α₁(t - t_up)) - σ(α₂(t - t_down))]

where σ(x) = 1/(1 + exp(-x)) is the logistic sigmoid,
t_up = emergence inflection, t_down = senescence inflection,
α₁ = green-up steepness, α₂ = senescence steepness.

Water stress reduces the amplitude proportionally to stress intensity
and the crop's stage-specific sensitivity.

This module also generates the growth-stage ground truth at every
time step, derived from the SAME crop-curve parameters (not inferred
from the noisy index).

Reference:
    Beck et al. (2006) — double-logistic phenology model
    FAO-56 — crop coefficient stage durations
    
DISCLAIMER: All data is synthetically generated.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import CROP_PARAMS, SIMULATION, STAGE_NAMES


@dataclass
class PlotSimulation:
    """Simulation result for a single plot/field."""
    plot_id: int
    crop_type: str
    crop_label: int
    sowing_day: int
    total_duration: int
    stress_applied: bool
    stress_onset_day: int = 0
    stress_duration: int = 0
    stress_intensity: float = 0.0
    
    # Time series (filled after simulation)
    days: np.ndarray = field(default_factory=lambda: np.array([]))
    clean_ndvi: np.ndarray = field(default_factory=lambda: np.array([]))
    stage_labels: np.ndarray = field(default_factory=lambda: np.array([]))
    stage_names_ts: list = field(default_factory=list)
    stress_signal: np.ndarray = field(default_factory=lambda: np.array([]))
    eto: np.ndarray = field(default_factory=lambda: np.array([]))
    rainfall: np.ndarray = field(default_factory=lambda: np.array([]))


class CropGrowthSimulator:
    """
    Parametric crop growth curve simulator.
    
    Generates NDVI trajectories for ≥4 crop types using a double-logistic
    model with water-stress modulation. Also produces ground-truth
    growth-stage labels tied to the same parameters.
    
    Equations:
    ---------
    Clean NDVI(t):
        NDVI(t) = base + (peak - base) × [σ(α_up(t - t_up)) × (1 - σ(α_down(t - t_down)))]
        
        where:
            base     = crop-specific base NDVI (bare soil/residue)
            peak     = crop-specific peak NDVI (full canopy)
            t_up     = sowing_day + emergence_day (green-up inflection point)
            t_down   = sowing_day + peak_day_frac × total_duration (senescence inflection)
            α_up     = green_up_rate × scaling_factor
            α_down   = senescence_rate × scaling_factor
            σ(x)     = 1 / (1 + exp(-x))  (logistic sigmoid)
    
    Water Stress Effect:
        stressed_ndvi(t) = clean_ndvi(t) × (1 - stress_intensity × stage_sensitivity(t))
        
        where stress_intensity ∈ [0.15, 0.45] and stage_sensitivity is
        crop- and stage-specific (e.g., flowering stage has sensitivity=1.0).
    """
    
    def __init__(self, seed: int = None, config: dict = None):
        self.config = config or SIMULATION
        self.seed = seed or self.config["default_seed"]
        self.rng = np.random.RandomState(self.seed)
        self.crop_params = CROP_PARAMS
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Logistic sigmoid function: σ(x) = 1/(1+exp(-x))."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
    
    def generate_clean_ndvi(self, crop: str, sowing_day: int,
                             total_duration: int,
                             days: np.ndarray) -> np.ndarray:
        """
        Generate a clean (noise-free) NDVI trajectory using the
        double-logistic model.
        
        Args:
            crop: crop type key (e.g., "rice")
            sowing_day: day of sowing (0-indexed in season)
            total_duration: crop duration in days
            days: array of observation days
        
        Returns:
            ndvi: clean NDVI at each observation day
        """
        p = self.crop_params[crop]
        
        base = p["base_ndvi"]
        peak = p["peak_ndvi"]
        amplitude = peak - base
        
        # Inflection points
        t_up = sowing_day + p["emergence_day"]
        t_down = sowing_day + p["peak_day_frac"] * total_duration
        
        # Steepness parameters (scaled for sigmoid)
        # α controls how sharp the green-up / senescence transition is
        alpha_up = p["green_up_rate"] * 0.5      # lower → more gradual
        alpha_down = p["senescence_rate"] * 0.5
        
        # Double-logistic: rise × fall
        rise = self._sigmoid(alpha_up * (days - t_up))
        fall = 1.0 - self._sigmoid(alpha_down * (days - t_down))
        
        ndvi = base + amplitude * rise * fall
        
        # Clip to physical range and set pre-sowing / post-harvest to base
        ndvi = np.where(days < sowing_day, base, ndvi)
        harvest_day = sowing_day + total_duration
        ndvi = np.where(days > harvest_day, base * 0.8, ndvi)
        
        return np.clip(ndvi, 0.0, 1.0)
    
    def generate_stage_labels(self, crop: str, sowing_day: int,
                                total_duration: int,
                                days: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Generate growth-stage ground truth at every time step.
        
        Stage boundaries are derived from the SAME crop parameters
        used for the growth curve (not inferred from NDVI).
        
        Returns:
            stage_ids: integer stage labels (0-4) per time step
            stage_names: string stage names per time step
        """
        p = self.crop_params[crop]
        stages = p["stages"]
        
        stage_ids = np.full(len(days), -1, dtype=int)   # -1 = no crop
        stage_name_list = ["no_crop"] * len(days)
        
        for i, day in enumerate(days):
            # Fraction of crop duration elapsed
            if day < sowing_day or day > sowing_day + total_duration:
                stage_ids[i] = -1
                stage_name_list[i] = "no_crop"
                continue
            
            frac = (day - sowing_day) / total_duration
            frac = np.clip(frac, 0, 1)
            
            for stage_idx, (stage_name, (start_frac, end_frac)) in enumerate(stages.items()):
                if start_frac <= frac < end_frac:
                    stage_ids[i] = stage_idx
                    stage_name_list[i] = stage_name
                    break
            else:
                # If past all stages, mark as harvest
                stage_ids[i] = len(stages) - 1
                stage_name_list[i] = list(stages.keys())[-1]
        
        return stage_ids, stage_name_list
    
    def apply_water_stress(self, clean_ndvi: np.ndarray,
                            crop: str, sowing_day: int,
                            total_duration: int, days: np.ndarray,
                            stress_onset: int, stress_duration: int,
                            stress_intensity: float,
                            stage_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply water stress to the clean NDVI curve.
        
        Stress reduces NDVI proportionally to:
            - stress_intensity (global severity)
            - stage-specific sensitivity (e.g., flowering is most sensitive)
        
        Args:
            clean_ndvi: noise-free NDVI array
            crop: crop type key
            stress_onset: day when stress begins
            stress_duration: how many days stress lasts
            stress_intensity: severity [0, 1]
            stage_labels: integer stage labels per time step
        
        Returns:
            stressed_ndvi: NDVI with stress effect
            stress_signal: binary/intensity stress signal (ground truth)
        """
        p = self.crop_params[crop]
        sensitivities = p["stress_sensitivity"]
        stage_keys = list(p["stages"].keys())
        
        stressed_ndvi = clean_ndvi.copy()
        stress_signal = np.zeros(len(days))
        
        stress_end = stress_onset + stress_duration
        
        for i, day in enumerate(days):
            if stress_onset <= day <= stress_end:
                stage_idx = stage_labels[i]
                if stage_idx >= 0 and stage_idx < len(stage_keys):
                    stage_name = stage_keys[stage_idx]
                    sensitivity = sensitivities.get(stage_name, 0.5)
                    
                    # Stress ramps up and down (trapezoidal envelope)
                    ramp_days = max(stress_duration * 0.15, 3)
                    if day < stress_onset + ramp_days:
                        ramp = (day - stress_onset) / ramp_days
                    elif day > stress_end - ramp_days:
                        ramp = (stress_end - day) / ramp_days
                    else:
                        ramp = 1.0
                    
                    reduction = stress_intensity * sensitivity * ramp
                    stressed_ndvi[i] *= (1.0 - reduction)
                    stress_signal[i] = reduction
        
        return np.clip(stressed_ndvi, 0.0, 1.0), stress_signal
    
    def generate_weather(self, days: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic reference ET (ETo) and rainfall.
        
        ETo: Sinusoidal seasonal pattern (higher in summer).
        Rainfall: Gamma-distributed with seasonal wet-season boost.
        
        Returns:
            eto: reference ET in mm/day per time step
            rainfall: rainfall in mm/day per time step
        """
        cfg = self.config
        season_len = cfg["season_length_days"]
        
        # ETo — sinusoidal
        phase = 2 * np.pi * (days / season_len - cfg["eto_peak_day_frac"])
        eto = cfg["eto_mean"] + cfg["eto_amplitude"] * np.cos(phase)
        eto = np.maximum(eto, 1.0)
        
        # Rainfall — gamma distributed with wet-season boost
        rainfall = self.rng.gamma(
            shape=cfg["rainfall_shape"],
            scale=cfg["rainfall_mean"] / cfg["rainfall_shape"],
            size=len(days)
        )
        
        # Boost during wet season
        wet_start = cfg["rainfall_wet_season_frac"][0] * season_len
        wet_end = cfg["rainfall_wet_season_frac"][1] * season_len
        wet_mask = (days >= wet_start) & (days <= wet_end)
        rainfall[wet_mask] *= 2.5
        
        # Zero out some dry days
        dry_mask = self.rng.random(len(days)) > 0.4
        rainfall[~wet_mask & dry_mask] = 0
        
        return eto, rainfall
    
    def simulate_season(self, num_plots: int = None,
                         seed: int = None) -> List[PlotSimulation]:
        """
        Simulate a full season for multiple plots.
        
        Each plot gets:
            - A random crop type
            - Randomized sowing date within the crop's window
            - Randomized duration (with std deviation)
            - Optional water stress with random onset/duration/intensity
            - Clean NDVI curve from the double-logistic model
            - Ground-truth stage labels from the same parameters
            - Synthetic weather (ETo, rainfall)
        
        Args:
            num_plots: number of plots to simulate
            seed: random seed for reproducibility
        
        Returns:
            List of PlotSimulation objects
        """
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        
        num_plots = num_plots or self.config["num_plots"]
        cfg = self.config
        season_len = cfg["season_length_days"]
        obs_interval = cfg["observation_interval_days"]
        
        # Observation days (satellite revisit cadence)
        obs_days = np.arange(0, season_len, obs_interval)
        
        crop_keys = list(self.crop_params.keys())
        plots = []
        
        for plot_id in range(num_plots):
            # Random crop assignment
            crop = self.rng.choice(crop_keys)
            p = self.crop_params[crop]
            
            # Randomized sowing and duration
            sow_min, sow_max = p["sowing_window"]
            sowing_day = self.rng.randint(sow_min, sow_max + 1)
            total_duration = int(p["total_duration_days"] + 
                                  self.rng.normal(0, p["duration_std"]))
            total_duration = max(total_duration, 60)
            
            # Generate clean NDVI
            clean_ndvi = self.generate_clean_ndvi(crop, sowing_day, total_duration, obs_days)
            
            # Generate ground-truth stage labels
            stage_ids, stage_name_list = self.generate_stage_labels(
                crop, sowing_day, total_duration, obs_days
            )
            
            # Decide on water stress
            apply_stress = self.rng.random() < cfg["stress_probability"]
            stress_onset = 0
            stress_duration = 0
            stress_intensity = 0.0
            stress_signal = np.zeros(len(obs_days))
            
            if apply_stress:
                onset_frac = self.rng.uniform(*cfg["stress_onset_range"])
                stress_onset = int(sowing_day + onset_frac * total_duration)
                stress_duration = self.rng.randint(*cfg["stress_duration_range"])
                stress_intensity = self.rng.uniform(*cfg["stress_intensity_range"])
                
                clean_ndvi, stress_signal = self.apply_water_stress(
                    clean_ndvi, crop, sowing_day, total_duration, obs_days,
                    stress_onset, stress_duration, stress_intensity, stage_ids
                )
            
            # Generate weather
            eto, rainfall = self.generate_weather(obs_days)
            
            plot = PlotSimulation(
                plot_id=plot_id,
                crop_type=crop,
                crop_label=p["label"],
                sowing_day=sowing_day,
                total_duration=total_duration,
                stress_applied=apply_stress,
                stress_onset_day=stress_onset,
                stress_duration=stress_duration,
                stress_intensity=stress_intensity,
                days=obs_days,
                clean_ndvi=clean_ndvi,
                stage_labels=stage_ids,
                stage_names_ts=stage_name_list,
                stress_signal=stress_signal,
                eto=eto,
                rainfall=rainfall,
            )
            plots.append(plot)
        
        print(f"✅ Simulated {num_plots} plots across {len(crop_keys)} crop types")
        crop_counts = {}
        stress_count = 0
        for pl in plots:
            crop_counts[pl.crop_type] = crop_counts.get(pl.crop_type, 0) + 1
            if pl.stress_applied:
                stress_count += 1
        
        for crop, count in sorted(crop_counts.items()):
            print(f"   {self.crop_params[crop]['display_name']}: {count} plots")
        print(f"   Stressed plots: {stress_count}/{num_plots} ({100*stress_count/num_plots:.0f}%)")
        
        return plots
    
    def plots_to_dataframe(self, plots: List[PlotSimulation]) -> pd.DataFrame:
        """
        Convert list of PlotSimulations into a flat DataFrame
        suitable for downstream analysis and the dashboard.
        """
        records = []
        for plot in plots:
            for i, day in enumerate(plot.days):
                records.append({
                    "plot_id": plot.plot_id,
                    "crop_type": plot.crop_type,
                    "crop_label": plot.crop_label,
                    "crop_display": self.crop_params[plot.crop_type]["display_name"],
                    "day": day,
                    "sowing_day": plot.sowing_day,
                    "total_duration": plot.total_duration,
                    "clean_ndvi": plot.clean_ndvi[i],
                    "stage_id": plot.stage_labels[i],
                    "stage_name": plot.stage_names_ts[i],
                    "stress_applied": plot.stress_applied,
                    "stress_signal": plot.stress_signal[i],
                    "stress_intensity": plot.stress_intensity,
                    "eto": plot.eto[i],
                    "rainfall": plot.rainfall[i],
                })
        
        return pd.DataFrame(records)


# ============================================================
# CLI ENTRY POINT
# ============================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    sim = CropGrowthSimulator(seed=42)
    plots = sim.simulate_season(num_plots=8, seed=42)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharey=True)
    axes = axes.flatten()
    
    for idx, plot in enumerate(plots[:8]):
        ax = axes[idx]
        p = sim.crop_params[plot.crop_type]
        ax.plot(plot.days, plot.clean_ndvi, color=p["color"], linewidth=2,
                label=f'{p["display_name"]}')
        
        # Color background by stage
        stage_colors = ["#FFF9C4", "#C8E6C9", "#BBDEFB", "#FFE0B2", "#E0E0E0"]
        for i in range(len(plot.days) - 1):
            sid = plot.stage_labels[i]
            if sid >= 0:
                ax.axvspan(plot.days[i], plot.days[i+1], alpha=0.2,
                           color=stage_colors[sid])
        
        if plot.stress_applied:
            ax.axvspan(plot.stress_onset_day,
                       plot.stress_onset_day + plot.stress_duration,
                       alpha=0.15, color='red', label='Stress period')
        
        ax.set_title(f"Plot {plot.plot_id}: {p['display_name']}", fontsize=10)
        ax.set_xlabel("Day")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    
    axes[0].set_ylabel("NDVI (clean)")
    plt.suptitle("KrishiDrishti — Parametric Crop Growth Curves (Clean, Pre-noise)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "outputs", "growth_curves.png"), dpi=150)
    plt.show()
    print("\n📊 Growth curves plotted.")
