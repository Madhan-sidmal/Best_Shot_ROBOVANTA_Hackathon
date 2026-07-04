"""
Irrigation Advisory Module — FAO-56 Crop Water Balance
========================================================
Estimates crop water demand (ETc), computes water deficit,
and generates 8-day irrigation advisory maps for canal command areas.

Methodology:
    ETc = Kc × ETo  (FAO-56 single crop coefficient approach)
    Water Deficit = ETc - Effective Rainfall
    Advisory = Classified based on deficit thresholds

References:
    FAO Irrigation and Drainage Paper No. 56
    Allen, R.G., Pereira, L.S., Raes, D., Smith, M. (1998)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import FAO56_KC, IRRIGATION_ADVISORY, OUTPUT_DIR, CROP_CLASSES


class IrrigationAdvisory:
    """
    FAO-56 based irrigation advisory generator.
    
    Pipeline:
        1. Get crop type and growth stage (from Pillar 1 & 2)
        2. Look up Kc value for crop × stage
        3. Compute ETc = Kc × ETo
        4. Compute effective rainfall
        5. Compute water deficit = ETc - effective rainfall
        6. Classify irrigation advisory status
        7. Generate advisory map and pixel-level recommendations
    """
    
    def __init__(self, kc_table=None, advisory_config=None):
        """
        Args:
            kc_table: dict — Crop coefficient table (crop → stage → Kc)
            advisory_config: dict — Deficit thresholds and advisory classes
        """
        self.kc_table = kc_table or FAO56_KC
        self.advisory_config = advisory_config or IRRIGATION_ADVISORY
    
    # ============================================================
    # CROP COEFFICIENT (Kc) LOOKUP
    # ============================================================
    def get_kc(self, crop_name, growth_stage):
        """
        Look up crop coefficient for a given crop and growth stage.
        
        Args:
            crop_name: str — Crop type (e.g., "Rice (Paddy)")
            growth_stage: str — Stage name ("Germination", "Vegetative", etc.)
        
        Returns:
            float — Kc value
        """
        # Map growth stage to FAO-56 stage names
        stage_mapping = {
            'Germination': 'initial',
            'Pre-season': 'initial',
            'Vegetative': 'development',
            'Reproductive': 'mid_season',
            'Maturity': 'late_season',
            'Post-harvest': 'late_season'
        }
        
        fao_stage = stage_mapping.get(growth_stage, 'mid_season')
        
        # Look up in Kc table
        crop_key = None
        for key in self.kc_table:
            if key.lower() in crop_name.lower() or crop_name.lower() in key.lower():
                crop_key = key
                break
        
        if crop_key and fao_stage in self.kc_table[crop_key]:
            return self.kc_table[crop_key][fao_stage]
        else:
            # Default Kc if crop not found
            return 1.0
    
    # ============================================================
    # EVAPOTRANSPIRATION CALCULATION
    # ============================================================
    def compute_etc(self, eto, kc):
        """
        Compute crop evapotranspiration.
        
        ETc = Kc × ETo
        
        Args:
            eto: float or np.array — Reference ET (mm/day or mm/8-day)
            kc: float or np.array — Crop coefficient
        
        Returns:
            float or np.array — Crop ET (mm/day or mm/8-day)
        """
        return kc * eto
    
    def compute_effective_rainfall(self, rainfall, method='fao'):
        """
        Compute effective rainfall (portion actually used by crops).
        
        Args:
            rainfall: float or np.array — Total rainfall (mm/8-day)
            method: str — 'fao' for FAO method, 'fixed' for fixed factor
        
        Returns:
            float or np.array — Effective rainfall (mm/8-day)
        """
        if method == 'fao':
            # FAO method for effective rainfall
            # Pe = 0.8P - 25 for P > 75 mm/month; Pe = 0.6P - 10 for P ≤ 75mm/month
            # Simplified for 8-day: use fixed factor
            factor = self.advisory_config['effective_rainfall_factor']
            return np.maximum(rainfall * factor, 0)
        else:
            return rainfall * 0.8
    
    # ============================================================
    # WATER DEFICIT CALCULATION
    # ============================================================
    def compute_water_deficit(self, etc, effective_rainfall):
        """
        Compute crop water deficit.
        
        Deficit = ETc - Effective Rainfall
        
        Positive deficit = crop needs more water
        Negative deficit = surplus (no irrigation needed)
        
        Args:
            etc: float or np.array — Crop ET (mm/8-day)
            effective_rainfall: float or np.array — Effective rainfall (mm/8-day)
        
        Returns:
            float or np.array — Water deficit (mm/8-day)
        """
        deficit = etc - effective_rainfall
        return np.maximum(deficit, 0)  # Deficit can't be negative for advisory
    
    # ============================================================
    # ADVISORY CLASSIFICATION
    # ============================================================
    def classify_advisory(self, deficit):
        """
        Classify irrigation advisory based on water deficit.
        
        Args:
            deficit: float or np.array — Water deficit (mm/8-day)
        
        Returns:
            int or np.array — Advisory class (0-3)
            str or list — Advisory message
        """
        thresholds = self.advisory_config['deficit_thresholds']
        classes = self.advisory_config['classes']
        
        if isinstance(deficit, (int, float)):
            if deficit <= thresholds['adequate']:
                cls = 0
            elif deficit <= thresholds['watch']:
                cls = 1
            elif deficit <= thresholds['urgent']:
                cls = 2
            else:
                cls = 3
            
            message = classes[cls]['message'].format(deficit=f"{deficit:.1f}")
            return cls, message
        
        # Array version
        advisory_class = np.zeros_like(deficit, dtype=int)
        advisory_class = np.where(deficit <= thresholds['adequate'], 0, advisory_class)
        advisory_class = np.where(
            (deficit > thresholds['adequate']) & (deficit <= thresholds['watch']),
            1, advisory_class
        )
        advisory_class = np.where(
            (deficit > thresholds['watch']) & (deficit <= thresholds['urgent']),
            2, advisory_class
        )
        advisory_class = np.where(deficit > thresholds['urgent'], 3, advisory_class)
        
        return advisory_class
    
    # ============================================================
    # FULL ADVISORY PIPELINE
    # ============================================================
    def generate_advisory(self, crop_types, growth_stages, eto_values, rainfall_values):
        """
        Generate complete irrigation advisory for all pixels.
        
        Args:
            crop_types: np.array — Crop type labels (strings)
            growth_stages: np.array — Growth stage labels (strings)
            eto_values: np.array — Reference ET per pixel (mm/8-day)
            rainfall_values: np.array — Rainfall per pixel (mm/8-day)
        
        Returns:
            pd.DataFrame — Pixel-level irrigation advisory
        """
        n_pixels = len(crop_types)
        
        results = []
        for i in range(n_pixels):
            crop = crop_types[i]
            stage = growth_stages[i]
            eto = eto_values[i]
            rain = rainfall_values[i]
            
            # Get Kc
            kc = self.get_kc(crop, stage)
            
            # Compute ETc
            etc = self.compute_etc(eto, kc)
            
            # Effective rainfall
            pe = self.compute_effective_rainfall(rain)
            
            # Water deficit
            deficit = self.compute_water_deficit(etc, pe)
            
            # Advisory classification
            adv_class, adv_message = self.classify_advisory(deficit)
            
            results.append({
                'pixel_id': i,
                'crop_type': crop,
                'growth_stage': stage,
                'kc': kc,
                'eto_mm_8day': eto,
                'etc_mm_8day': etc,
                'rainfall_mm_8day': rain,
                'effective_rainfall_mm_8day': pe,
                'water_deficit_mm_8day': deficit,
                'irrigation_depth_mm': max(0, deficit * 1.1),  # 10% extra for efficiency
                'advisory_class': adv_class,
                'advisory_status': self.advisory_config['classes'][adv_class]['name'],
                'advisory_emoji': self.advisory_config['classes'][adv_class]['emoji'],
                'advisory_message': adv_message
            })
        
        return pd.DataFrame(results)
    
    # ============================================================
    # RASTER-BASED ADVISORY MAP
    # ============================================================
    def generate_advisory_raster(self, crop_map, stage_map, eto_raster, rainfall_raster):
        """
        Generate an advisory raster (2D array) for map visualization.
        
        Args:
            crop_map: 2D np.array — Crop type IDs
            stage_map: 2D np.array — Growth stage IDs
            eto_raster: 2D np.array — Reference ET (mm/8-day)
            rainfall_raster: 2D np.array — Rainfall (mm/8-day)
        
        Returns:
            advisory_raster: 2D np.array — Advisory class per pixel
            deficit_raster: 2D np.array — Water deficit per pixel
        """
        h, w = crop_map.shape
        advisory_raster = np.zeros((h, w), dtype=int)
        deficit_raster = np.zeros((h, w))
        
        # Stage name mapping
        stage_names = {0: 'Germination', 1: 'Vegetative', 2: 'Reproductive', 3: 'Maturity'}
        
        # Crop name mapping (from CROP_CLASSES config)
        crop_names = {k: v['name'] for k, v in CROP_CLASSES.items()}
        
        for i in range(h):
            for j in range(w):
                crop_id = crop_map[i, j]
                stage_id = stage_map[i, j]
                
                crop_name = crop_names.get(crop_id, 'Other Crops')
                stage_name = stage_names.get(stage_id, 'Vegetative')
                
                kc = self.get_kc(crop_name, stage_name)
                etc = self.compute_etc(eto_raster[i, j], kc)
                pe = self.compute_effective_rainfall(rainfall_raster[i, j])
                deficit = self.compute_water_deficit(etc, pe)
                
                advisory_raster[i, j] = self.classify_advisory(deficit) if isinstance(
                    self.classify_advisory(deficit), int
                ) else self.classify_advisory(deficit)[0]
                deficit_raster[i, j] = deficit
        
        return advisory_raster, deficit_raster
    
    # ============================================================
    # VISUALIZATION
    # ============================================================
    def plot_advisory_map(self, advisory_raster, deficit_raster, title="8-Day Irrigation Advisory"):
        """Create a color-coded irrigation advisory map."""
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Advisory Map
        cmap_advisory = mcolors.ListedColormap(['#00AA00', '#FFDD00', '#FF8800', '#FF0000'])
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
        norm = mcolors.BoundaryNorm(bounds, cmap_advisory.N)
        
        im1 = axes[0].imshow(advisory_raster, cmap=cmap_advisory, norm=norm, interpolation='nearest')
        axes[0].set_title('Irrigation Advisory Status', fontsize=14, fontweight='bold')
        
        # Custom colorbar
        cbar1 = plt.colorbar(im1, ax=axes[0], ticks=[0, 1, 2, 3], shrink=0.8)
        cbar1.ax.set_yticklabels(['🟢 Adequate', '🟡 Watch', '🟠 Urgent', '🔴 Critical'])
        
        # Deficit Map
        im2 = axes[1].imshow(deficit_raster, cmap='YlOrRd', interpolation='nearest')
        axes[1].set_title('Water Deficit (mm/8-day)', fontsize=14, fontweight='bold')
        cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.8)
        cbar2.set_label('Deficit (mm)')
        
        plt.suptitle(f'🚿 KrishiDrishti — {title}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(OUTPUT_DIR, 'irrigation_advisory_map.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"🗺️ Advisory map saved to: {save_path}")
    
    def plot_advisory_summary(self, advisory_df):
        """Create summary charts for the advisory."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Advisory distribution (pie)
        status_counts = advisory_df['advisory_status'].value_counts()
        colors = ['#00AA00', '#FFDD00', '#FF8800', '#FF0000']
        axes[0].pie(
            status_counts.values, labels=status_counts.index,
            colors=colors[:len(status_counts)],
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11}
        )
        axes[0].set_title('Advisory Distribution', fontsize=13, fontweight='bold')
        
        # 2. Deficit by crop type (box plot)
        advisory_df.boxplot(column='water_deficit_mm_8day', by='crop_type', ax=axes[1], rot=45)
        axes[1].set_title('Water Deficit by Crop', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('')
        axes[1].set_ylabel('Deficit (mm/8-day)')
        plt.sca(axes[1])
        plt.title('Water Deficit by Crop', fontsize=13, fontweight='bold')
        
        # 3. Deficit by growth stage
        advisory_df.boxplot(column='water_deficit_mm_8day', by='growth_stage', ax=axes[2], rot=45)
        axes[2].set_title('Water Deficit by Growth Stage', fontsize=13, fontweight='bold')
        axes[2].set_xlabel('')
        axes[2].set_ylabel('Deficit (mm/8-day)')
        plt.sca(axes[2])
        plt.title('Water Deficit by Growth Stage', fontsize=13, fontweight='bold')
        
        plt.suptitle('🚿 KrishiDrishti — Irrigation Advisory Summary', fontsize=15, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(OUTPUT_DIR, 'advisory_summary.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"📊 Advisory summary saved to: {save_path}")
    
    def generate_sms_advisory(self, advisory_df, field_id=None):
        """
        Generate SMS-ready advisory messages.
        
        This is the out-of-box feature for farmer communication.
        """
        messages = []
        
        for _, row in advisory_df.iterrows():
            msg = (
                f"{row['advisory_emoji']} KrishiDrishti Alert\n"
                f"Crop: {row['crop_type']}\n"
                f"Stage: {row['growth_stage']}\n"
                f"Status: {row['advisory_status']}\n"
                f"Water Deficit: {row['water_deficit_mm_8day']:.1f} mm\n"
                f"Recommendation: {row['advisory_message']}\n"
                f"---"
            )
            messages.append(msg)
        
        return messages


# ============================================================
# DEMO
# ============================================================
def demo_irrigation_advisory():
    """Demonstrate the irrigation advisory with synthetic data."""
    print("🚿 KrishiDrishti — Irrigation Advisory Demo")
    print("=" * 60)
    
    np.random.seed(42)
    n_pixels = 100
    
    # Simulated inputs
    crops = np.random.choice(
        ['Rice (Paddy)', 'Cotton', 'Maize', 'Soybean', 'Sugarcane'],
        n_pixels
    )
    stages = np.random.choice(
        ['Germination', 'Vegetative', 'Reproductive', 'Maturity'],
        n_pixels
    )
    eto = np.random.uniform(3, 8, n_pixels) * 8  # mm/8-day (3-8 mm/day × 8 days)
    rainfall = np.random.exponential(10, n_pixels)  # mm/8-day
    
    # Generate advisory
    advisor = IrrigationAdvisory()
    advisory_df = advisor.generate_advisory(crops, stages, eto, rainfall)
    
    print("\n📋 Sample Advisory Output:")
    print(advisory_df[['crop_type', 'growth_stage', 'kc', 'etc_mm_8day', 
                         'water_deficit_mm_8day', 'advisory_status', 'advisory_emoji']].head(15).to_string())
    
    # Summary statistics
    print("\n📊 Advisory Summary:")
    print(advisory_df['advisory_status'].value_counts())
    print(f"\nMean Water Deficit: {advisory_df['water_deficit_mm_8day'].mean():.1f} mm/8-day")
    print(f"Max Water Deficit:  {advisory_df['water_deficit_mm_8day'].max():.1f} mm/8-day")
    
    # Visualizations
    advisor.plot_advisory_summary(advisory_df)
    
    # Raster demo
    print("\n🗺️ Generating advisory raster map...")
    crop_map = np.random.randint(1, 6, (50, 50))
    stage_map = np.random.randint(0, 4, (50, 50))
    eto_raster = np.random.uniform(30, 60, (50, 50))
    rain_raster = np.random.exponential(15, (50, 50))
    
    advisory_raster, deficit_raster = advisor.generate_advisory_raster(
        crop_map, stage_map, eto_raster, rain_raster
    )
    advisor.plot_advisory_map(advisory_raster, deficit_raster)
    
    # SMS demo
    print("\n📱 Sample SMS Advisories:")
    sms_messages = advisor.generate_sms_advisory(advisory_df.head(3))
    for msg in sms_messages:
        print(msg)
    
    return advisory_df


if __name__ == '__main__':
    advisory_df = demo_irrigation_advisory()
