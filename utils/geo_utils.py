"""
KrishiDrishti — Geospatial & GIS Utility Module
=================================================
Generates realistic field polygon grids (GeoPandas/Shapely) in major
Indian Canal Command Areas, exports GeoJSON/Shapefiles, and builds
interactive Folium map layers for dashboard display.

AOI Support:
1. Indira Gandhi Canal Command Area (Rajasthan/Punjab belt)
2. Godavari Delta Canal Command Area (Andhra Pradesh)
3. Bhakra Nangal Command Area (Haryana/Punjab)

DISCLAIMER: All field polygons and coordinates are synthetically generated
for hackathon demonstration and spatial modeling purposes.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GeoUtils")

# Try importing geospatial libraries with graceful fallback
try:
    import shapely
    from shapely.geometry import Polygon, Point, mapping
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    logger.warning("Shapely not installed. Using manual GeoJSON coordinate arrays.")

try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    logger.warning("GeoPandas not installed. Using standard pandas and GeoJSON export.")

try:
    import folium
    from folium import plugins
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    logger.warning("Folium not installed. Map rendering will be disabled.")

# Define Realistic Indian Canal Command AOIs (Latitude, Longitude center and bbox)
AOI_CATALOG = {
    "Indira Gandhi Canal Command": {
        "center": [29.88, 75.82],
        "state": "Rajasthan / Punjab",
        "canal_name": "Indira Gandhi Canal (IGNP)",
        "bbox": [29.80, 75.72, 29.96, 75.92],
        "default_zoom": 13
    },
    "Godavari Delta Command Area": {
        "center": [16.80, 81.70],
        "state": "Andhra Pradesh",
        "canal_name": "Sir Arthur Cotton Barrage Command",
        "bbox": [16.72, 81.60, 16.88, 81.80],
        "default_zoom": 13
    },
    "Bhakra Nangal Command Area": {
        "center": [29.50, 76.00],
        "state": "Haryana / Punjab",
        "canal_name": "Bhakra Main Line Canal",
        "bbox": [29.42, 75.90, 29.58, 76.10],
        "default_zoom": 13
    }
}


class GeoFieldManager:
    """Manages synthetic field polygons, spatial exports, and interactive GIS map layers."""
    
    def __init__(self, aoi_name: str = "Indira Gandhi Canal Command", seed: int = 42):
        self.aoi_name = aoi_name if aoi_name in AOI_CATALOG else "Indira Gandhi Canal Command"
        self.aoi = AOI_CATALOG[self.aoi_name]
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
    def generate_field_polygons(self, plots_df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign spatial polygon boundaries to each simulated plot in the dataframe.
        Creates a realistic grid with small random variations mimicking farming plots.
        """
        df = plots_df.copy()
        n_plots = len(df)
        
        center_lat, center_lon = self.aoi["center"]
        
        # Calculate grid dimensions
        grid_cols = int(np.ceil(np.sqrt(n_plots * 1.5)))
        grid_rows = int(np.ceil(n_plots / grid_cols))
        
        # Grid spacing (approx 200m x 200m fields ~ 4 hectares each)
        lat_step = 0.0025
        lon_step = 0.0030
        
        start_lat = center_lat - (grid_rows * lat_step) / 2.0
        start_lon = center_lon - (grid_cols * lon_step) / 2.0
        
        polygons = []
        centroids_lat = []
        centroids_lon = []
        geojson_geoms = []
        
        for i in range(n_plots):
            row_idx = i // grid_cols
            col_idx = i % grid_cols
            
            # Base corner coordinate
            base_lat = start_lat + row_idx * (lat_step + 0.0004) + self.rng.uniform(-0.0002, 0.0002)
            base_lon = start_lon + col_idx * (lon_step + 0.0004) + self.rng.uniform(-0.0002, 0.0002)
            
            # Create irregular 4-sided field polygon
            w_lat = lat_step * self.rng.uniform(0.85, 1.15)
            w_lon = lon_step * self.rng.uniform(0.85, 1.15)
            
            coords = [
                [base_lon, base_lat],
                [base_lon + w_lon, base_lat + self.rng.uniform(-0.0003, 0.0003)],
                [base_lon + w_lon + self.rng.uniform(-0.0003, 0.0003), base_lat + w_lat],
                [base_lon + self.rng.uniform(-0.0003, 0.0003), base_lat + w_lat],
                [base_lon, base_lat] # Close loop
            ]
            
            c_lat = base_lat + w_lat / 2.0
            c_lon = base_lon + w_lon / 2.0
            
            centroids_lat.append(c_lat)
            centroids_lon.append(c_lon)
            
            geom_dict = {"type": "Polygon", "coordinates": [coords]}
            geojson_geoms.append(geom_dict)
            
            if SHAPELY_AVAILABLE:
                polygons.append(Polygon(coords))
            else:
                polygons.append(str(coords))
                
        df["latitude"] = centroids_lat
        df["longitude"] = centroids_lon
        df["geojson_geometry"] = geojson_geoms
        
        if SHAPELY_AVAILABLE and GEOPANDAS_AVAILABLE:
            gdf = gpd.GeoDataFrame(df, geometry=polygons, crs="EPSG:4326")
            return gdf
            
        df["geometry"] = polygons
        return df

    def export_spatial_data(self, df: pd.DataFrame, output_dir: str = "outputs/geospatial") -> Dict[str, str]:
        """
        Export field polygons and attributes to GeoJSON and Shapefile formats.
        Returns paths to the generated files.
        """
        os.makedirs(output_dir, exist_ok=True)
        paths = {}
        
        # 1. GeoJSON Export (Universal standard)
        geojson_path = os.path.join(output_dir, f"krishidrishti_fields_{self.aoi_name.lower().replace(' ', '_')}.geojson")
        
        features = []
        for idx, row in df.iterrows():
            geom = row.get("geojson_geometry")
            if not geom and SHAPELY_AVAILABLE and hasattr(row, "geometry"):
                try:
                    geom = mapping(row["geometry"])
                except Exception:
                    geom = None
            
            # Clean attributes for JSON serialization
            props = {}
            for col in df.columns:
                if col not in ["geometry", "geojson_geometry"]:
                    val = row[col]
                    if isinstance(val, (np.integer, np.floating)):
                        val = val.item()
                    elif pd.isna(val):
                        val = None
                    props[col] = val
                    
            if geom:
                features.append({
                    "type": "Feature",
                    "id": str(row.get("plot_id", idx)),
                    "geometry": geom,
                    "properties": props
                })
                
        geojson_obj = {
            "type": "FeatureCollection",
            "metadata": {
                "project": "KrishiDrishti AI Crop Intelligence",
                "aoi": self.aoi_name,
                "canal": self.aoi["canal_name"],
                "total_fields": len(features)
            },
            "features": features
        }
        
        with open(geojson_path, "w", encoding="utf-8") as f:
            json.dump(geojson_obj, f, indent=2)
        paths["geojson"] = geojson_path
        logger.info(f"✅ Exported GeoJSON to {geojson_path}")
        
        # 2. Shapefile Export (via GeoPandas if available)
        if GEOPANDAS_AVAILABLE and isinstance(df, gpd.GeoDataFrame):
            try:
                shp_dir = os.path.join(output_dir, "shapefile_export")
                os.makedirs(shp_dir, exist_ok=True)
                shp_path = os.path.join(shp_dir, "krishidrishti_fields.shp")
                
                # Shapefiles require string column names <= 10 chars
                gdf_export = df.copy()
                if "geojson_geometry" in gdf_export.columns:
                    gdf_export = gdf_export.drop(columns=["geojson_geometry"])
                gdf_export.columns = [str(c)[:10] for c in gdf_export.columns]
                
                gdf_export.to_file(shp_path, driver="ESRI Shapefile")
                paths["shapefile"] = shp_path
                logger.info(f"✅ Exported Shapefile to {shp_path}")
            except Exception as e:
                logger.warning(f"⚠️ Shapefile export failed: {str(e)}. GeoJSON is available.")
        else:
            logger.info("ℹ️ GeoPandas not available or not a GeoDataFrame; skipped ESRI Shapefile export.")
            
        return paths

    def create_interactive_map(self, df: pd.DataFrame, layer_type: str = "advisory") -> Any:
        """
        Create a rich Folium interactive map displaying simulated fields, canal command
        boundaries, and color-coded stress/advisory layers.
        """
        if not FOLIUM_AVAILABLE:
            logger.warning("Folium not installed. Cannot generate interactive map.")
            return None
            
        center_lat, center_lon = self.aoi["center"]
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=self.aoi["default_zoom"],
            tiles="CartoDB dark_matter"
        )
        
        # Add base layers
        folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
        folium.TileLayer("Stamen Terrain", name="Terrain").add_to(m)
        
        # Draw Canal Command Main Line (Synthetic Polyline through the AOI)
        canal_coords = [
            [center_lat - 0.08, center_lon - 0.08],
            [center_lat - 0.02, center_lon - 0.03],
            [center_lat + 0.04, center_lon + 0.05],
            [center_lat + 0.09, center_lon + 0.09]
        ]
        folium.PolyLine(
            locations=canal_coords,
            color="#00CED1",
            weight=5,
            opacity=0.8,
            popup=f"<b>{self.aoi['canal_name']}</b><br>Main Feeder Channel",
            tooltip=f"🌊 {self.aoi['canal_name']} (Main Canal)"
        ).add_to(m)
        
        # Color mapping rules
        def get_color(row):
            if layer_type == "advisory":
                adv = row.get("advisory", row.get("advisory_pred", 0))
                if adv == 2 or "Critical" in str(row.get("Status", "")): return "#FF0000" # Red
                if adv == 1 or "Watch" in str(row.get("Status", "")): return "#FFDD00"    # Yellow/Orange
                return "#00AA00" # Green
            elif layer_type == "crop":
                crop = str(row.get("crop", row.get("crop_display", "Rice")))
                colors = {"Rice": "#228B22", "Wheat": "#DAA520", "Cotton": "#FF8C00", "Sugarcane": "#9370DB", "Maize": "#FFD700"}
                for k, v in colors.items():
                    if k in crop: return v
                return "#32CD32"
            return "#52b788"

        # Add field polygons
        for idx, row in df.iterrows():
            geom = row.get("geojson_geometry")
            if not geom: continue
            
            color = get_color(row)
            plot_id = row.get("plot_id", f"F-{idx:03d}")
            crop = row.get("crop", row.get("crop_display", "Crop"))
            stage = row.get("stage", "Vegetative")
            deficit = row.get("deficit_mm", row.get("Deficit (mm)", 0.0))
            status = row.get("status", row.get("Status", "Adequate"))
            
            popup_html = f"""
            <div style="font-family: Arial, sans-serif; width: 220px; padding: 5px;">
                <h4 style="margin: 0 0 5px 0; color: #1a472a;">🌾 Field {plot_id}</h4>
                <hr style="margin: 5px 0;">
                <b>Crop Type:</b> {crop}<br>
                <b>Growth Stage:</b> {stage}<br>
                <b>Water Deficit:</b> {float(deficit):.1f} mm/8d<br>
                <b>Advisory Status:</b> <span style="color: {color}; font-weight: bold;">{status}</span><br>
                <hr style="margin: 5px 0;">
                <small>📍 {self.aoi_name}</small>
            </div>
            """
            
            folium.GeoJson(
                geom,
                style_function=lambda x, col=color: {
                    "fillColor": col,
                    "color": "#ffffff",
                    "weight": 1.5,
                    "fillOpacity": 0.65
                },
                highlight_function=lambda x: {
                    "fillOpacity": 0.9,
                    "weight": 3,
                    "color": "#00FFFF"
                },
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"Field {plot_id} ({crop}) — {status}"
            ).add_to(m)
            
        folium.LayerControl().add_to(m)
        return m


# ============================================================
# CLI TEST / DEMO
# ============================================================
if __name__ == "__main__":
    print("='*60")
    print("  Testing KrishiDrishti GeoUtils (Spatial & GIS Stack)")
    print("='*60")
    
    # Create synthetic plot records
    demo_df = pd.DataFrame({
        "plot_id": [f"F-{i:03d}" for i in range(12)],
        "crop": ["Rice", "Wheat", "Cotton", "Sugarcane"] * 3,
        "stage": ["Vegetative", "Flowering", "Maturity", "Harvest"] * 3,
        "deficit_mm": np.random.uniform(0, 35, 12).round(1),
        "advisory": [0, 1, 2, 0] * 3,
        "Status": ["🟢 Adequate", "🟡 Watch", "🔴 Critical", "🟢 Adequate"] * 3
    })
    
    geo_mgr = GeoFieldManager(aoi_name="Indira Gandhi Canal Command")
    gdf = geo_mgr.generate_field_polygons(demo_df)
    
    print(f"\n✅ Generated {len(gdf)} field polygons in AOI: {geo_mgr.aoi_name}")
    print(f"   Center Lat/Lon: {geo_mgr.aoi['center']}")
    
    exports = geo_mgr.export_spatial_data(gdf, output_dir="outputs/geospatial_test")
    print(f"\n✅ Exported Files:\n  • GeoJSON: {exports.get('geojson')}")
    if "shapefile" in exports:
        print(f"  • Shapefile: {exports.get('shapefile')}")
        
    map_obj = geo_mgr.create_interactive_map(gdf, layer_type="advisory")
    if map_obj:
        map_path = "outputs/geospatial_test/demo_map.html"
        map_obj.save(map_path)
        print(f"✅ Interactive Folium Map saved to: {map_path}")
        
    print("='*60")
