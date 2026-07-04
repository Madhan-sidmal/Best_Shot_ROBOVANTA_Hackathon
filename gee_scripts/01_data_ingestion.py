"""
GEE Data Ingestion & Preprocessing
====================================
Script 01: Loads Sentinel-1, Sentinel-2, and ancillary data from Google Earth Engine.
Performs cloud masking, speckle filtering, and temporal compositing.

Usage:
    Run in Google Colab with Earth Engine authenticated, or locally with `earthengine authenticate`.
"""

import ee
import geemap
import numpy as np
from datetime import datetime, timedelta

# Initialize Earth Engine
try:
    ee.Initialize(project='your-gee-project-id')  # Replace with your project ID
except Exception:
    ee.Authenticate()
    ee.Initialize(project='your-gee-project-id')


# ============================================================
# STUDY AREA DEFINITION
# ============================================================
def define_roi(coordinates):
    """
    Define Region of Interest from coordinate list.
    
    Args:
        coordinates: List of [lon, lat] pairs forming a polygon
    
    Returns:
        ee.Geometry.Polygon
    """
    return ee.Geometry.Polygon(coordinates)


# Example ROI — Replace with actual pilot area coordinates
ROI = define_roi([
    [78.0, 17.0], [78.5, 17.0],
    [78.5, 17.5], [78.0, 17.5],
    [78.0, 17.0]
])

# Date ranges
START_DATE = '2024-06-01'
END_DATE = '2024-12-31'


# ============================================================
# SENTINEL-2 OPTICAL DATA
# ============================================================
def mask_s2_clouds(image):
    """
    Cloud masking for Sentinel-2 using the SCL band (Scene Classification Layer).
    Removes cloud, cloud shadow, cirrus, and snow pixels.
    """
    scl = image.select('SCL')
    # SCL values: 3=Cloud Shadow, 7=Unclassified, 8=Cloud Medium, 9=Cloud High, 10=Cirrus
    cloud_mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return image.updateMask(cloud_mask).divide(10000)  # Scale to reflectance [0,1]


def add_vegetation_indices(image):
    """
    Compute vegetation and moisture indices from Sentinel-2 bands.
    
    Indices computed:
        - NDVI: Normalized Difference Vegetation Index
        - EVI: Enhanced Vegetation Index
        - NDWI: Normalized Difference Water Index
        - NDMI: Normalized Difference Moisture Index
        - SAVI: Soil Adjusted Vegetation Index
    """
    # Band aliases for readability
    blue = image.select('B2')
    green = image.select('B3')
    red = image.select('B4')
    re1 = image.select('B5')    # Red Edge 1
    re2 = image.select('B6')    # Red Edge 2
    re3 = image.select('B7')    # Red Edge 3
    nir = image.select('B8')
    nir_narrow = image.select('B8A')
    swir1 = image.select('B11')
    swir2 = image.select('B12')
    
    # NDVI
    ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
    
    # EVI — Enhanced, less saturation in dense vegetation
    evi = image.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
        {'NIR': nir, 'RED': red, 'BLUE': blue}
    ).rename('EVI')
    
    # NDWI — Water content
    ndwi = green.subtract(nir).divide(green.add(nir)).rename('NDWI')
    
    # NDMI — Moisture Index (canopy moisture)
    ndmi = nir.subtract(swir1).divide(nir.add(swir1)).rename('NDMI')
    
    # SAVI — Works better in sparse vegetation (soil influence reduction)
    savi = image.expression(
        '1.5 * (NIR - RED) / (NIR + RED + 0.5)',
        {'NIR': nir, 'RED': red}
    ).rename('SAVI')
    
    # LSWI — Land Surface Water Index
    lswi = nir.subtract(swir1).divide(nir.add(swir1)).rename('LSWI')
    
    return image.addBands([ndvi, evi, ndwi, ndmi, savi, lswi])


def get_sentinel2_collection(roi, start_date, end_date, max_cloud=30):
    """
    Load Sentinel-2 Surface Reflectance collection with cloud masking and indices.
    
    Args:
        roi: ee.Geometry — Region of Interest
        start_date: str — Start date (YYYY-MM-DD)
        end_date: str — End date (YYYY-MM-DD)
        max_cloud: int — Maximum cloud coverage percentage
    
    Returns:
        ee.ImageCollection with vegetation indices
    """
    collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud))
        .map(mask_s2_clouds)
        .map(add_vegetation_indices)
        .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12',
                 'NDVI', 'EVI', 'NDWI', 'NDMI', 'SAVI', 'LSWI'])
    )
    
    print(f"Sentinel-2 images found: {collection.size().getInfo()}")
    return collection


# ============================================================
# SENTINEL-1 SAR DATA
# ============================================================
def add_sar_features(image):
    """
    Compute SAR-derived features from Sentinel-1 backscatter.
    
    Features:
        - VV, VH: Raw backscatter (already in dB for GRD)
        - VH_VV_ratio: Cross-pol ratio (sensitive to vegetation structure)
        - RVI: Radar Vegetation Index
    """
    vv = image.select('VV')
    vh = image.select('VH')
    
    # Cross-pol ratio (VH/VV in dB = VH - VV)
    ratio = vh.subtract(vv).rename('VH_VV_ratio')
    
    # Radar Vegetation Index (4 * VH / (VV + VH)) — in linear scale
    vv_linear = ee.Image(10).pow(vv.divide(10))
    vh_linear = ee.Image(10).pow(vh.divide(10))
    rvi = vh_linear.multiply(4).divide(vv_linear.add(vh_linear)).rename('RVI')
    
    return image.addBands([ratio, rvi])


def refined_lee_filter(image):
    """
    Apply Refined Lee speckle filter to Sentinel-1 imagery.
    This is a simplified version using a focal mean as proxy.
    For production, use SNAP or the refined Lee implementation.
    
    Note: In GEE, true Refined Lee is complex. We use a 
    focal median as a practical approximation for hackathon speed.
    """
    # Apply focal median (7x7 kernel) as speckle reduction
    smoothed_vv = image.select('VV').focal_median(radius=30, kernelType='circle', units='meters').rename('VV')
    smoothed_vh = image.select('VH').focal_median(radius=30, kernelType='circle', units='meters').rename('VH')
    
    return image.addBands([smoothed_vv, smoothed_vh], overwrite=True)


def get_sentinel1_collection(roi, start_date, end_date, orbit='DESCENDING'):
    """
    Load Sentinel-1 GRD collection with speckle filtering and SAR features.
    
    Args:
        roi: ee.Geometry
        start_date, end_date: str
        orbit: str — 'ASCENDING' or 'DESCENDING'
    
    Returns:
        ee.ImageCollection with VV, VH, VH_VV_ratio, RVI
    """
    collection = (
        ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        .filter(ee.Filter.eq('orbitProperties_pass', orbit))
        .select(['VV', 'VH'])
        .map(refined_lee_filter)
        .map(add_sar_features)
    )
    
    print(f"Sentinel-1 images found: {collection.size().getInfo()}")
    return collection


# ============================================================
# TEMPORAL COMPOSITING
# ============================================================
def create_temporal_composites(collection, roi, start_date, end_date, interval_days=8):
    """
    Create N-day temporal composites using median reduction.
    This aligns optical and SAR data to a common temporal grid.
    
    Args:
        collection: ee.ImageCollection
        roi: ee.Geometry
        start_date: str — YYYY-MM-DD
        end_date: str — YYYY-MM-DD
        interval_days: int — Composite interval (default: 8 days)
    
    Returns:
        ee.ImageCollection of composites
    """
    start = ee.Date(start_date)
    end = ee.Date(end_date)
    n_intervals = end.difference(start, 'day').divide(interval_days).ceil()
    
    def make_composite(i):
        i = ee.Number(i)
        composite_start = start.advance(i.multiply(interval_days), 'day')
        composite_end = composite_start.advance(interval_days, 'day')
        
        composite = collection.filterDate(composite_start, composite_end).median()
        
        return composite.set({
            'system:time_start': composite_start.millis(),
            'composite_start': composite_start.format('YYYY-MM-dd'),
            'composite_end': composite_end.format('YYYY-MM-dd')
        })
    
    sequence = ee.List.sequence(0, n_intervals.subtract(1))
    composites = ee.ImageCollection.fromImages(sequence.map(make_composite))
    
    return composites


# ============================================================
# DATA FUSION: OPTICAL + SAR
# ============================================================
def fuse_optical_sar(s2_composites, s1_composites, roi, start_date, end_date, interval_days=8):
    """
    Fuse Sentinel-2 and Sentinel-1 composites on the same temporal grid.
    
    This creates a combined feature stack per time step with both
    spectral indices and SAR backscatter features.
    
    Returns:
        ee.ImageCollection of fused composites
    """
    start = ee.Date(start_date)
    end = ee.Date(end_date)
    n_intervals = end.difference(start, 'day').divide(interval_days).ceil()
    
    def fuse_interval(i):
        i = ee.Number(i)
        t_start = start.advance(i.multiply(interval_days), 'day')
        t_end = t_start.advance(interval_days, 'day')
        
        # Optical composite
        s2_comp = s2_composites.filterDate(t_start, t_end).median()
        
        # SAR composite
        s1_comp = s1_composites.filterDate(t_start, t_end).median()
        
        # Fuse by stacking bands
        fused = s2_comp.addBands(s1_comp)
        
        return fused.set({
            'system:time_start': t_start.millis(),
            'date': t_start.format('YYYY-MM-dd')
        })
    
    sequence = ee.List.sequence(0, n_intervals.subtract(1))
    fused_collection = ee.ImageCollection.fromImages(sequence.map(fuse_interval))
    
    return fused_collection


# ============================================================
# MODIS ET DATA (for irrigation advisory)
# ============================================================
def get_modis_et(roi, start_date, end_date):
    """
    Load MODIS 8-day ET product (MOD16A2) for reference ET computation.
    
    Returns:
        ee.ImageCollection of ET values (scaled to mm/8-day)
    """
    et_collection = (
        ee.ImageCollection('MODIS/061/MOD16A2')
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .select(['ET', 'PET'])  # Actual ET and Potential ET
    )
    
    # Scale factor: values are in kg/m²/8-day (= mm/8-day) * 0.1
    def scale_et(image):
        return image.multiply(0.1).copyProperties(image, ['system:time_start'])
    
    return et_collection.map(scale_et)


# ============================================================
# MAIN PIPELINE
# ============================================================
def run_data_ingestion(roi, start_date, end_date, interval_days=8):
    """
    Main data ingestion pipeline.
    Loads, preprocesses, and fuses all satellite data sources.
    
    Returns:
        dict with 's2', 's1', 'fused', 'modis_et' collections
    """
    print("=" * 60)
    print("KrishiDrishti — Data Ingestion Pipeline")
    print("=" * 60)
    
    # Step 1: Load Sentinel-2
    print("\n[1/5] Loading Sentinel-2 optical data...")
    s2 = get_sentinel2_collection(roi, start_date, end_date)
    
    # Step 2: Load Sentinel-1
    print("[2/5] Loading Sentinel-1 SAR data...")
    s1 = get_sentinel1_collection(roi, start_date, end_date)
    
    # Step 3: Create temporal composites
    print(f"[3/5] Creating {interval_days}-day composites...")
    s2_composites = create_temporal_composites(s2, roi, start_date, end_date, interval_days)
    s1_composites = create_temporal_composites(s1, roi, start_date, end_date, interval_days)
    
    # Step 4: Fuse optical + SAR
    print("[4/5] Fusing optical and SAR data...")
    fused = fuse_optical_sar(s2_composites, s1_composites, roi, start_date, end_date, interval_days)
    
    # Step 5: Load MODIS ET
    print("[5/5] Loading MODIS ET data...")
    modis_et = get_modis_et(roi, start_date, end_date)
    
    print("\n✅ Data ingestion complete!")
    print(f"   Fused composites: ~{365 // interval_days} time steps")
    
    return {
        's2_raw': s2,
        's1_raw': s1,
        's2_composites': s2_composites,
        's1_composites': s1_composites,
        'fused': fused,
        'modis_et': modis_et,
        'roi': roi
    }


# ============================================================
# EXPORT UTILITIES
# ============================================================
def export_to_drive(image, description, roi, scale=10, folder='KrishiDrishti'):
    """
    Export an image to Google Drive as GeoTIFF.
    """
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        region=roi,
        scale=scale,
        maxPixels=1e13,
        fileFormat='GeoTIFF'
    )
    task.start()
    print(f"Export started: {description}")
    return task


def export_features_for_training(fused_collection, training_points, roi, scale=10):
    """
    Sample features at ground truth points for ML model training.
    
    Args:
        fused_collection: ee.ImageCollection of fused composites
        training_points: ee.FeatureCollection with 'label' property
        roi: ee.Geometry
        scale: int — sampling resolution
    
    Returns:
        ee.FeatureCollection with sampled features
    """
    # Stack all composites into a single multi-band image
    stacked = fused_collection.toBands()
    
    # Sample at training points
    training_data = stacked.sampleRegions(
        collection=training_points,
        properties=['label'],
        scale=scale,
        geometries=True
    )
    
    return training_data


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == '__main__':
    # Run the pipeline
    data = run_data_ingestion(ROI, START_DATE, END_DATE, interval_days=8)
    
    # Visualize first composite
    Map = geemap.Map(center=[17.25, 78.25], zoom=11)
    
    first_composite = ee.Image(data['fused'].first())
    
    # True color
    Map.addLayer(first_composite, {
        'bands': ['B4', 'B3', 'B2'],
        'min': 0, 'max': 0.3
    }, 'True Color')
    
    # NDVI
    Map.addLayer(first_composite, {
        'bands': ['NDVI'],
        'min': 0, 'max': 0.8,
        'palette': ['red', 'yellow', 'green', 'darkgreen']
    }, 'NDVI')
    
    # SAR VV
    Map.addLayer(first_composite, {
        'bands': ['VV'],
        'min': -20, 'max': 0
    }, 'SAR VV')
    
    Map
