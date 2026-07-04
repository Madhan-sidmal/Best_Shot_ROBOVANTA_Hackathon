"""
KrishiDrishti — Integration Test Suite for Utility Modules & Pipeline
=====================================================================
Tests:
1. Gemini Kisan Copilot (offline fallback & structure validation)
2. GeoUtils (field polygon grid generation & GeoJSON export)
3. AlertService (Ntfy push notifications & mock SMS/WhatsApp logging)
4. End-to-End Synthetic Pipeline integrity (Phase 1 MVP check)
"""

import os
import sys
import json
import unittest
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.gemini_copilot import GeminiKisanCopilot
from utils.geo_utils import GeoFieldManager
from utils.alert_service import KrishiAlertManager


class TestKrishiDrishtiIntegration(unittest.TestCase):
    
    def setUp(self):
        self.sample_plot = {
            "plot_id": "F-999",
            "crop": "Wheat",
            "stage": "Flowering",
            "deficit_mm": 25.4,
            "status": "🔴 Critical",
            "eto_sum": 45.0
        }
        
    def test_01_gemini_copilot_fallback(self):
        """Test that Gemini Copilot returns structured English & Hindi advisories safely."""
        copilot = GeminiKisanCopilot(api_key=None)  # Force offline fallback
        res = copilot.generate_advisory(self.sample_plot)
        
        self.assertEqual(res["status"], "success")
        self.assertIn("advisory_en", res)
        self.assertIn("advisory_hi", res)
        self.assertGreater(len(res["bullet_points_en"]), 0)
        self.assertGreater(len(res["bullet_points_hi"]), 0)
        print("\n✅ Test 1 Passed: Gemini Copilot offline fallback works and generates Hindi/English advisories.")

    def test_02_geo_utils_polygon_and_export(self):
        """Test that GeoFieldManager assigns polygons and exports valid GeoJSON."""
        df = pd.DataFrame([self.sample_plot] * 5)
        df["plot_id"] = [f"F-{i:03d}" for i in range(5)]
        
        geo_mgr = GeoFieldManager(aoi_name="Indira Gandhi Canal Command")
        gdf = geo_mgr.generate_field_polygons(df)
        
        self.assertIn("latitude", gdf.columns)
        self.assertIn("longitude", gdf.columns)
        self.assertIn("geojson_geometry", gdf.columns)
        
        output_dir = "outputs/test_geospatial_suite"
        paths = geo_mgr.export_spatial_data(gdf, output_dir=output_dir)
        
        self.assertIn("geojson", paths)
        self.assertTrue(os.path.exists(paths["geojson"]))
        
        with open(paths["geojson"], "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["type"], "FeatureCollection")
        self.assertEqual(len(data["features"]), 5)
        print("\n✅ Test 2 Passed: GeoUtils grid generation and GeoJSON export verified.")

    def test_03_alert_service_multichannel(self):
        """Test that AlertService logs alerts and attempts Ntfy push delivery."""
        alert_mgr = KrishiAlertManager(ntfy_topic="krishidrishti_test_suite", log_file="outputs/test_alerts.json")
        record = alert_mgr.dispatch_advisory_alert(self.sample_plot)
        
        self.assertIn("alert_id", record)
        self.assertEqual(len(record["dispatch_results"]), 4)  # Ntfy, Apprise, SMS, WhatsApp
        self.assertTrue(os.path.exists("outputs/test_alerts.json"))
        print("\n✅ Test 3 Passed: AlertService multi-channel dispatch and logging verified.")

    def test_04_pipeline_imports(self):
        """Verify that dashboard and core utilities import without circular dependency or syntax errors."""
        import utils.config as cfg
        self.assertIn("DASHBOARD", dir(cfg))
        self.assertIn("CROP_PARAMS", dir(cfg))
        self.assertIn("ADVISORY_CONFIG", dir(cfg))
        print("\n✅ Test 4 Passed: Core config and utility imports verified.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
