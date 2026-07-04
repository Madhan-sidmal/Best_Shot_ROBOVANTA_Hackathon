"""
KrishiDrishti — AI Kisan Copilot (Gemini Free API Integration)
==============================================================
Provides Generative AI agronomy advisories using Google Gemini Free Tier
(gemini-1.5-flash / gemini-2.5-flash). 

Features:
- Live Gemini Free API integration via google-generativeai or REST API
- Safe offline fallback advisory generator (100% demo-safe)
- English + Hindi (Regional) advisory generation
- Zero unhandled exceptions (always returns structured JSON/dict)

DISCLAIMER: All agricultural data and recommendations are for hackathon demo purposes.
"""

import os
import json
import logging
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GeminiCopilot")

# Try importing Google Generative AI SDK
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logger.warning("google-generativeai package not installed. Gemini Copilot will run in offline fallback mode.")


class GeminiKisanCopilot:
    """
    AI Agronomist Copilot powered by Google Gemini Free API.
    Translates raw ML metrics (crop type, growth stage, water deficit mm)
    into actionable, localized farming advice in English and Hindi.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model_name = model_name
        self.is_configured = False
        
        if GENAI_AVAILABLE and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
                self.is_configured = True
                logger.info(f"✅ Gemini Copilot configured successfully with model: {self.model_name}")
            except Exception as e:
                logger.error(f"❌ Failed to configure Gemini API: {str(e)}. Falling back to offline mode.")
        else:
            logger.info("ℹ️ Gemini API key not found or SDK missing. Using offline rule-based agronomy fallback.")

    def generate_advisory(self, plot_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an agronomy advisory for a specific field/plot.
        
        Args:
            plot_data: Dict containing crop_type, stage, deficit_mm, status/advisory, etc.
            
        Returns:
            Dict containing status, source ('gemini' or 'offline_fallback'), advisory_en, advisory_hi, bullet_points
        """
        crop = plot_data.get("crop", plot_data.get("crop_display", "Crop"))
        stage = plot_data.get("stage", "Vegetative")
        deficit = float(plot_data.get("deficit_mm", plot_data.get("Deficit (mm)", 0.0)))
        status = plot_data.get("status", plot_data.get("Status", "Adequate"))
        eto = float(plot_data.get("eto_sum", plot_data.get("ETc (mm/8d)", 35.0)))
        
        # If API is configured, attempt live Gemini generation
        if self.is_configured:
            try:
                return self._generate_live_gemini(crop, stage, deficit, status, eto)
            except Exception as e:
                logger.warning(f"⚠️ Live Gemini API call failed: {str(e)}. Switching to offline fallback.")
        
        # Safe Offline Fallback (100% reliable for hackathon judging)
        return self._generate_offline_fallback(crop, stage, deficit, status, eto)

    def _generate_live_gemini(self, crop: str, stage: str, deficit: float, status: str, eto: float) -> Dict[str, Any]:
        """Call Google Gemini Free API to generate localized agronomy advice."""
        prompt = f"""
You are an expert Indian Agronomist (Kisan Copilot) assisting farmers in canal command areas.
Analyze the following satellite-derived crop and water balance data:
- Crop Type: {crop}
- Growth Stage: {stage}
- 8-Day Water Deficit: {deficit:.1f} mm (Reference Crop Evapotranspiration ETc: {eto:.1f} mm)
- Water Stress Status: {status}

Generate an actionable, high-urgency irrigation and nutrient advisory.
Provide your response in valid JSON format ONLY with the following exact structure:
{{
  "advisory_en": "A 2-sentence empathetic summary in English explaining why this water deficit matters at the {stage} stage and what to do.",
  "advisory_hi": "The exact same summary translated into natural, professional Hindi (Devanagari script) suitable for Indian farmers.",
  "bullet_points_en": [
    "Actionable bullet 1 (e.g. exact irrigation amount recommended)",
    "Actionable bullet 2 (e.g. fertilizer/potassium recommendation to withstand stress)",
    "Actionable bullet 3 (e.g. next monitoring step or canal schedule suggestion)"
  ],
  "bullet_points_hi": [
    "Actionable bullet 1 in Hindi",
    "Actionable bullet 2 in Hindi",
    "Actionable bullet 3 in Hindi"
  ]
}}
Do NOT include any markdown code blocks, backticks, or extra text outside the JSON object.
"""
        response = self.model.generate_content(prompt)
        text = response.text.strip()
        
        # Clean up code blocks if model added them
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        data = json.loads(text)
        
        return {
            "status": "success",
            "source": f"Google Gemini Free API ({self.model_name})",
            "advisory_en": data.get("advisory_en", ""),
            "advisory_hi": data.get("advisory_hi", ""),
            "bullet_points_en": data.get("bullet_points_en", []),
            "bullet_points_hi": data.get("bullet_points_hi", []),
            "raw_metrics": {"crop": crop, "stage": stage, "deficit_mm": deficit}
        }

    def _generate_offline_fallback(self, crop: str, stage: str, deficit: float, status: str, eto: float) -> Dict[str, Any]:
        """Rule-based agronomy engine providing guaranteed offline fallback advisories."""
        # Determine severity
        is_stress = deficit > 15.0 or "Critical" in str(status) or "Stress" in str(status) or "Urgent" in str(status)
        is_watch = 5.0 < deficit <= 15.0 or "Watch" in str(status) or "Soon" in str(status)
        
        if is_stress:
            adv_en = (f"CRITICAL ALERT: {crop} at {stage} stage is facing severe water deficit of {deficit:.1f} mm. "
                      f"Immediate irrigation is required to prevent canopy wilting and up to 25-30% yield loss.")
            adv_hi = (f"चेतावनी: {stage} अवस्था में {crop} को {deficit:.1f} मिमी पानी की भारी कमी का सामना करना पड़ रहा है। "
                      f"फसल सूखने और 25-30% उपज के नुकसान से बचने के लिए तुरंत सिंचाई करें।")
            bullets_en = [
                f"Apply at least {max(25.0, deficit * 1.2):.0f} mm of irrigation water immediately within the next 48 hours.",
                f"Foliar spray of 1% Potassium Nitrate (KNO3) is recommended to improve drought tolerance and stomatal regulation.",
                f"Coordinate with local water user association (WUA) for canal roster release priority."
            ]
            bullets_hi = [
                f"अगले 48 घंटों के भीतर तुरंत कम से कम {max(25.0, deficit * 1.2):.0f} मिमी सिंचाई जल दें।",
                f"सूखा सहनशीलता और पत्तियों के बचाव के लिए 1% पोटेशियम नाइट्रेट (KNO3) का छिड़काव करें।",
                f"नहर से पानी की प्राथमिकता के लिए स्थानीय जल उपभोक्ता संघ (WUA) से संपर्क करें।"
            ]
        elif is_watch:
            adv_en = (f"WATCH ADVISORY: {crop} at {stage} stage has a mild deficit of {deficit:.1f} mm. "
                      f"Soil moisture is depleting; schedule irrigation during the next canal turn.")
            adv_hi = (f"सलाह: {stage} अवस्था में {crop} में {deficit:.1f} मिमी पानी की सामान्य कमी है। "
                      f"मिट्टी की नमी घट रही है; अगली नहर बारी में सिंचाई का कार्यक्रम बनाएं।")
            bullets_en = [
                f"Plan irrigation of {max(15.0, deficit):.0f} mm within the next 5 to 7 days.",
                f"Monitor soil moisture at root zone (15-30 cm depth) avoiding unnecessary flooding.",
                f"Maintain weed-free field bunds to conserve residual soil moisture."
            ]
            bullets_hi = [
                f"अगले 5 से 7 दिनों के भीतर {max(15.0, deficit):.0f} मिमी सिंचाई की योजना बनाएं।",
                f"अनावश्यक जलभराव से बचते हुए जड़ क्षेत्र (15-30 सेमी गहराई) में मिट्टी की नमी की निगरानी करें।",
                f"मिट्टी की नमी बनाए रखने के लिए खेत की मेड़ों को खरपतवार मुक्त रखें।"
            ]
        else:
            adv_en = (f"STATUS ADEQUATE: {crop} at {stage} stage has sufficient moisture (Deficit: {deficit:.1f} mm). "
                      f"No immediate irrigation is required. Crop evapotranspiration is well balanced.")
            adv_hi = (f"स्थिति संतोषजनक: {stage} अवस्था में {crop} के पास पर्याप्त नमी है (कमी: {deficit:.1f} मिमी)। "
                      f"तत्काल सिंचाई की आवश्यकता नहीं है। फसल का वाष्पोत्सर्जन संतुलित है।")
            bullets_en = [
                "No irrigation needed for the current 8-day period; conserve canal quota.",
                f"Continue routine pest and disease scouting appropriate for the {stage} stage.",
                "Verify drainage outlets to prevent accidental waterlogging if rain occurs."
            ]
            bullets_hi = [
                "वर्तमान 8-दिवसीय अवधि के लिए किसी सिंचाई की आवश्यकता नहीं है; नहर का पानी बचाएं।",
                f"{stage} अवस्था के अनुकूल नियमित कीट और रोग की निगरानी जारी रखें।",
                "बारिश होने पर जलभराव से बचने के लिए जल निकासी की जांच करें।"
            ]
            
        return {
            "status": "success",
            "source": "Offline Agronomy Rules Engine (100% Demo-Safe Fallback)",
            "advisory_en": adv_en,
            "advisory_hi": adv_hi,
            "bullet_points_en": bullets_en,
            "bullet_points_hi": bullets_hi,
            "raw_metrics": {"crop": crop, "stage": stage, "deficit_mm": deficit}
        }


# ============================================================
# CLI TEST / DEMO
# ============================================================
if __name__ == "__main__":
    print("='*60")
    print("  Testing KrishiDrishti Gemini Kisan Copilot")
    print("='*60")
    
    copilot = GeminiKisanCopilot()
    sample_plot = {
        "crop": "Wheat",
        "stage": "Flowering",
        "deficit_mm": 24.5,
        "status": "🔴 Critical",
        "eto_sum": 42.0
    }
    
    result = copilot.generate_advisory(sample_plot)
    print(f"\nSource: {result['source']}")
    print(f"\n[English Advisory]:\n{result['advisory_en']}")
    print(f"\n[Hindi Advisory]:\n{result['advisory_hi']}")
    print(f"\n[Actionable Bullets - EN]:")
    for b in result['bullet_points_en']:
        print(f"  • {b}")
    print(f"\n[Actionable Bullets - HI]:")
    for b in result['bullet_points_hi']:
        print(f"  • {b}")
    print("='*60")
