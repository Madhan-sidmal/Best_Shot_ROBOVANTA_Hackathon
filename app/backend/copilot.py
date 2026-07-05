"""KrishiDrishti AI Kisan Copilot.

Wraps Google Gemini for agronomy advisory generation, with a
deterministic offline rule-based fallback so demos never break.
Trilingual output: English + Hindi (Devanagari) + Tamil.
"""
from __future__ import annotations
import os
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai  # type: ignore
    _GENAI_OK = True
except Exception:  # pragma: no cover
    _GENAI_OK = False


# ------------------ Deterministic fallback engine ---------------------------

_STAGE_ACTIONS = {
    "Germination": {
        "en": "seed rot / poor germination",
        "hi": "बीज सड़न / अंकुरण में कमी",
        "ta": "விதை அழுகல் / மோசமான முளைப்பு",
    },
    "Vegetative": {
        "en": "canopy stunting and reduced tillering",
        "hi": "पौधे की वृद्धि रुकना और कल्ले कम बनना",
        "ta": "வளர்ச்சி தடைப்பாடு மற்றும் குறைந்த கிளைத்தல்",
    },
    "Reproductive": {
        "en": "flowering abortion and grain-fill loss (25–40% yield hit)",
        "hi": "फूल गिरना और दाना कमजोर बनना (25–40% उपज नुकसान)",
        "ta": "பூக்கள் உதிர்தல் மற்றும் தானிய நிரப்புதலில் இழப்பு (25–40% விளைச்சல் இழப்பு)",
    },
    "Maturity": {
        "en": "premature senescence and reduced grain weight",
        "hi": "समय से पहले पकाव और दाने का वजन कम होना",
        "ta": "முன்கூட்டியே பழுத்தல் மற்றும் தானிய எடை குறைவு",
    },
    "Tillering": {
        "en": "reduced tillering and lower panicle count",
        "hi": "कल्ले कम बनना और बालियाँ कम होना",
        "ta": "குறைந்த கிளைத்தல் மற்றும் குறைவான கதிர்கள்",
    },
    "Flowering": {
        "en": "flower abortion and severe yield loss",
        "hi": "फूल गिरना और गंभीर उपज हानि",
        "ta": "பூக்கள் உதிர்தல் மற்றும் கடுமையான விளைச்சல் இழப்பு",
    },
    "Grain Filling": {
        "en": "poor grain filling and shrivelled kernels",
        "hi": "दाना कमजोर बनना और सिकुड़ना",
        "ta": "மோசமான தானிய நிரப்புதல் மற்றும் சுருங்கிய கருக்கள்",
    },
}

_DEFAULT_STAGE = {
    "en": "reduced photosynthesis and yield loss",
    "hi": "प्रकाश-संश्लेषण में कमी और उपज हानि",
    "ta": "ஒளிச்சேர்க்கை குறைவு மற்றும் விளைச்சல் இழப்பு",
}

_SEVERITY = {
    "critical": {
        "en": ("CRITICAL ALERT", "within 24 hours"),
        "hi": ("गंभीर चेतावनी", "24 घंटों के भीतर"),
        "ta": ("அவசர எச்சரிக்கை", "24 மணி நேரத்திற்குள்"),
    },
    "urgent": {
        "en": ("URGENT ADVISORY", "within 48 hours"),
        "hi": ("तत्काल सलाह", "48 घंटों के भीतर"),
        "ta": ("அவசர ஆலோசனை", "48 மணி நேரத்திற்குள்"),
    },
    "watch": {
        "en": ("WATCH ADVISORY", "within 5 days"),
        "hi": ("निगरानी सलाह", "5 दिनों के भीतर"),
        "ta": ("கண்காணிப்பு ஆலோசனை", "5 நாட்களுக்குள்"),
    },
    "routine": {
        "en": ("ROUTINE ADVISORY", "as scheduled"),
        "hi": ("सामान्य सलाह", "समयानुसार"),
        "ta": ("வழக்கமான ஆலோசனை", "திட்டமிடப்பட்டபடி"),
    },
}


def _severity_key(status: str) -> str:
    s = (status or "").lower()
    if "critical" in s or "🔴" in status:
        return "critical"
    if "urgent" in s or "🟠" in status:
        return "urgent"
    if "watch" in s or "🟡" in status:
        return "watch"
    return "routine"


def _fallback_advisory(payload: Dict[str, Any]) -> Dict[str, Any]:
    crop = payload.get("crop", "the crop")
    stage = payload.get("stage", "current")
    deficit = float(payload.get("deficit_mm", 0.0))
    status = payload.get("status", "Watch")
    etc = float(payload.get("etc_mm", max(deficit + 5, 25)))
    plot = payload.get("plot_id", "the field")
    dose = int(round(deficit * 1.1))

    impact = _STAGE_ACTIONS.get(stage, _DEFAULT_STAGE)
    sev_key = _severity_key(status)
    sev = _SEVERITY[sev_key]

    advisory_en = (
        f"{sev['en'][0]}: {crop} at the {stage} stage on {plot} is facing a "
        f"water deficit of {deficit:.1f} mm against ETc of {etc:.1f} mm. Untreated, "
        f"this will cause {impact['en']}. Apply approximately {dose} mm of irrigation "
        f"{sev['en'][1]} with a 10% application buffer (FAO-56 guidance)."
    )
    advisory_hi = (
        f"{sev['hi'][0]}: {plot} पर {stage} अवस्था में {crop} को ETc {etc:.1f} मिमी के "
        f"मुकाबले {deficit:.1f} मिमी पानी की कमी है। अनदेखा करने पर {impact['hi']} हो सकता है। "
        f"{sev['hi'][1]} लगभग {dose} मिमी सिंचाई दें (FAO-56 दिशा-निर्देश, 10% अतिरिक्त बफर सहित)।"
    )
    advisory_ta = (
        f"{sev['ta'][0]}: {plot} இல் {stage} நிலையில் உள்ள {crop}, ETc {etc:.1f} மி.மீ. "
        f"க்கு எதிராக {deficit:.1f} மி.மீ. நீர் பற்றாக்குறையை எதிர்கொள்கிறது. "
        f"கவனிக்காமல் விட்டால் {impact['ta']} ஏற்படும். {sev['ta'][1]} தோராயமாக {dose} மி.மீ. "
        f"நீர்ப்பாசனம் அளிக்கவும் (FAO-56 வழிகாட்டல், 10% கூடுதல் இடையாற்றுடன்)."
    )

    bullets_en: List[str] = [
        f"Apply at least {dose} mm of irrigation water {sev['en'][1]}.",
        "Prefer early-morning or evening application to reduce evaporation losses.",
        "Coordinate with the local Water User Association (WUA) for canal roster priority.",
    ]
    bullets_hi: List[str] = [
        f"{sev['hi'][1]} कम से कम {dose} मिमी सिंचाई जल दें।",
        "वाष्पीकरण कम करने के लिए सुबह जल्दी या शाम के समय सिंचाई करें।",
        "नहर से पानी की प्राथमिकता हेतु स्थानीय जल उपभोक्ता संघ (WUA) से समन्वय करें।",
    ]
    bullets_ta: List[str] = [
        f"{sev['ta'][1]} குறைந்தபட்சம் {dose} மி.மீ. நீர் பாய்ச்சவும்.",
        "ஆவியாதலைக் குறைக்க அதிகாலை அல்லது மாலை நேரத்தில் நீர்ப்பாசனம் செய்யவும்.",
        "கால்வாய் நீர் முன்னுரிமைக்கு உள்ளூர் நீர்ப் பயனாளர் சங்கத்துடன் (WUA) ஒருங்கிணைக்கவும்.",
    ]

    if stage in {"Reproductive", "Flowering", "Grain Filling"}:
        bullets_en.append("Foliar spray of 1% Potassium Nitrate (KNO₃) to reduce stress damage.")
        bullets_hi.append("तनाव से बचाव हेतु 1% पोटैशियम नाइट्रेट (KNO₃) का पर्ण छिड़काव करें।")
        bullets_ta.append("சேதத்தைக் குறைக்க 1% பொட்டாசியம் நைட்ரேட் (KNO₃) இலைத்தெளிப்பு செய்யவும்.")

    action_plan: List[str] = [
        f"Confirm soil moisture at {plot} using tensiometer or thumb-test.",
        f"Schedule {dose} mm irrigation via drip / border-strip {sev['en'][1]}.",
        "Prioritise this plot in WUA canal-water roster.",
        "Log irrigation event in KrishiDrishti dashboard for FAO-56 water-balance update.",
        "Re-monitor field within 3–5 days via next 8-day satellite composite.",
    ]

    return {
        "status": "success",
        "source": "Offline rule-based advisory (FAO-56 heuristics)",
        "advisory_en": advisory_en,
        "advisory_hi": advisory_hi,
        "advisory_ta": advisory_ta,
        "bullet_points_en": bullets_en,
        "bullet_points_hi": bullets_hi,
        "bullet_points_ta": bullets_ta,
        "action_plan": action_plan,
    }


# ------------------ Gemini path ---------------------------------------------

_PROMPT_TMPL = """You are KrishiDrishti's AI Kisan Copilot, an Indian agronomy assistant. Return ONLY valid JSON matching this schema and no other prose.

Schema:
{{
  "advisory_en": "<one paragraph, English, ~60 words>",
  "advisory_hi": "<same paragraph in natural Hindi (Devanagari)>",
  "advisory_ta": "<same paragraph in natural Tamil (Tamil script)>",
  "bullet_points_en": ["<3–4 short actionable bullets, English>"],
  "bullet_points_hi": ["<same bullets in Hindi>"],
  "bullet_points_ta": ["<same bullets in Tamil>"],
  "action_plan": ["<5 numbered concrete 48-hour steps for the farmer/WUA operator, English>"]
}}

Field context:
plot_id: {plot_id}
crop: {crop}
growth_stage: {stage}
water_deficit_mm: {deficit_mm}
etc_mm: {etc_mm}
advisory_status: {status}

Recommended irrigation depth (FAO-56, deficit × 1.1): {dose} mm

Return ONLY the JSON object. Do NOT wrap in code fences.
"""


def _try_gemini(payload: Dict[str, Any]) -> Dict[str, Any] | None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not (_GENAI_OK and api_key):
        return None
    dose = int(round(float(payload.get("deficit_mm", 0)) * 1.1))
    prompt = _PROMPT_TMPL.format(dose=dose, **{
        "plot_id": payload.get("plot_id", ""),
        "crop": payload.get("crop", ""),
        "stage": payload.get("stage", ""),
        "deficit_mm": payload.get("deficit_mm", 0),
        "etc_mm": payload.get("etc_mm", 0),
        "status": payload.get("status", ""),
    })
    for model_name in ("gemini-2.5-flash", "gemini-1.5-flash"):
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(prompt, generation_config={
                "temperature": 0.4,
                "response_mime_type": "application/json",
            })
            text = getattr(resp, "text", None) or ""
            data = json.loads(text)
            for key in ("advisory_en", "advisory_hi", "advisory_ta",
                        "bullet_points_en", "bullet_points_hi",
                        "bullet_points_ta", "action_plan"):
                if key not in data:
                    raise ValueError(f"missing key: {key}")
            return {
                "status": "success",
                "source": f"Google Gemini Free API ({model_name})",
                **data,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gemini model %s failed: %s", model_name, exc)
            continue
    return None


def generate_advisory(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Try Gemini; on any failure, return the rule-based advisory."""
    try:
        result = _try_gemini(payload)
        if result:
            return result
    except Exception as exc:  # noqa: BLE001
        logger.exception("Gemini path errored: %s", exc)
    return _fallback_advisory(payload)
