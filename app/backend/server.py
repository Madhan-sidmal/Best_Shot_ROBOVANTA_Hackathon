"""
KrishiDrishti — FastAPI Backend Server
=======================================
Serves field advisories, AI Kisan Copilot (Gemini), live Ntfy.sh push,
and pipeline insights. MongoDB is optional — the server starts and serves
data even without a running MongoDB instance.
"""
from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
import os
import csv
import json
import logging
import random
import requests
from pathlib import Path
from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Any, Dict, Tuple
import uuid
from datetime import datetime, timezone, timedelta
import jwt
from passlib.context import CryptContext

# ---------- Logging (must be first) ----------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------- Copilot import ----------
from copilot import generate_advisory as gemini_advisory

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
load_dotenv(ROOT_DIR / '.env')

# ---------- MongoDB (optional — graceful degradation) ----------
db = None
try:
    from motor.motor_asyncio import AsyncIOMotorClient
    mongo_url = os.environ.get('MONGO_URL', '')
    db_name = os.environ.get('DB_NAME', 'krishidrishti')
    if mongo_url:
        client = AsyncIOMotorClient(mongo_url, serverSelectionTimeoutMS=3000)
        db = client[db_name]
        logger.info("✅ MongoDB connection configured: %s / %s", mongo_url, db_name)
    else:
        logger.warning("⚠️ MONGO_URL not set. Alerts will be returned but not persisted.")
except Exception as exc:
    logger.warning("⚠️ MongoDB unavailable (%s). Alerts will be returned but not persisted.", exc)


app = FastAPI(title="KrishiDrishti API")

@app.on_event("startup")
async def startup_db_client():
    global db
    if db is not None:
        try:
            # Ping database to see if MongoDB is actually running
            await db.command("ping")
            logger.info("✅ MongoDB connection verified and active.")
        except Exception as exc:
            logger.warning("⚠️ MongoDB connection check failed: %s. Falling back to JSON-file-based mock database for all auth and alerts (no 3s lag).", exc)
            db = None

api_router = APIRouter(prefix="/api")


# ---------- Models ----------
class FieldRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")
    field_id: str
    crop_type: str
    growth_stage: str
    csi: float  # Combined Stress Index (0-1)
    water_deficit_mm: float
    advisory_status: str  # Adequate | Watch | Urgent | Critical
    latitude: float
    longitude: float


class AlertRequest(BaseModel):
    field_id: str
    channel: str  # sms | whatsapp
    message: Optional[str] = None
    ntfy_topic: Optional[str] = None
    crop: Optional[str] = None


class AlertResponse(BaseModel):
    id: str
    field_id: str
    channel: str
    message: str
    status: str
    dispatched_at: str
    ntfy_status: Optional[str] = None
    ntfy_topic: Optional[str] = None


class CopilotRequest(BaseModel):
    plot_id: str
    crop: str
    stage: str
    deficit_mm: float
    status: str
    etc_mm: Optional[float] = None

# ---------- Auth Models & Config ----------
SECRET_KEY = "krishidrishti-super-secret-key"
ALGORITHM = "HS256"
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# fallback persistent store if MongoDB is down
MOCK_USERS_FILE = DATA_DIR / "mock_users.json"

def load_mock_users():
    if MOCK_USERS_FILE.exists():
        try:
            return json.loads(MOCK_USERS_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("⚠️ Failed to parse mock_users.json: %s", e)
            return {}
    return {}

def save_mock_users(users):
    try:
        MOCK_USERS_FILE.write_text(json.dumps(users, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("⚠️ Failed to save mock_users.json: %s", e)

MOCK_USERS = load_mock_users()

class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str

class LoginRequest(BaseModel):
    email: str
    password: str

class AuthResponse(BaseModel):
    token: str
    user: dict

# ---------- Parametric Simulator ----------
CROPS = ["Rice", "Wheat", "Cotton", "Sugarcane"]
STAGES = {
    "Rice": ["Nursery", "Tillering", "Panicle Initiation", "Flowering", "Grain Filling"],
    "Wheat": ["Germination", "Tillering", "Jointing", "Heading", "Grain Filling"],
    "Cotton": ["Emergence", "Squaring", "Flowering", "Boll Development", "Maturity"],
    "Sugarcane": ["Germination", "Tillering", "Grand Growth", "Maturation", "Ripening"],
}


def _status_from_csi(csi: float) -> str:
    if csi < 0.3:
        return "Adequate"
    if csi < 0.55:
        return "Watch"
    if csi < 0.78:
        return "Urgent"
    return "Critical"


def generate_fields(n: int = 24, seed: int = 42) -> List[FieldRecord]:
    rng = random.Random(seed)
    fields = []
    # Anchor around Indira Gandhi Canal Command (Rajasthan, India)
    base_lat, base_lng = 29.88, 75.82
    for i in range(n):
        crop = rng.choice(CROPS)
        stage = rng.choice(STAGES[crop])
        csi = round(rng.betavariate(2, 3), 3)  # skewed 0..1
        deficit = round(csi * rng.uniform(20, 95), 1)
        fields.append(FieldRecord(
            field_id=f"FLD-{1000 + i:04d}",
            crop_type=crop,
            growth_stage=stage,
            csi=csi,
            water_deficit_mm=deficit,
            advisory_status=_status_from_csi(csi),
            latitude=round(base_lat + rng.uniform(-0.4, 0.4), 4),
            longitude=round(base_lng + rng.uniform(-0.4, 0.4), 4),
        ))
    return fields


# ---------- Routes ----------
@api_router.get("/")
async def root():
    return {"message": "KrishiDrishti API", "status": "ok"}


# ---------- Auth Routes ----------
def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=7)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

@api_router.post("/auth/register", response_model=AuthResponse)
async def register(req: RegisterRequest):
    email = req.email.lower()
    hashed = get_password_hash(req.password)
    user_doc = {"email": email, "name": req.name, "password": hashed, "created_at": datetime.utcnow().isoformat()}
    
    use_mock = True
    if db is not None:
        try:
            existing = await db.users.find_one({"email": email})
            if existing:
                raise HTTPException(status_code=400, detail="Email already registered")
            res = await db.users.insert_one(user_doc)
            user_doc["_id"] = str(res.inserted_id)
            use_mock = False
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("⚠️ MongoDB write failed, falling back to in-memory: %s", exc)
            use_mock = True
            
    if use_mock:
        if email in MOCK_USERS:
            raise HTTPException(status_code=400, detail="Email already registered")
        MOCK_USERS[email] = user_doc
        save_mock_users(MOCK_USERS)

    token = create_access_token({"sub": email})
    return {"token": token, "user": {"email": email, "name": req.name}}

@api_router.post("/auth/login", response_model=AuthResponse)
async def login(req: LoginRequest):
    email = req.email.lower()
    user = None
    
    use_mock = True
    if db is not None:
        try:
            user = await db.users.find_one({"email": email})
            use_mock = False
        except Exception as exc:
            logger.warning("⚠️ MongoDB read failed, falling back to in-memory: %s", exc)
            use_mock = True

    if use_mock:
        user = MOCK_USERS.get(email)

    if not user or not verify_password(req.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    token = create_access_token({"sub": email})
    return {"token": token, "user": {"email": email, "name": user["name"]}}

@api_router.get("/auth/me")
async def get_me():
    # Mock endpoint since we decode JWT on frontend for this demo
    return {"status": "ok"}

@api_router.post("/auth/logout")
async def logout():
    return {"status": "logged_out"}


# ---------- Data Routes ----------
@api_router.get("/fields", response_model=List[FieldRecord])
async def get_fields():
    """Return field advisories from real pipeline files if present, else parametric simulator."""
    real = _read_fields_from_data()
    if real:
        return real
    return generate_fields()

@api_router.get("/advisories")
async def get_advisories():
    """Alias for /fields to satisfy specific component requests if needed."""
    return await get_fields()

@api_router.get("/stats")
async def get_stats():
    """Return aggregated stats for KPIs."""
    fields = await get_fields()
    total = len(fields)
    adequate = sum(1 for f in fields if f.advisory_status == 'Adequate')
    watch = sum(1 for f in fields if f.advisory_status == 'Watch')
    urgent = sum(1 for f in fields if f.advisory_status == 'Urgent')
    critical = sum(1 for f in fields if f.advisory_status == 'Critical')
    
    return {
        "total_fields": total,
        "adequate": adequate,
        "stressed": watch + urgent + critical,
        "critical": critical,
        "average_csi": round(sum(f.csi for f in fields) / total, 3) if total else 0
    }


def _read_fields_from_data() -> Optional[List[FieldRecord]]:
    """If predictions.csv OR advisory.geojson exists in data/, materialise fields from it."""
    predictions_path = DATA_DIR / "predictions.csv"
    advisory_path = DATA_DIR / "advisory.geojson"
    fields: List[FieldRecord] = []

    if predictions_path.exists():
        try:
            with predictions_path.open("r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    csi = float(row.get("csi", 0) or 0)
                    fields.append(FieldRecord(
                        field_id=str(row.get("field_id") or row.get("plot_id") or "FLD-?"),
                        crop_type=str(row.get("crop_type") or row.get("crop") or "Rice"),
                        growth_stage=str(row.get("growth_stage") or "Vegetative"),
                        csi=csi,
                        water_deficit_mm=float(row.get("water_deficit_mm") or 0),
                        advisory_status=str(row.get("advisory_status")
                                            or _status_from_csi(csi)),
                        latitude=float(row.get("latitude") or 29.88),
                        longitude=float(row.get("longitude") or 75.82),
                    ))
        except Exception as exc:
            logger.warning("predictions.csv parse failed: %s", exc)

    if not fields and advisory_path.exists():
        try:
            data = json.loads(advisory_path.read_text(encoding="utf-8"))
            for feat in data.get("features", []):
                p = feat.get("properties", {}) or {}
                coords = (feat.get("geometry") or {}).get("coordinates", [[]])
                # Use centroid of first ring for lat/lng
                ring = coords[0] if coords and isinstance(coords[0], list) else []
                if ring:
                    lng = sum(pt[0] for pt in ring) / len(ring)
                    lat = sum(pt[1] for pt in ring) / len(ring)
                else:
                    lat, lng = 29.88, 75.82
                csi = float(p.get("csi", 0) or 0)
                fields.append(FieldRecord(
                    field_id=str(p.get("field_id") or p.get("plot_id") or "FLD-?"),
                    crop_type=str(p.get("crop_type") or p.get("crop") or "Rice"),
                    growth_stage=str(p.get("growth_stage") or "Vegetative"),
                    csi=csi,
                    water_deficit_mm=float(p.get("water_deficit_mm") or 0),
                    advisory_status=str(p.get("advisory_status")
                                        or p.get("status")
                                        or _status_from_csi(csi)),
                    latitude=lat,
                    longitude=lng,
                ))
        except Exception as exc:
            logger.warning("advisory.geojson parse failed: %s", exc)

    return fields or None


def _push_ntfy(topic: str, title: str, body: str, tags: str = "warning,seedling,droplet") -> str:
    """Send a live push via Ntfy.sh. Returns 'sent' | 'failed:<reason>' | 'skipped'."""
    if not topic:
        return "skipped"
    try:
        r = requests.post(
            f"https://ntfy.sh/{topic}",
            data=body.encode("utf-8"),
            headers={
                "Title": title,
                "Priority": "high",
                "Tags": tags,
            },
            timeout=6,
        )
        if r.status_code // 100 == 2:
            return "sent"
        return f"failed:{r.status_code}"
    except Exception as exc:  # noqa: BLE001
        return f"failed:{type(exc).__name__}"


def _build_copilot_broadcast(result: Dict[str, Any], plot_id: str, crop: str) -> Tuple[str, str]:
    """Assemble a multilingual push body for the copilot advisory broadcast."""
    title = f"KrishiDrishti AI Copilot Alert: {plot_id} ({crop})"
    body = (
        f"[EN] {result.get('advisory_en', '')}\n\n"
        f"[HI] {result.get('advisory_hi', '')}\n\n"
        f"[TA] {result.get('advisory_ta', '')}"
    )
    return title, body


@api_router.post("/alerts/dispatch", response_model=AlertResponse)
async def dispatch_alert(req: AlertRequest):
    if req.channel not in {"sms", "whatsapp"}:
        raise HTTPException(status_code=400, detail="channel must be 'sms' or 'whatsapp'")
    default_msg = (
        f"KrishiDrishti Advisory: Field {req.field_id} requires immediate irrigation. "
        "Apply 30-40mm water within 24h. Reply STOP to opt out."
    )
    body = req.message or default_msg
    record = {
        "id": str(uuid.uuid4()),
        "field_id": req.field_id,
        "channel": req.channel,
        "message": body,
        "status": "dispatched_mock",  # SMS/WhatsApp is MOCKED
        "dispatched_at": datetime.now(timezone.utc).isoformat(),
        "ntfy_topic": req.ntfy_topic,
        "ntfy_status": None,
    }
    if req.ntfy_topic:
        title = f"KrishiDrishti Alert: {req.crop or 'Field'} ({req.field_id})"
        record["ntfy_status"] = _push_ntfy(req.ntfy_topic, title, body)

    # Persist to MongoDB if available
    if db is not None:
        try:
            await db.alerts.insert_one({**record})
        except Exception as exc:
            logger.warning("MongoDB insert failed: %s", exc)

    record.pop("_id", None)
    return AlertResponse(**record)


@api_router.post("/copilot/advisory")
async def copilot_advisory(req: CopilotRequest, bg: BackgroundTasks):
    """AI Kisan Copilot — Gemini-powered agronomy advisory with offline fallback.
    On success, broadcasts a multilingual EN/HI/TA push to ntfy.sh/krishidrishti_demo
    in the background (fire-and-forget) so live judges see the phone light up.
    """
    etc = req.etc_mm if req.etc_mm is not None else max(req.deficit_mm + 5.0, 25.0)
    payload = {
        "plot_id": req.plot_id,
        "crop": req.crop,
        "stage": req.stage,
        "deficit_mm": req.deficit_mm,
        "status": req.status,
        "etc_mm": etc,
    }
    result = gemini_advisory(payload)

    # Live-judging Ntfy broadcast — fire-and-forget, tri-lingual body.
    title, body = _build_copilot_broadcast(result, req.plot_id, req.crop)
    bg.add_task(_push_ntfy, "krishidrishti_demo", title, body, "robot,tractor,droplet")
    result["broadcast"] = {"topic": "krishidrishti_demo", "queued": True}
    return result


@api_router.get("/pipeline/geojson")
async def pipeline_geojson():
    """Serve advisory.geojson if present; else synthesise polygon features from
    the parametric simulator so the Leaflet map always renders something."""
    path = DATA_DIR / "advisory.geojson"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("advisory.geojson read failed: %s", exc)

    # Synthesise small square polygons around each simulator field.
    fields = generate_fields()
    features = []
    for f in fields:
        lat, lng = f.latitude, f.longitude
        d = 0.008  # ~800 m half-side
        ring = [
            [lng - d, lat - d],
            [lng + d, lat - d],
            [lng + d, lat + d],
            [lng - d, lat + d],
            [lng - d, lat - d],
        ]
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {
                "field_id": f.field_id,
                "crop_type": f.crop_type,
                "growth_stage": f.growth_stage,
                "csi": f.csi,
                "water_deficit_mm": f.water_deficit_mm,
                "advisory_status": f.advisory_status,
                "source": "demo",
            },
        })
    return {"type": "FeatureCollection", "features": features, "meta": {"source": "demo"}}


# ---------- Pipeline insights (optional real files → graceful mock) ----------
def _read_csv_first_row(path: Path) -> Optional[Dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                return row
    except Exception:
        return None
    return None


def _read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


@api_router.get("/pipeline/insights")
async def pipeline_insights():
    """Return 5 leaf-style insights for a representative field.
    Reads real pipeline files if present; otherwise returns clearly labeled demo data.
    """
    sources: Dict[str, str] = {}

    predictions_path = DATA_DIR / "predictions.csv"
    metrics_path = DATA_DIR / "metrics.json"
    advisory_path = DATA_DIR / "advisory.geojson"
    model_cmp_path = DATA_DIR / "model_comparison.csv"

    pred_row = _read_csv_first_row(predictions_path)
    metrics = _read_json(metrics_path)
    advisory = _read_json(advisory_path)
    model_cmp = _read_csv_first_row(model_cmp_path)

    # Fallback synthetic field from parametric simulator
    fallback = generate_fields(n=1, seed=7)[0]

    def pick(real_val, demo_val, key: str) -> tuple:
        if real_val is None or real_val == "":
            sources[key] = "demo"
            return demo_val, "demo"
        sources[key] = "backend"
        return real_val, "backend"

    crop, crop_src = pick(
        (pred_row or {}).get("crop") or (pred_row or {}).get("crop_type"),
        fallback.crop_type,
        "crop",
    )
    stage, stage_src = pick(
        (pred_row or {}).get("growth_stage"),
        fallback.growth_stage,
        "growth_stage",
    )
    csi_raw = (pred_row or {}).get("csi") or (metrics or {}).get("csi")
    try:
        csi_val = float(csi_raw) if csi_raw is not None else None
    except (TypeError, ValueError):
        csi_val = None
    csi_final, csi_src = pick(csi_val, fallback.csi, "moisture_stress")

    deficit_raw = (pred_row or {}).get("water_deficit_mm") or (
        (advisory or {}).get("features", [{}])[0]
        .get("properties", {}) .get("water_deficit_mm")
        if isinstance(advisory, dict) else None
    )
    try:
        deficit_val = float(deficit_raw) if deficit_raw is not None else None
    except (TypeError, ValueError):
        deficit_val = None
    deficit_final, deficit_src = pick(
        deficit_val, fallback.water_deficit_mm, "water_deficit"
    )

    advisory_status_val = (pred_row or {}).get("advisory_status") or (
        (advisory or {}).get("features", [{}])[0]
        .get("properties", {}).get("status")
        if isinstance(advisory, dict) else None
    )
    advisory_final, advisory_src = pick(
        advisory_status_val, fallback.advisory_status, "irrigation_advisory"
    )

    model_acc = (model_cmp or {}).get("accuracy") or (metrics or {}).get("model_accuracy") if isinstance(metrics, dict) else None

    return {
        "field_id": (pred_row or {}).get("field_id") or fallback.field_id,
        "insights": {
            "crop": {"value": crop, "source": crop_src},
            "growth_stage": {"value": stage, "source": stage_src},
            "moisture_stress": {
                "value": round(float(csi_final), 3),
                "label": _status_from_csi(float(csi_final)),
                "source": csi_src,
            },
            "water_deficit_mm": {
                "value": round(float(deficit_final), 1),
                "source": deficit_src,
            },
            "irrigation_advisory": {
                "value": advisory_final,
                "source": advisory_src,
            },
        },
        "meta": {
            "model_accuracy": model_acc,
            "all_sources_demo": all(v == "demo" for v in sources.values()),
            "sources": sources,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }


@api_router.get("/pipeline/data-status")
async def pipeline_data_status():
    """Report which pipeline data files exist in the backend/data/ directory."""
    expected = ["predictions.csv", "advisory.geojson", "metrics.json", "model_comparison.csv"]
    status = {}
    for fname in expected:
        p = DATA_DIR / fname
        status[fname] = {
            "exists": p.exists(),
            "size_bytes": p.stat().st_size if p.exists() else 0,
        }
    return {
        "data_dir": str(DATA_DIR),
        "files": status,
        "all_present": all(s["exists"] for s in status.values()),
    }


app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
async def shutdown_db_client():
    if db is not None:
        try:
            db.client.close()
        except Exception:
            pass
