"""
KrishiDrishti — Multi-Channel Alert & Notification Service
============================================================
Dispatches irrigation advisories and stress alerts across multiple channels:
1. Live Push Notifications via Ntfy.sh (100% open-source, zero cost, demo-ready)
2. Universal Notification Routing via Apprise (open-source alerting library)
3. Mocked SMS & WhatsApp alerts (logged cleanly when external API keys are absent)
4. Persistent Alert Logging (all dispatches saved to outputs/alert_history.json)

DISCLAIMER: All alerts are generated from synthetic crop simulation data.
"""

import os
import json
import logging
import datetime
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AlertService")

# Try importing Apprise for multi-channel open-source routing
try:
    import apprise
    APPRISE_AVAILABLE = True
except ImportError:
    APPRISE_AVAILABLE = False
    logger.warning("Apprise library not installed. Using direct Ntfy.sh and mocked SMS/WhatsApp logging.")

# Try importing Requests
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class KrishiAlertManager:
    """
    Manages multi-channel dispatch of crop water stress and irrigation advisories.
    Designed for zero-cost, real-time interactive hackathon demonstrations.
    """
    
    def __init__(self, ntfy_topic: str = "krishidrishti_demo", log_file: str = "outputs/alert_history.json"):
        self.ntfy_topic = ntfy_topic
        self.log_file = log_file
        self.history = self._load_history()
        
        self.apprise_obj = None
        if APPRISE_AVAILABLE:
            self.apprise_obj = apprise.Apprise()
            # Check for custom apprise URLs in environment
            apprise_urls = os.environ.get("APPRISE_URLS", "").split(",")
            for url in apprise_urls:
                if url.strip():
                    self.apprise_obj.add(url.strip())
                    
    def _load_history(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load alert history from {self.log_file}: {str(e)}")
        return []

    def _save_history(self):
        try:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save alert history: {str(e)}")

    def send_ntfy_push(self, title: str, message: str, priority: str = "high", tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Send a real-time push notification to phones or web browsers via Ntfy.sh.
        No API keys or signups required! Anyone subscribed to ntfy.sh/{topic} receives it instantly.
        """
        url = f"https://ntfy.sh/{self.ntfy_topic}"
        headers = {
            "Title": title.encode('utf-8').decode('latin-1', 'ignore'),
            "Priority": priority,
            "Tags": ",".join(tags or ["satellite", "tractor", "warning"])
        }
        
        status = "failed"
        detail = ""
        
        try:
            if REQUESTS_AVAILABLE:
                resp = requests.post(url, data=message.encode("utf-8"), headers=headers, timeout=5)
                if resp.status_code == 200:
                    status = "success"
                    detail = f"Delivered to ntfy.sh/{self.ntfy_topic}"
                else:
                    detail = f"HTTP {resp.status_code}: {resp.text}"
            else:
                req = urllib.request.Request(url, data=message.encode("utf-8"), headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        status = "success"
                        detail = f"Delivered via urllib to ntfy.sh/{self.ntfy_topic}"
        except Exception as e:
            detail = f"Offline/Network exception: {str(e)}"
            logger.warning(f"⚠️ Ntfy push delivery failed ({detail}). Alert recorded in offline logs.")
            
        return {"channel": "Ntfy.sh Push Notification", "status": status, "detail": detail, "topic": self.ntfy_topic}

    def send_apprise_alert(self, title: str, message: str) -> Dict[str, Any]:
        """Send notification via Apprise to all configured multi-channel endpoints."""
        if not APPRISE_AVAILABLE or not self.apprise_obj or len(self.apprise_obj) == 0:
            return {"channel": "Apprise Router", "status": "skipped", "detail": "No Apprise URLs configured in environment."}
            
        try:
            result = self.apprise_obj.notify(body=message, title=title)
            status = "success" if result else "failed"
            return {"channel": "Apprise Router", "status": status, "detail": f"Dispatched to {len(self.apprise_obj)} endpoints."}
        except Exception as e:
            return {"channel": "Apprise Router", "status": "failed", "detail": str(e)}

    def send_mock_sms_whatsapp(self, recipient: str, title: str, message: str, channel_type: str = "SMS") -> Dict[str, Any]:
        """
        Mock SMS (e.g. Twilio/MSG91) and WhatsApp alert dispatch.
        Logs the exact formatted payload cleanly when external API keys are absent.
        """
        timestamp = datetime.datetime.now().isoformat()
        logger.info(f"📱 [MOCK {channel_type} DISPATCH] To: {recipient} | Title: {title} | Body: {message[:60]}...")
        return {
            "channel": f"Mocked {channel_type} Gateway",
            "status": "mock_delivered",
            "recipient": recipient,
            "timestamp": timestamp,
            "detail": f"Simulated delivery to {recipient} (No external gateway fee incurred)."
        }

    def dispatch_advisory_alert(self, plot_data: Dict[str, Any], recipient_phone: str = "+91-9876543210") -> Dict[str, Any]:
        """
        Main orchestration entry point: takes a plot advisory, formats the alert,
        dispatches across all active channels, and logs to history.
        """
        plot_id = plot_data.get("plot_id", "F-001")
        crop = plot_data.get("crop", plot_data.get("crop_display", "Crop"))
        stage = plot_data.get("stage", "Vegetative")
        deficit = float(plot_data.get("deficit_mm", plot_data.get("Deficit (mm)", 0.0)))
        status = plot_data.get("status", plot_data.get("Status", "Adequate"))
        
        # Build Title & Message
        if deficit > 15.0 or "Critical" in str(status) or "Stress" in str(status):
            title = f"🔴 KrishiDrishti URGENT: Water Stress in {crop} ({plot_id})"
            message = (f"Field {plot_id} ({crop} - {stage} stage) is under severe water stress with an 8-day deficit of {deficit:.1f} mm. "
                       f"Immediate irrigation is required within 48 hours to prevent canopy damage and yield loss.")
            priority = "high"
            tags = ["rotating_light", "droplet", "warning"]
        elif deficit > 5.0 or "Watch" in str(status) or "Soon" in str(status):
            title = f"🟡 KrishiDrishti WATCH: Irrigation Due for {crop} ({plot_id})"
            message = (f"Field {plot_id} ({crop} - {stage} stage) shows a moderate water deficit of {deficit:.1f} mm. "
                       f"Please schedule irrigation during the upcoming canal water release.")
            priority = "default"
            tags = ["clock3", "tractor"]
        else:
            title = f"🟢 KrishiDrishti STATUS: Adequate Moisture ({plot_id})"
            message = f"Field {plot_id} ({crop} - {stage} stage) has adequate soil moisture (Deficit: {deficit:.1f} mm). No irrigation needed."
            priority = "low"
            tags = ["white_check_mark", "seedling"]
            
        # Dispatch to channels
        results = []
        results.append(self.send_ntfy_push(title, message, priority=priority, tags=tags))
        results.append(self.send_apprise_alert(title, message))
        results.append(self.send_mock_sms_whatsapp(recipient_phone, title, message, channel_type="SMS"))
        results.append(self.send_mock_sms_whatsapp(recipient_phone, title, message, channel_type="WhatsApp"))
        
        # Record history
        record = {
            "alert_id": f"ALT-{len(self.history)+1:04d}",
            "timestamp": datetime.datetime.now().isoformat(),
            "plot_id": plot_id,
            "crop": crop,
            "stage": stage,
            "deficit_mm": deficit,
            "status_label": status,
            "title": title,
            "message": message,
            "dispatch_results": results
        }
        self.history.append(record)
        self._save_history()
        
        return record

    def batch_dispatch_critical_alerts(self, advisory_df, max_alerts: int = 5) -> List[Dict[str, Any]]:
        """
        Scan an advisory dataframe and automatically dispatch alerts for all critical/urgent fields.
        Returns a summary list of dispatched alert records.
        """
        dispatched = []
        count = 0
        for idx, row in advisory_df.iterrows():
            if count >= max_alerts:
                break
            status = str(row.get("status", row.get("Status", "")))
            deficit = float(row.get("deficit_mm", row.get("Deficit (mm)", 0.0)))
            
            if deficit > 15.0 or "Critical" in status or "Stress" in status or "Urgent" in status:
                record = self.dispatch_advisory_alert(row.to_dict())
                dispatched.append(record)
                count += 1
                
        logger.info(f"✅ Batch dispatch completed: {len(dispatched)} critical alerts dispatched.")
        return dispatched


# ============================================================
# CLI TEST / DEMO
# ============================================================
if __name__ == "__main__":
    print("='*60")
    print("  Testing KrishiDrishti AlertService (Multi-Channel Stack)")
    print("='*60")
    
    mgr = KrishiAlertManager(ntfy_topic="krishidrishti_test_demo")
    sample_plot = {
        "plot_id": "F-104",
        "crop": "Cotton",
        "stage": "Flowering",
        "deficit_mm": 28.4,
        "status": "🔴 Critical"
    }
    
    record = mgr.dispatch_advisory_alert(sample_plot, recipient_phone="+91-9811223344")
    print(f"\n✅ Alert Dispatched: {record['alert_id']} | Timestamp: {record['timestamp']}")
    print(f"   Title: {record['title']}")
    print(f"\n   Channel Delivery Statuses:")
    for res in record['dispatch_results']:
        print(f"     • [{res['channel']}]: {res['status']} ({res['detail']})")
        
    print(f"\n✅ Total alerts stored in history: {len(mgr.history)}")
    print("='*60")
