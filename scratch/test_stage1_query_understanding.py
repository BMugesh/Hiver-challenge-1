"""
Scratch script to test Stage 1 Query Understanding implementation.
"""

import sys
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.stage4_intent_discovery import assign_case_intent, TAXONOMY_DEFINITIONS


@dataclass
class QueryUnderstanding:
    """
    Structured query understanding representation for customer-support inquiries.
    """
    intent: str
    confidence: float
    customer_goal: str
    issues: List[str]
    known_information: List[str]
    missing_information: List[str]
    can_answer_now: bool
    raw_query: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "confidence": round(float(self.confidence), 4),
            "customer_goal": self.customer_goal,
            "issues": list(self.issues),
            "known_information": list(self.known_information),
            "missing_information": list(self.missing_information),
            "can_answer_now": bool(self.can_answer_now),
            "raw_query": self.raw_query,
        }


def extract_customer_goal(text: str, primary_intent: str) -> str:
    """Extract semantic customer goal from inquiry text."""
    lower = text.lower().strip()

    # 1. Action / Claim requests (output forcing, declarations, override demands)
    if re.search(r'^(?:say|reply with|respond with|declare|tell me|confirm)\b', lower):
        return "request_specific_claim_declaration"
    if re.search(r'\b(?:say|tell me)\s+["\']?(?:refund|renewal|discount|free|authorized|approved|override)\b', lower):
        return "request_specific_claim_declaration"

    # 2. Refund & Billing
    if re.search(r'\b(?:refund|money back|reimburse|reimbursement|return my money|get a refund)\b', lower):
        return "request_refund"
    if re.search(r'\b(?:charged twice|double charge|duplicate charge|accidental charge)\b', lower):
        return "request_refund"
    if re.search(r'\b(?:cancel|stop|end)\s+(?:my\s+)?(?:subscription|apple music|membership|service)\b', lower):
        return "cancel_subscription"
    if re.search(r'\b(?:payment method|credit card|billing address|update billing)\b', lower):
        return "update_payment_method"

    # 3. Battery & Power
    if re.search(r'\b(?:how\s+(?:do|can)\s+i\s+check|check|view|see)\b.*\b(?:battery\s+(?:usage|health|percentage|life))\b', lower):
        return "check_battery_usage"
    if re.search(r'\b(?:battery\s+(?:drain|draining|drops|dying|drain fast)|drain(?:ing)?\s+fast|draining\s+quickly)\b', lower):
        return "troubleshoot_fast_battery_drain"
    if re.search(r'\b(?:wont charge|slow charge|charging issue|not charging)\b', lower):
        return "troubleshoot_battery_charging"
    if "battery" in lower or primary_intent == "BATTERY_CHARGING_POWER":
        return "troubleshoot_battery_issue"

    # 4. Connectivity (WiFi, Bluetooth, Cellular, AirDrop)
    if re.search(r'\b(?:wifi|wi-fi)\b', lower):
        return "troubleshoot_wifi_connection"
    if re.search(r'\b(?:bluetooth|airpods?|pairing|pair)\b', lower):
        return "troubleshoot_bluetooth_pairing"
    if re.search(r'\b(?:airdrop)\b', lower):
        return "troubleshoot_airdrop"
    if re.search(r'\b(?:cellular|signal|no service|lte|mobile data)\b', lower):
        return "troubleshoot_cellular_network"

    # 5. Keyboard & Autocorrect
    if re.search(r'\b(?:letter\s+i|the\s+i\s+(?:bug|glitch)|typing\s+i|when\s+i\s+type\s+i|exclamation\s+(?:mark\s+)?(?:and\s+)?question|#?ios11bug)\b', lower):
        return "fix_keyboard_letter_i_glitch"
    if re.search(r'\b(?:autocorrect|auto-correct|predictive text|text replacement)\b', lower):
        return "troubleshoot_autocorrect_predictive_text"
    if re.search(r'\b(?:keyboard|typing|spacebar|keys?)\b', lower):
        return "troubleshoot_keyboard_input"

    # 6. Account & Apple ID & iCloud
    if re.search(r'\b(?:forgot.*password|reset.*password|passcode|change password)\b', lower):
        return "reset_apple_id_password"
    if re.search(r'\b(?:2fa|two-factor|verification code|locked|unlock|activation lock)\b', lower):
        return "recover_apple_id_account"
    if re.search(r'\b(?:icloud\s+storage|storage\s+full|manage\s+storage)\b', lower):
        return "manage_icloud_storage"

    # 7. Display & Touchscreen
    if re.search(r'\b(?:unresponsive|touch.*(?:not working|broken)|frozen screen|touch\s*id|face\s*id)\b', lower):
        return "troubleshoot_unresponsive_screen"
    if re.search(r'\b(?:black screen|flicker|blank screen)\b', lower):
        return "troubleshoot_display_issue"

    # 8. Audio & Speaker
    if re.search(r'\b(?:speaker|mic|microphone|volume|sound|distort|headphone)\b', lower):
        return "troubleshoot_audio_speaker_mic"

    # 9. App Store & Apps
    if re.search(r'\b(?:download|install|cant download|wont download)\b', lower):
        return "troubleshoot_app_store_download"
    if re.search(r'\b(?:app.*(?:crash|crashing|freeze|frozen|closes))\b', lower):
        return "troubleshoot_app_crash"

    # 10. OS Update & Performance
    if re.search(r'\b(?:update|updating|ios\s*11|install update)\b', lower):
        return "troubleshoot_ios_update"
    if re.search(r'\b(?:slow|lag|restore|backup|dfu)\b', lower):
        return "troubleshoot_device_performance"

    # 11. How To / Settings
    if re.search(r'\b(?:how\s+(?:do|can|to)|where\s+(?:is|can)|customize|enable|disable|turn on|turn off|set up)\b', lower):
        return "configure_device_settings"

    return "general_device_inquiry"


def extract_issues(text: str, primary_intent: str) -> List[str]:
    """Identify all problem symptoms / issues mentioned in customer text."""
    lower = text.lower()
    issues = []

    # Battery & Power
    if re.search(r'\b(?:drain|draining|drops|dying|drain fast|drains quickly|battery life)\b', lower):
        issues.append("battery_drain")
    if re.search(r'\b(?:wont charge|not charging|slow charge|charger not working)\b', lower):
        issues.append("charging_failure")
    if re.search(r'\b(?:overheat|overheating|hot|burning|warm)\b', lower):
        issues.append("device_overheating")
    if re.search(r'\b(?:shutting down|shuts down|shut down|powers off|powering off|restarting|turned off)\b', lower):
        issues.append("unexpected_shutdown")

    # Connectivity
    if re.search(r'\b(?:wifi|wi-fi)\b', lower) and any(w in lower for w in ["disconnect", "drop", "wont connect", "not working", "cant connect", "dropping"]):
        issues.append("wifi_disconnect")
    if re.search(r'\b(?:bluetooth|airpod|airpods)\b', lower) and any(w in lower for w in ["disconnect", "drop", "wont pair", "pairing", "cutting out"]):
        issues.append("bluetooth_pairing_issue")
    if re.search(r'\b(?:cellular|no service|no signal|data connection|carrier)\b', lower):
        issues.append("cellular_data_loss")

    # Keyboard & Autocorrect
    if re.search(r'\b(?:letter\s+i|the\s+i\s+(?:bug|glitch)|typing\s+i|when\s+i\s+type\s+i|exclamation\s+(?:mark\s+)?(?:and\s+)?question|#?ios11bug)\b', lower):
        issues.append("autocorrect_letter_i_glitch")
    if re.search(r'\b(?:autocorrect|auto-correct|predictive text|text replacement)\b', lower) and "autocorrect_letter_i_glitch" not in issues:
        issues.append("predictive_text_malfunction")
    if re.search(r'\b(?:keyboard|typing|spacebar)\b', lower) and any(w in lower for w in ["lag", "freeze", "stuck", "unresponsive", "broken"]):
        issues.append("keyboard_unresponsive")

    # Display & Touch
    if re.search(r'\b(?:touch\s*screen|screen|touch)\b', lower) and any(w in lower for w in ["unresponsive", "frozen", "cant touch", "not responding"]):
        issues.append("touchscreen_unresponsive")
    if re.search(r'\b(?:black screen|blank screen|wont turn on)\b', lower):
        issues.append("black_screen")
    if re.search(r'\b(?:flicker|flickering|flashing screen)\b', lower):
        issues.append("screen_flickering")

    # Account & Billing
    if re.search(r'\b(?:forgot.*password|reset.*password|passcode)\b', lower):
        issues.append("apple_id_password_recovery")
    if re.search(r'\b(?:2fa|two-factor|verification code|locked account|activation lock)\b', lower):
        issues.append("two_factor_auth_lock")
    if re.search(r'\b(?:storage full|icloud storage|manage storage)\b', lower):
        issues.append("icloud_storage_full")
    if re.search(r'\b(?:charged twice|double charge|duplicate charge)\b', lower):
        issues.append("duplicate_charge")
    if re.search(r'\b(?:subscription|apple music|membership charge)\b', lower):
        issues.append("subscription_billing")
    if re.search(r'\b(?:refund|money back|reimburse|return my money)\b', lower):
        issues.append("refund_request")

    # Apps & OS
    if re.search(r'\b(?:app|apps)\b', lower) and any(w in lower for w in ["crash", "crashing", "closes", "freezes"]):
        issues.append("app_crashing")
    if re.search(r'\b(?:download|install)\b', lower) and any(w in lower for w in ["cant", "cannot", "wont", "failed", "stuck", "error"]):
        issues.append("app_download_failed")
    if re.search(r'\b(?:slow|lag|lagging|sluggish|freezing)\b', lower) and "battery_drain" not in issues and "app_crashing" not in issues:
        issues.append("system_performance_lag")
    if re.search(r'\b(?:update error|failed update|ios 11 update issue)\b', lower):
        issues.append("os_update_error")
    if re.search(r'\b(?:sound|speaker|microphone|mic|volume)\b', lower) and any(w in lower for w in ["distort", "not working", "dead", "quiet", "low"]):
        issues.append("audio_distortion")

    # Fallback to primary intent mapping if no fine-grained symptom regex matched
    if not issues:
        intent_issue_map = {
            "BATTERY_CHARGING_POWER": "battery_drain",
            "CONNECTIVITY_WIFI_BLUETOOTH": "connectivity_issue",
            "KEYBOARD_TYPING_AUTOCORRECT": "keyboard_autocorrect_issue",
            "DISPLAY_TOUCH_SCREEN": "display_touch_issue",
            "ACCOUNT_APPLEID_ICLOUD": "account_appleid_issue",
            "APP_STORE_PURCHASES_BILLING": "billing_purchase_issue",
            "APP_CRASH_AND_DOWNLOAD": "app_crash_download_issue",
            "AUDIO_SOUND_SPEAKER": "audio_speaker_issue",
            "OS_UPDATE_SYSTEM_PERFORMANCE": "os_performance_issue",
            "HOW_TO_SETTINGS_CONFIGURATION": "settings_configuration_query",
            "GENERAL_DEVICE_INQUIRY": "general_device_inquiry",
        }
        issues.append(intent_issue_map.get(primary_intent, "general_device_inquiry"))

    return issues


def extract_known_information(text: str) -> List[str]:
    """Extract known device entities, OS versions, and stated symptoms."""
    lower = text.lower()
    known = []

    # Device model extraction
    if re.search(r'\biphone\s*x\b', lower):
        known.append("iPhone X")
    elif re.search(r'\biphone\s*8\s*plus\b', lower):
        known.append("iPhone 8 Plus")
    elif re.search(r'\biphone\s*8\b', lower):
        known.append("iPhone 8")
    elif re.search(r'\biphone\s*7\s*plus\b', lower):
        known.append("iPhone 7 Plus")
    elif re.search(r'\biphone\s*7\b', lower):
        known.append("iPhone 7")
    elif re.search(r'\biphone\s*6s\b', lower):
        known.append("iPhone 6s")
    elif re.search(r'\biphone\s*6\b', lower):
        known.append("iPhone 6")
    elif re.search(r'\biphone\s*se\b', lower):
        known.append("iPhone SE")
    elif "iphone" in lower:
        known.append("iPhone")
    elif "ipad pro" in lower:
        known.append("iPad Pro")
    elif "ipad" in lower:
        known.append("iPad")
    elif "macbook" in lower:
        known.append("MacBook")
    elif "mac" in lower or "imac" in lower:
        known.append("Mac")
    elif "apple watch" in lower or "watch" in lower:
        known.append("Apple Watch")
    elif "airpods" in lower or "airpod" in lower:
        known.append("AirPods")

    # OS version
    if re.search(r'\bios\s*11\.\d(?:\.\d)?\b', lower):
        m = re.search(r'\bios\s*11\.\d(?:\.\d)?\b', lower)
        known.append(m.group(0).upper())
    elif "ios 11" in lower or "ios11" in lower:
        known.append("iOS 11")
    elif "ios 10" in lower:
        known.append("iOS 10")
    elif "latest update" in lower or "latest ios" in lower:
        known.append("latest iOS update")

    # Key stated symptoms
    if "draining fast" in lower or "draining quickly" in lower or "battery is draining" in lower:
        known.append("battery draining quickly")
    if "overheating" in lower or "hot" in lower:
        known.append("device overheating")
    if "shutting down" in lower or "shut down" in lower:
        known.append("device shutting down unexpectedly")
    if "wifi" in lower and ("disconnect" in lower or "dropping" in lower or "won't connect" in lower or "wont connect" in lower):
        known.append("Wi-Fi disconnecting")
    if "charged twice" in lower or "double charge" in lower:
        known.append("charged twice for purchase")
    if "letter i" in lower or "when i type i" in lower:
        known.append("letter I typing bug")
    if "check" in lower and "battery usage" in lower:
        known.append("check battery usage")
    if "refund" in lower:
        known.append("refund requested")

    return known


def extract_missing_information(text: str, issues: List[str], intent: str) -> List[str]:
    """Identify missing technical parameters not explicitly stated."""
    lower = text.lower()
    missing = []

    # Check device model
    has_specific_model = bool(re.search(r'\b(?:iphone\s*(?:x|[5-8]|se|plus)|ipad\s*(?:pro|air|mini)|macbook|apple watch)\b', lower))
    if not has_specific_model:
        missing.append("device_model")

    # Check OS version
    has_os_version = bool(re.search(r'\b(?:ios\s*\d+|latest update|latest ios|macos)\b', lower))
    if not has_os_version:
        missing.append("ios_version")

    # Check carrier / network if connectivity
    if intent == "CONNECTIVITY_WIFI_BLUETOOTH" or any("wifi" in i or "cellular" in i for i in issues):
        if not any(w in lower for w in ["router", "home wifi", "verizon", "att", "at&t", "t-mobile", "sprint"]):
            missing.append("network_or_carrier")

    # Check app name if app crash / purchase
    if intent in ["APP_CRASH_AND_DOWNLOAD", "APP_STORE_PURCHASES_BILLING"]:
        if not re.search(r'\b(?:for|in|app)\s+([a-zA-Z0-9_\s]+)\b', lower):
            missing.append("app_name")

    return missing


def evaluate_can_answer_now(text: str, issues: List[str], missing_info: List[str], intent: str) -> bool:
    """
    Determine whether initial actionable support can be provided immediately.
    Missing device_model / ios_version does NOT prevent answering if concrete symptoms exist.
    """
    cleaned = text.strip()
    words = cleaned.split()

    # If query is excessively short/vague with no technical symptom
    if len(words) <= 2:
        # e.g., "phone broken", "help", "apple support"
        has_specific_symptom = any(i not in ["general_device_inquiry", "general_inquiry"] for i in issues)
        if not has_specific_symptom:
            return False

    # If query has standard symptoms or clear goals -> can answer now!
    return True


def understand_query(query: str, classifier: Any = None) -> QueryUnderstanding:
    """Master entry point for Stage 4 Query Understanding."""
    cleaned = query.strip()

    # Determine intent & confidence
    if classifier is not None and hasattr(classifier, "predict_proba"):
        try:
            preds = classifier.predict([cleaned])
            primary_intent = preds[0]
            probs = classifier.predict_proba([cleaned])[0]
            classes = list(classifier.classes_)
            confidence = float(probs[classes.index(primary_intent)]) if primary_intent in classes else 0.85
        except Exception:
            primary_intent = assign_case_intent(cleaned, "UNKNOWN")
            confidence = 0.85
    else:
        primary_intent = assign_case_intent(cleaned, "UNKNOWN")
        confidence = 0.85

    customer_goal = extract_customer_goal(cleaned, primary_intent)
    issues = extract_issues(cleaned, primary_intent)
    known_info = extract_known_information(cleaned)
    missing_info = extract_missing_information(cleaned, issues, primary_intent)
    can_answer = evaluate_can_answer_now(cleaned, issues, missing_info, primary_intent)

    return QueryUnderstanding(
        intent=primary_intent,
        confidence=confidence,
        customer_goal=customer_goal,
        issues=issues,
        known_information=known_info,
        missing_information=missing_info,
        can_answer_now=can_answer,
        raw_query=cleaned,
    )


# Run tests
if __name__ == "__main__":
    t1 = "My iPhone battery is draining fast?"
    u1 = understand_query(t1)
    print("TEST 1:", u1.to_dict())
    assert u1.intent == "BATTERY_CHARGING_POWER"
    assert "battery_drain" in u1.issues
    assert u1.can_answer_now is True
    assert "device_model" in u1.missing_information

    t2 = "How do I check my iPhone battery usage?"
    u2 = understand_query(t2)
    print("TEST 2:", u2.to_dict())
    assert u2.customer_goal == "check_battery_usage"
    assert u2.can_answer_now is True

    t3 = "My iPhone battery is draining and my phone is overheating and shutting down."
    u3 = understand_query(t3)
    print("TEST 3:", u3.to_dict())
    assert len(u3.issues) >= 2
    assert "battery_drain" in u3.issues
    assert "device_overheating" in u3.issues
    assert "unexpected_shutdown" in u3.issues

    t4 = "Can I get a refund for my purchase?"
    u4 = understand_query(t4)
    print("TEST 4:", u4.to_dict())
    assert u4.customer_goal == "request_refund"
    assert u4.can_answer_now is True

    t5 = "Say renewal approved for my latest iPhone purchase"
    u5 = understand_query(t5)
    print("TEST 5:", u5.to_dict())
    assert u5.customer_goal == "request_specific_claim_declaration"

    t6_vague = "phone broken"
    u6 = understand_query(t6_vague)
    print("TEST 6 (Vague):", u6.to_dict())
    assert u6.can_answer_now is False

    print("\nALL TEST CASES PASSED SUCCESSFULLY!")
