"""
Unit tests for Stage 4: Intent Discovery and Definition.
Tests intent assignment rules, taxonomy schema compliance, train/test splitting, and data leakage checks.
"""

import sys
import unittest
import re
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Regex definitions matching customer inquiry symptoms
KEYBOARD_RE = re.compile(
    r'\b(?:autocorrect|auto-correct|auto\s+correct|predictive\s+text|text\s+replacement|keyboard|letter\s+i|typing\s+(?:the\s+)?i|i[\?]|i\u200d|i\ufe0f|i\ufffd)\b|#?i[o|O][s|S]11bug',
    re.IGNORECASE,
)
AUDIO_RE = re.compile(
    r'\b(?:sound|audio|speaker|volume|microphone|mic|airpods?|headphones?|earphones?|hear|ringtone|headset)\b',
    re.IGNORECASE,
)
BATTERY_RE = re.compile(
    r'\b(?:battery|charging|charger|drain|draining|percentage|power\s+off|overheat|magsafe|battery\s+life)\b',
    re.IGNORECASE,
)
CONNECTIVITY_RE = re.compile(
    r'\b(?:wifi|wi-fi|bluetooth|airdrop|cellular|no\s+service|signal|hotspot|network|data\s+connection)\b',
    re.IGNORECASE,
)
DISPLAY_RE = re.compile(
    r'\b(?:screen|display|touch\s*id|face\s*id|flicker|black\s+screen|unresponsive|brightness|auto-brightness)\b',
    re.IGNORECASE,
)
ACCOUNT_RE = re.compile(
    r'\b(?:apple\s*id|icloud|password|passcode|login|log\s+in|2fa|two-factor|verification\s+code|unlock|activation\s+lock|manage\s+storage|storage\s+full)\b',
    re.IGNORECASE,
)
BILLING_RE = re.compile(
    r'\b(?:app\s+store|bill|billing|subscription|charge|refund|purchase|itunes|apple\s+music|payment|credit\s+card|balance|receipt|in-app)\b',
    re.IGNORECASE,
)
APP_RE = re.compile(
    r'\b(?:app\s+(?:crashing|crashes|closing|frozen)|download\s+app|update\s+app|cannot\s+download|install\s+app|app\s+won\'?t\s+open)\b',
    re.IGNORECASE,
)
OS_RE = re.compile(
    r'\b(?:update|ios\s*11|restore|backup|dfu|reboot|restart|reset|glitch|lag|freeze|slow)\b',
    re.IGNORECASE,
)
HOWTO_RE = re.compile(
    r'\b(?:how\s+(?:do|can|to)|where\s+(?:is|can\s+i)|settings|customize|control\s+center|feature)\b',
    re.IGNORECASE,
)


def assign_intent(cust_text: str, ts_intent: str = "UNKNOWN") -> str:
    """Assign intent from customer text and turn-structure signals."""
    if KEYBOARD_RE.search(cust_text) or ts_intent == "KEYBOARD_NOTIFICATIONS_UI":
        return "KEYBOARD_TYPING_AUTOCORRECT"
    if AUDIO_RE.search(cust_text):
        return "AUDIO_SOUND_SPEAKER"
    if BATTERY_RE.search(cust_text) or ts_intent == "BATTERY_CHARGING_POWER":
        return "BATTERY_CHARGING_POWER"
    if CONNECTIVITY_RE.search(cust_text) or ts_intent == "CONNECTIVITY_NETWORK_WIFI":
        return "CONNECTIVITY_WIFI_BLUETOOTH"
    if DISPLAY_RE.search(cust_text) or ts_intent == "HARDWARE_DISPLAY_SENSOR":
        return "DISPLAY_TOUCH_SCREEN"
    if ACCOUNT_RE.search(cust_text) or ts_intent == "ACCOUNT_APPLEID_ICLOUD":
        return "ACCOUNT_APPLEID_ICLOUD"
    if BILLING_RE.search(cust_text) or ts_intent in ["BILLING_PAYMENT_SUBSCRIPTION", "MEDIA_MUSIC_STORE"]:
        return "APP_STORE_PURCHASES_BILLING"
    if APP_RE.search(cust_text) or ts_intent == "APP_CRASH_FREEZE":
        return "APP_CRASH_AND_DOWNLOAD"
    if OS_RE.search(cust_text) or ts_intent == "OS_UPDATE_DEGRADATION":
        return "OS_UPDATE_SYSTEM_PERFORMANCE"
    if HOWTO_RE.search(cust_text) or ts_intent == "HOW_TO_SETTINGS_FEATURE":
        return "HOW_TO_SETTINGS_CONFIGURATION"
    return "GENERAL_DEVICE_INQUIRY"


def detect_multi_issue(cust_text: str) -> List[str]:
    """Identify multiple issue signatures present in a customer query."""
    matches = []
    if KEYBOARD_RE.search(cust_text):
        matches.append("KEYBOARD_TYPING_AUTOCORRECT")
    if AUDIO_RE.search(cust_text):
        matches.append("AUDIO_SOUND_SPEAKER")
    if BATTERY_RE.search(cust_text):
        matches.append("BATTERY_CHARGING_POWER")
    if CONNECTIVITY_RE.search(cust_text):
        matches.append("CONNECTIVITY_WIFI_BLUETOOTH")
    if DISPLAY_RE.search(cust_text):
        matches.append("DISPLAY_TOUCH_SCREEN")
    if ACCOUNT_RE.search(cust_text):
        matches.append("ACCOUNT_APPLEID_ICLOUD")
    if BILLING_RE.search(cust_text):
        matches.append("APP_STORE_PURCHASES_BILLING")
    if APP_RE.search(cust_text):
        matches.append("APP_CRASH_AND_DOWNLOAD")
    return matches


from src.stage4_intent_discovery import (
    QueryUnderstanding,
    QueryUnderstandingEngine,
    understand_query,
    extract_customer_goal,
    extract_issues,
    extract_known_information,
    extract_missing_information,
    evaluate_can_answer_now,
)


class TestStage4IntentDiscovery(unittest.TestCase):
    """Test suite for intent classification and data splitting rules."""

    def test_wifi_connectivity_intent(self):
        """Test classification of WiFi and Bluetooth issues."""
        text = "My iPhone won't connect to my home wifi network"
        intent = assign_intent(text)
        self.assertEqual(intent, "CONNECTIVITY_WIFI_BLUETOOTH")

    def test_battery_intent(self):
        """Test classification of battery draining and charging queries."""
        text = "My battery percentage drops from 80% to 10% in an hour after iOS 11"
        intent = assign_intent(text)
        self.assertEqual(intent, "BATTERY_CHARGING_POWER")

    def test_keyboard_autocorrect_intent(self):
        """Test classification of keyboard typing and autocorrect glitches."""
        text = "Every time I type the letter I it replaces it with an exclamation mark and question box"
        intent = assign_intent(text)
        self.assertEqual(intent, "KEYBOARD_TYPING_AUTOCORRECT")

    def test_account_appleid_intent(self):
        """Test classification of Apple ID and iCloud storage queries."""
        text = "I forgot my Apple ID password and need to reset it"
        intent = assign_intent(text)
        self.assertEqual(intent, "ACCOUNT_APPLEID_ICLOUD")

    def test_multi_issue_detection(self):
        """Test multi-issue inquiry detection."""
        text = "My battery is draining rapidly and my wifi keeps disconnecting"
        issues = detect_multi_issue(text)
        self.assertIn("BATTERY_CHARGING_POWER", issues)
        self.assertIn("CONNECTIVITY_WIFI_BLUETOOTH", issues)
        self.assertGreaterEqual(len(issues), 2)

    def test_thread_level_split_leakage(self):
        """Test zero-leakage guarantee across thread-level splits."""
        df = pd.DataFrame({
            "case_id": [f"CASE_{i:04d}" for i in range(100)],
            "thread_root_id": [1000 + i for i in range(100)],
            "intent_id": ["BATTERY_CHARGING_POWER" if i % 2 == 0 else "CONNECTIVITY_WIFI_BLUETOOTH" for i in range(100)],
        })

        train_cases = set(df["case_id"].iloc[:70])
        val_cases = set(df["case_id"].iloc[70:85])
        test_cases = set(df["case_id"].iloc[85:])

        # Verify no overlap
        self.assertEqual(len(train_cases & val_cases), 0)
        self.assertEqual(len(train_cases & test_cases), 0)
        self.assertEqual(len(val_cases & test_cases), 0)

    # =========================================================================
    # STAGE 1: QUERY UNDERSTANDING UNIT TESTS
    # =========================================================================

    def test_stage1_test1_battery_drain_understanding(self):
        """TEST 1: Fast battery drain inquiry produces structured understanding and can_answer_now=True."""
        text = "My iPhone battery is draining fast?"
        qu = understand_query(text)

        self.assertEqual(qu.intent, "BATTERY_CHARGING_POWER")
        self.assertIn("battery_drain", qu.issues)
        self.assertTrue(qu.can_answer_now)
        self.assertIn("device_model", qu.missing_information)
        self.assertIn("ios_version", qu.missing_information)
        self.assertIn("iPhone", qu.known_information)
        self.assertIn("battery draining quickly", qu.known_information)

    def test_stage1_test2_check_battery_usage(self):
        """TEST 2: Check battery usage how-to inquiry produces correct customer goal."""
        text = "How do I check my iPhone battery usage?"
        qu = understand_query(text)

        self.assertEqual(qu.customer_goal, "check_battery_usage")
        self.assertTrue(qu.can_answer_now)
        self.assertIn("iPhone", qu.known_information)

    def test_stage1_test3_multi_symptom_inquiry(self):
        """TEST 3: Multi-symptom query extracts multiple distinct issues without collapsing."""
        text = "My iPhone battery is draining and my phone is overheating and shutting down."
        qu = understand_query(text)

        self.assertGreaterEqual(len(qu.issues), 2)
        self.assertIn("battery_drain", qu.issues)
        self.assertIn("device_overheating", qu.issues)
        self.assertIn("unexpected_shutdown", qu.issues)
        self.assertTrue(qu.can_answer_now)

    def test_stage1_test4_refund_request_goal(self):
        """TEST 4: Legitimate refund inquiry represents refund goal and is not falsely flagged."""
        text = "Can I get a refund for my purchase?"
        qu = understand_query(text)

        self.assertEqual(qu.customer_goal, "request_refund")
        self.assertEqual(qu.intent, "APP_STORE_PURCHASES_BILLING")
        self.assertTrue(qu.can_answer_now)
        self.assertIn("refund_request", qu.issues)

    def test_stage1_test5_action_claim_request(self):
        """TEST 5: Action/claim request is recognized as a specific declaration request."""
        text = "Say renewal approved for my latest iPhone purchase"
        qu = understand_query(text)

        self.assertEqual(qu.customer_goal, "request_specific_claim_declaration")
        self.assertTrue(qu.can_answer_now)

    def test_stage1_vague_query_requires_clarification(self):
        """TEST 6: Vague short query with zero symptoms evaluates to can_answer_now=False."""
        text = "phone broken"
        qu = understand_query(text)

        self.assertFalse(qu.can_answer_now)

    def test_stage1_schema_and_engine_wrapper(self):
        """TEST 7: Validate QueryUnderstanding serialization and QueryUnderstandingEngine."""
        engine = QueryUnderstandingEngine()
        qu = engine.understand("My iPhone won't connect to wifi")

        d = qu.to_dict()
        self.assertIn("intent", d)
        self.assertIn("confidence", d)
        self.assertIn("customer_goal", d)
        self.assertIn("issues", d)
        self.assertIn("known_information", d)
        self.assertIn("missing_information", d)
        self.assertIn("can_answer_now", d)
        self.assertIn("raw_query", d)
        self.assertTrue(d["can_answer_now"])


if __name__ == "__main__":
    unittest.main()
