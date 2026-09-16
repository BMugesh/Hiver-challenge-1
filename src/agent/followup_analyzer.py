"""
Follow-up understanding and conversational intent analyzer for SupportDNA.
Detects:
  - NEW_ISSUE
  - FOLLOW_UP
  - UNRESOLVED_FOLLOW_UP
  - CONFIRMATION
  - ADDITIONAL_INFORMATION
  - CORRECTION
  - ESCALATION_REQUEST

Preserves original business intent across conversational follow-ups
unless the customer introduces a genuinely new issue.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

ESCALATION_PHRASES = [
    "contact apple support",
    "contact support",
    "talk to a human",
    "talk to human",
    "speak to a human",
    "speak with a representative",
    "speak with representative",
    "human agent",
    "agent please",
    "escalate this",
    "escalate to support",
    "transfer me",
    "speak to an agent",
    "contact the support team",
    "contact apple support team",
    "speak to support"
]

CONFIRMATION_PHRASES = [
    "that worked",
    "it worked",
    "that fixed it",
    "fixed it",
    "problem solved",
    "issue is resolved",
    "resolved now",
    "working now",
    "it is working now",
    "thank you it worked",
    "that solved it",
    "working fine now",
    "all good now"
]

CORRECTION_PHRASES = [
    "not what i meant",
    "not what i asked",
    "that is not what i meant",
    "no, that's not what i meant",
    "i didn't ask about",
    "that is wrong",
    "you misunderstood",
    "that's incorrect"
]

UNRESOLVED_PHRASES = [
    "still not resolved",
    "still not working",
    "still having the issue",
    "still having the same issue",
    "still having this problem",
    "still draining",
    "still happening",
    "didn't help",
    "did not help",
    "didn't work",
    "did not work",
    "still persists",
    "same problem",
    "problem remains",
    "no luck",
    "issue remains",
    "still the same"
]

ADDITIONAL_INFO_PHRASES = [
    "i already",
    "already tried",
    "already restarted",
    "already checked",
    "i have restarted",
    "i tried restarting",
    "i did that",
    "already done",
    "have already done",
    "running ios",
    "battery health is",
    "it drains mostly"
]

# Intent domain keywords to detect when customer transitions to a NEW issue
DOMAIN_KEYWORDS = {
    "CONNECTIVITY_WIFI_BLUETOOTH_CELLULAR": ["wifi", "wi-fi", "bluetooth", "cellular", "data", "airdrop", "hotspot"],
    "APP_STORE_PURCHASES_BILLING": ["refund", "billing", "charged", "purchase", "subscription", "receipt", "payment"],
    "ACCOUNT_APPLE_ID_SECURITY": ["apple id", "password", "locked", "icloud login", "two-factor", "2fa", "recovery"],
    "BATTERY_CHARGING_POWER": ["battery", "draining", "charge", "charging", "overheating", "warm", "power off", "shutdown"],
    "BACKUP_SYNC_RESTORE_DATA_LOSS": ["backup", "restore", "icloud backup", "lost photos", "lost contacts", "syncing"],
    "AUDIO_SOUND_MICROPHONE": ["sound", "speaker", "microphone", "mic", "volume", "audio", "earpiece"],
    "HARDWARE_SCREEN_PHYSICAL_DAMAGE": ["screen", "display", "cracked", "flickering", "touch screen", "unresponsive screen"],
    "CAMERA_PHOTOS_RECORDING": ["camera", "photo", "blurry", "focus", "flash", "recording", "video recording"],
    "WATCH_AIRPODS_ACCESSORY_ECOSYSTEM": ["airpods", "apple watch", "pencil", "case", "pairing airpods"],
    "SYSTEM_PERFORMANCE_FREEZING": ["frozen", "freezing", "lagging", "slow", "unresponsive phone", "stuck on apple logo"]
}


class FollowUpAnalyzer:
    """Analyzes user turn context, follow-up intent, and attempted steps."""

    @staticmethod
    def classify_turn(
        current_query: str,
        recent_messages: List[Dict[str, Any]],
        previous_intent: Optional[str] = None
    ) -> Tuple[str, bool, Optional[str]]:
        """
        Classify the turn type:
          Returns: (follow_up_type, is_new_issue, detected_new_intent)
        """
        q = current_query.strip().lower()

        # If no prior conversation history exists
        if not recent_messages or not previous_intent:
            return "NEW_ISSUE", True, None

        # 1. Escalation request (highest priority)
        if any(phrase in q for phrase in ESCALATION_PHRASES):
            return "ESCALATION_REQUEST", False, previous_intent

        # 2. Confirmation (issue resolved)
        if any(phrase in q for phrase in CONFIRMATION_PHRASES):
            return "CONFIRMATION", False, previous_intent

        # 3. Correction
        if any(phrase in q for phrase in CORRECTION_PHRASES):
            return "CORRECTION", False, previous_intent

        # 4. Check if user is introducing a completely new distinct issue
        # E.g. "My WiFi is also not connecting" while previous was BATTERY
        for domain_intent, kws in DOMAIN_KEYWORDS.items():
            if domain_intent != previous_intent:
                # If query contains explicit keywords of another domain and is a symptom assertion
                if any(re.search(rf"\b{re.escape(kw)}\b", q) for kw in kws):
                    # Check if it has symptom/problem markers
                    problem_markers = ["not working", "not connecting", "won't", "cant", "can't", "broken", "issue", "problem", "failing", "also"]
                    if any(pm in q for pm in problem_markers):
                        return "NEW_ISSUE", True, domain_intent

        # 5. Unresolved follow-up
        if any(phrase in q for phrase in UNRESOLVED_PHRASES):
            return "UNRESOLVED_FOLLOW_UP", False, previous_intent

        # 6. Additional Information / already attempted
        if any(phrase in q for phrase in ADDITIONAL_INFO_PHRASES):
            return "ADDITIONAL_INFORMATION", False, previous_intent

        # 7. Conversational pronouns or short follow-ups referencing prior context
        short_followup_markers = ["that", "it", "this", "still", "what else", "what next", "how to", "how do i", "where"]
        words = q.split()
        if len(words) <= 7 and any(marker in q for marker in short_followup_markers):
            return "FOLLOW_UP", False, previous_intent

        # Default: if closely related to previous intent keywords, preserve intent as FOLLOW_UP
        if previous_intent and previous_intent in DOMAIN_KEYWORDS:
            prev_kws = DOMAIN_KEYWORDS[previous_intent]
            if any(re.search(rf"\b{re.escape(kw)}\b", q) for kw in prev_kws):
                return "FOLLOW_UP", False, previous_intent

        # If it doesn't clearly match previous intent and has substantial new words
        return "FOLLOW_UP", False, previous_intent

    @staticmethod
    def extract_attempted_steps_from_user(message: str) -> List[Tuple[str, str]]:
        """
        Extract concrete troubleshooting steps mentioned by the user as already tried.
        Returns list of (step_description, result).
        """
        q = message.lower()
        steps = []

        # Restart
        if any(w in q for w in ["restart", "restarted", "reboot", "rebooted"]):
            if any(p in q for p in ["tried", "already", "did", "have", "i restarted"]):
                steps.append(("Restart device", "FAILED"))

        # Battery usage / health
        if "battery usage" in q or "battery health" in q:
            if any(p in q for p in ["tried", "already", "checked", "looked at"]):
                steps.append(("Check Battery Usage", "FAILED"))

        # Update iOS
        if any(w in q for w in ["updated", "update", "ios update"]):
            if any(p in q for p in ["already", "tried", "have", "updated to"]):
                steps.append(("Update iOS", "FAILED"))

        # Reset network settings
        if "network settings" in q:
            if any(p in q for p in ["tried", "already", "reset"]):
                steps.append(("Reset Network Settings", "FAILED"))

        # Force restart / hard reset
        if "force restart" in q or "hard reset" in q:
            steps.append(("Force restart device", "FAILED"))

        return steps

    @staticmethod
    def extract_recommended_steps_from_assistant(response_text: str) -> List[str]:
        """
        Extract concrete troubleshooting steps recommended by the assistant.
        """
        t = response_text.lower()
        steps = []

        if "restart your" in t or "force restart" in t or "turn your device off and on" in t:
            steps.append("Restart device")

        if "settings > battery" in t or "battery usage" in t or "battery health" in t:
            steps.append("Check Battery Usage")

        if "update to the latest" in t or "update your iphone" in t or "settings > general > software update" in t:
            steps.append("Update iOS")

        if "reset network settings" in t:
            steps.append("Reset Network Settings")

        if "toggle wi-fi" in t or "turn off wi-fi" in t or "forget this network" in t:
            steps.append("Reset Wi-Fi Connection")

        return steps
