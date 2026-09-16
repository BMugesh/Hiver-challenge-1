"""
Unit tests for Stage 3: Resolution Filtering & Classification Logic.
Tests resolution status classification across diverse conversation patterns.
"""

import unittest
import re
from typing import Dict, List, Any, Tuple


# Regex patterns matching domain-grounded heuristics
CONFIRMATION_PATTERNS = [
    r'\b(?:it\s+)?worked\b', r'\bfixed\s+(?:it|my|the|this)\b', r'\b(?:all\s+)?good\s+now\b',
    r'\bthanks?(?:\s+you)?\s*(?:!|\.|\b.*(?:helped|working|fixed|awesome|lifesaver|appreciate|solved))',
    r'\bthank\s+you\s+so\s+much\b', r'\bproblem\s+solved\b', r'\bgot\s+it\s+working\b',
    r'\bthat\s+(?:did\s+the\s+trick|solved\s+it|helped)\b', r'\bworking\s+(?:fine|again|now)\b'
]
CONFIRM_RE = re.compile('|'.join(CONFIRMATION_PATTERNS), re.IGNORECASE)

DM_PATTERNS = [
    r'\b(?:dm|direct\s+message)\s+us\b', r'\bjoin\s+us\s+in\s+(?:a\s+)?dm\b',
    r'\bsend\s+(?:us\s+)?(?:a\s+)?dm\b', r'\bmeet\s+us\s+in\s+dm\b', r'\bcontinue\s+(?:in|via)\s+dm\b',
    r'\bdms?\b.*https://t\.co/'
]
DM_RE = re.compile('|'.join(DM_PATTERNS), re.IGNORECASE)

INST_PATTERNS = [
    r'\btap\s+settings\b', r'\bsettings\s*(?:&gt;|>)\s*', r'\bcheck\s+out\s+this\s+article\b',
    r'\bfollow\s+these\s+steps\b', r'\bhttps?://(?:support\.apple\.com|apple\.co)\b', r'\bto\s+resolve\s+this\b',
    r'\brestart\s+(?:your\s+)?(?:device|iphone|ipad|mac)\b', r'\bupdate\s+to\s+ios\b', r'\btry\s+resetting\b',
    r'\bforce\s+restart\b', r'\bturn\s+(?:it\s+)?off\s+and\s+(?:back\s+)?on\b'
]
INST_RE = re.compile('|'.join(INST_PATTERNS), re.IGNORECASE)


def classify_conversation(
    turns: List[Dict[str, Any]],
    ts_outcome: str = "UNKNOWN",
    ts_action: str = "UNKNOWN",
) -> Tuple[str, str]:
    """
    Classify conversation resolution status based on text heuristics and validation signals.
    Returns: (resolution_status, reason)
    """
    if not turns:
        return "UNCLEAR", "Empty conversation"

    turn_count = len(turns)
    last_turn = turns[-1]
    last_speaker = last_turn.get("speaker", "unknown")
    last_text = last_turn.get("text", "")

    has_customer_confirmation = False
    has_agent_instructions = False
    has_agent_dm_redirect = False

    agent_seen = False
    for turn in turns:
        speaker = turn.get("speaker", "unknown")
        text = turn.get("text", "")
        if speaker == "AppleSupport":
            agent_seen = True
            if INST_RE.search(text):
                has_agent_instructions = True
            if DM_RE.search(text):
                has_agent_dm_redirect = True
        elif speaker == "customer" and agent_seen:
            if CONFIRM_RE.search(text):
                has_customer_confirmation = True

    # 1. CLEARLY_RESOLVED: Customer explicitly confirms fix / gratitude
    if has_customer_confirmation or ts_outcome == "RESOLVED":
        return "CLEARLY_RESOLVED", "Customer confirmed fix or problem resolution"

    # 2. PARTIALLY_RESOLVED: Concrete instructions provided without customer confirmation
    if (ts_outcome == "LIKELY_RESOLVED" or has_agent_instructions) and not has_agent_dm_redirect:
        return "PARTIALLY_RESOLVED", "AppleSupport provided concrete instructions without explicit customer confirmation"

    # 3. ESCALATED: DM redirect or out-of-channel handoff
    if (
        ts_outcome == "ESCALATED"
        or has_agent_dm_redirect
        or "REDIRECT_DM" in ts_action
        or "ESCALATE" in ts_action
    ):
        return "ESCALATED", "AppleSupport redirected conversation to Direct Message or internal escalation"

    # 4. ABANDONED: 2-turn thread ending on agent diagnostic question
    if turn_count == 2 and last_speaker == "AppleSupport" and ("REQUEST" in ts_action or "?" in last_text):
        return "ABANDONED", "Customer stopped responding after AppleSupport asked diagnostic question"

    # 5. UNCLEAR: Ambiguous or open dialogue
    return "UNCLEAR", "Inconclusive interaction without actionable resolution"


class TestResolutionFiltering(unittest.TestCase):
    """Unit tests for resolution classification policy."""

    def test_explicit_customer_confirmation(self):
        """Test 1: Explicit customer confirmation -> CLEARLY_RESOLVED."""
        turns = [
            {"speaker": "customer", "text": "My phone is not charging"},
            {"speaker": "AppleSupport", "text": "Try cleaning the lightning port with a brush"},
            {"speaker": "customer", "text": "That worked! Thank you so much!"},
        ]
        status, reason = classify_conversation(turns)
        self.assertEqual(status, "CLEARLY_RESOLVED")

    def test_instructions_without_confirmation(self):
        """Test 2: Instructions provided but no confirmation -> PARTIALLY_RESOLVED (not clearly resolved)."""
        turns = [
            {"speaker": "customer", "text": "How do I turn off auto-brightness?"},
            {"speaker": "AppleSupport", "text": "Tap Settings > General > Accessibility > Display Accommodations > Auto-Brightness"},
        ]
        status, reason = classify_conversation(turns)
        self.assertEqual(status, "PARTIALLY_RESOLVED")
        self.assertNotEqual(status, "CLEARLY_RESOLVED")

    def test_abandoned_after_question(self):
        """Test 3: Customer stops responding after diagnostic question -> ABANDONED."""
        turns = [
            {"speaker": "customer", "text": "My battery dies super fast"},
            {"speaker": "AppleSupport", "text": "Which version of iOS are you running on your device?"},
        ]
        status, reason = classify_conversation(turns)
        self.assertEqual(status, "ABANDONED")

    def test_redirect_to_dm(self):
        """Test 4: DM redirect -> ESCALATED (not resolved)."""
        turns = [
            {"speaker": "customer", "text": "Someone hacked my Apple ID account"},
            {"speaker": "AppleSupport", "text": "Let's take a look. Send us a DM here: https://t.co/xyz with your details."},
        ]
        status, reason = classify_conversation(turns)
        self.assertEqual(status, "ESCALATED")
        self.assertNotEqual(status, "CLEARLY_RESOLVED")

    def test_multi_turn_customer_fixed(self):
        """Test 5: Multi-turn troubleshooting ending in confirmed fix -> CLEARLY_RESOLVED."""
        turns = [
            {"speaker": "customer", "text": "My Apple Watch won't pair with iPhone X"},
            {"speaker": "AppleSupport", "text": "Is Bluetooth turned on in Settings?"},
            {"speaker": "customer", "text": "Yes it is on"},
            {"speaker": "AppleSupport", "text": "Please restart both devices and try again"},
            {"speaker": "customer", "text": "Restarted both and it paired. Fixed the issue, thanks!"},
        ]
        status, reason = classify_conversation(turns)
        self.assertEqual(status, "CLEARLY_RESOLVED")

    def test_internal_escalation_handoff(self):
        """Test 6: Escalation action -> ESCALATED."""
        turns = [
            {"speaker": "customer", "text": "Hardware defect on brand new iMac screen"},
            {"speaker": "AppleSupport", "text": "Please join us in DM to set up an appointment at the Genius Bar."},
        ]
        status, reason = classify_conversation(turns, ts_action="ESCALATE_INTERNAL")
        self.assertEqual(status, "ESCALATED")


if __name__ == "__main__":
    unittest.main()
