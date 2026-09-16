"""
Stage 5: Multi-Issue Detection & Context vs. Symptom Disambiguation
===================================================================
This module provides lightweight, deterministic multi-issue detection for
AppleSupport customer inquiries.

Key Capabilities:
1. Distinguishes Context Signals (e.g. 'after updating to iOS 11') from Symptom Signals
   (e.g. battery drain, screen freeze, keyboard lag).
2. Identifies whether an inquiry contains multiple distinct problem domains.
3. Provides conservative decision hints (ESCALATE_MULTI_ISSUE) for downstream Stage 7 safety.
"""

import re
from typing import Dict, List, Tuple, Any, Optional

# Context patterns (temporal or upgrade triggers that do NOT represent the actual defect)
CONTEXT_PATTERNS = [
    r"\bafter update\b",
    r"\bsince update\b",
    r"\bafter updating\b",
    r"\bsince updating\b",
    r"\bafter installing\b",
    r"\bfollowing the update\b",
    r"\bupdated to\b",
    r"\bnew ios\b",
    r"\bupdate got my\b",
    r"\bupdate made my\b",
    r"\bupgraded to\b",
    r"\bpost update\b",
    r"\blatest ios\b",
    r"\bnew update\b"
]

# Conjunction patterns that link distinct simultaneous problems
CONJUNCTION_PATTERNS = [
    r"\band\b",
    r"\b&\b",
    r"\balso\b",
    r"\bplus\b",
    r"\bas well\b",
    r"\bboth\b",
    r"\balong with\b",
    r"\bnot only\b",
    r"\btogether with\b",
    r"\bon top of that\b",
    r"\bin addition\b"
]

# Standard symptom keywords per support domain (derived from training taxonomy)
SYMPTOM_DOMAINS = {
    "BATTERY_CHARGING_POWER": [
        "battery", "drain", "draining", "charge", "charging", "charger",
        "overheating", "overheat", "dying fast", "battery life", "battery percentage",
        "power down", "shut off", "wont charge", "won't charge"
    ],
    "CONNECTIVITY_WIFI_BLUETOOTH": [
        "wifi", "wi-fi", "bluetooth", "cellular", "cellular data", "lte",
        "airdrop", "hotspot", "pairing", "no service", "carrier signal",
        "disconnecting wifi", "wont connect"
    ],
    "DISPLAY_TOUCH_SCREEN": [
        "touchscreen", "screen touch", "touch screen", "display", "unresponsive screen",
        "frozen screen", "screen freeze", "freezing screen", "black screen",
        "auto brightness", "screen flicker", "flickering", "3d touch", "touch id", "face id"
    ],
    "KEYBOARD_TYPING_AUTOCORRECT": [
        "keyboard", "autocorrect", "letter i", "predictive text", "typing lag",
        "type letter", "keypad", "auto correct", "capital i", "question box",
        "i glitch", "symbol box", "keyboard freeze"
    ],
    "AUDIO_SOUND_SPEAKER": [
        "speaker", "sound", "volume", "microphone", "mic", "earpiece",
        "headphones", "airpods audio", "quiet alarm", "crackling sound",
        "ringer", "no sound", "cannot hear"
    ],
    "APP_STORE_PURCHASES_BILLING": [
        "app store", "itunes", "apple music billing", "purchase", "billing",
        "refund", "subscription", "charged twice", "receipt", "credit card declined",
        "in-app purchase", "bought app"
    ],
    "APP_CRASH_AND_DOWNLOAD": [
        "app crash", "crashing", "crashes", "app freezes", "wont download app",
        "can't download app", "download spinning", "cant install app", "apps keep closing",
        "app closes"
    ],
    "ACCOUNT_APPLEID_ICLOUD": [
        "apple id", "icloud", "password reset", "appleid", "account locked",
        "id disabled", "activation lock", "two-factor", "2fa", "verification code",
        "icloud storage full", "icloud backup"
    ],
    "HOW_TO_SETTINGS_CONFIGURATION": [
        "how to change", "how do i disable", "how can i turn off", "settings configure",
        "how to customize", "how to delete app", "how to uninstall", "change wallpaper",
        "rearrange icons"
    ]
}


def extract_context_signals(query_text: str) -> List[str]:
    """Extract temporal or update-related context triggers from the query."""
    q_lower = query_text.lower()
    matched = []
    for pattern in CONTEXT_PATTERNS:
        matches = re.findall(pattern, q_lower)
        if matches:
            matched.extend(matches)
    return list(set(matched))


def extract_symptom_domains(
    query_text: str,
    prototypes: Optional[Dict[str, Any]] = None
) -> Dict[str, List[str]]:
    """
    Extract active symptom domains and matching distinguishing keywords.
    Uses custom prototype distinguishing terms if provided, otherwise defaults to SYMPTOM_DOMAINS.
    """
    q_lower = query_text.lower()
    active_domains = {}

    domain_specs = SYMPTOM_DOMAINS
    if prototypes:
        domain_specs = {}
        for intent_name, proto in prototypes.items():
            if intent_name in ["OS_UPDATE_SYSTEM_PERFORMANCE", "GENERAL_DEVICE_INQUIRY"]:
                continue
            terms = proto.get("distinguishing_terms", []) or proto.get("positive_terms", [])
            domain_specs[intent_name] = terms

    for domain, terms in domain_specs.items():
        matched_terms = []
        for term in terms:
            if re.search(r'\b' + re.escape(term) + r'\b', q_lower):
                matched_terms.append(term)
        if matched_terms:
            active_domains[domain] = matched_terms

    return active_domains


def detect_multi_issue(
    query_text: str,
    prototypes: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Analyze query to detect multiple distinct customer support issues.
    
    Returns structured analysis:
    - multi_issue: bool
    - symptom_domains: List[str]
    - symptom_keywords: Dict[str, List[str]]
    - has_context: bool
    - context_signals: List[str]
    - decision_hint: 'ESCALATE_MULTI_ISSUE' or 'PROCEED'
    - confidence: float (0.0 to 1.0)
    """
    q_lower = query_text.lower()
    context_signals = extract_context_signals(query_text)
    has_context = len(context_signals) > 0

    symptom_matches = extract_symptom_domains(query_text, prototypes)
    symptom_domains = list(symptom_matches.keys())

    # Check for conjunction patterns connecting symptoms
    has_conjunction = any(re.search(p, q_lower) for p in CONJUNCTION_PATTERNS)
    
    # Check for multi-sentence symptom distribution
    sentences = [s.strip() for s in re.split(r'[.!?\n]+', query_text) if len(s.strip()) > 3]
    multi_sentence = len(sentences) >= 2

    # Multi-issue condition: At least 2 distinct symptom domains with conjunction or sentence boundary
    is_multi_issue = (len(symptom_domains) >= 2 and (has_conjunction or multi_sentence))

    # Confidence score
    if is_multi_issue:
        confidence = min(0.95, 0.70 + 0.10 * len(symptom_domains))
        decision_hint = "ESCALATE_MULTI_ISSUE"
    else:
        confidence = 0.0
        decision_hint = "PROCEED"

    return {
        "multi_issue": is_multi_issue,
        "symptom_domains": symptom_domains,
        "symptom_keywords": symptom_matches,
        "has_context": has_context,
        "context_signals": context_signals,
        "decision_hint": decision_hint,
        "confidence": round(confidence, 4)
    }


def detect_multi_issue_batch(
    query_texts: List[str],
    prototypes: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Batch evaluate multi-issue status across a list of query texts."""
    return [detect_multi_issue(q, prototypes=prototypes) for q in query_texts]
