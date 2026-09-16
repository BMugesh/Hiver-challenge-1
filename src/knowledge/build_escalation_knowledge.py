"""
SupportDNA Knowledge Layer 3 — Escalation Knowledge Builder
===========================================================
Extracts escalation reasons, triggers, and patterns from the 72,325 excluded threads
to understand when, why, and how AppleSupport stops public troubleshooting.

Categories:
- ESCALATED_TO_DM
- UNRESOLVED
- ABANDONED
- UNCLEAR
- PARTIAL_RESOLUTION
- OTHER

Files generated:
- data/knowledge/escalation/escalation_cases.jsonl
- data/knowledge/escalation/escalation_patterns.json
- data/knowledge/escalation/escalation_statistics.json
"""

import sys
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple
import pandas as pd

from src.knowledge.common import (
    load_excluded_threads,
    write_jsonl,
    write_json,
    ESCALATION_DIR,
    clean_customer_text
)
from src.knowledge.provenance import ProvenanceRecord
from src.stage3_resolution_filter import DM_RE, INST_RE
from src.stage4_intent_discovery import (
    KEYBOARD_RE,
    AUDIO_RE,
    BATTERY_RE,
    CONNECTIVITY_RE,
    DISPLAY_RE,
    ACCOUNT_RE,
    BILLING_RE,
    APP_RE,
    OS_RE,
    HOWTO_RE,
    extract_issues
)

# Escalation Trigger Heuristics
HARDWARE_SERVICE_RE = re.compile(
    r'\b(?:genius\s*bar|apple\s*store|repair|hardware|screen\s*(?:cracked|broken)|service\s*(?:location|center)|battery\s*replacement)\b',
    re.IGNORECASE
)
CREDENTIALS_RE = re.compile(
    r'\b(?:apple\s*id|password|passcode|imei|serial\s*number|billing|credit\s*card|security\s*question|verification\s*code)\b',
    re.IGNORECASE
)
RECURRING_FAILURE_RE = re.compile(
    r'\b(?:already\s*tried|still\s*not\s*working|did\s*(?:that|all\s*of\s*that)|doesn\'?t\s*help|didn\'?t\s*work|same\s*issue)\b',
    re.IGNORECASE
)


def infer_quick_intent(text: str) -> str:
    """Infer intent for uncurated excluded thread."""
    if KEYBOARD_RE.search(text):
        return "KEYBOARD_TYPING_AUTOCORRECT"
    if BATTERY_RE.search(text):
        return "BATTERY_CHARGING_POWER"
    if CONNECTIVITY_RE.search(text):
        return "CONNECTIVITY_WIFI_BLUETOOTH"
    if DISPLAY_RE.search(text):
        return "DISPLAY_TOUCH_SCREEN"
    if ACCOUNT_RE.search(text):
        return "ACCOUNT_APPLEID_ICLOUD"
    if BILLING_RE.search(text):
        return "APP_STORE_PURCHASES_BILLING"
    if APP_RE.search(text):
        return "APP_CRASH_AND_DOWNLOAD"
    if AUDIO_RE.search(text):
        return "AUDIO_SOUND_SPEAKER"
    if OS_RE.search(text):
        return "OS_UPDATE_SYSTEM_PERFORMANCE"
    if HOWTO_RE.search(text):
        return "HOW_TO_SETTINGS_CONFIGURATION"
    return "GENERAL_DEVICE_INQUIRY"


def categorize_escalation(
    orig_status: str,
    turns: List[Dict[str, Any]],
    has_dm: bool,
    has_inst: bool
) -> Tuple[str, str, str]:
    """
    Categorize escalation trajectory and determine specific reason.
    Returns: (escalation_status, reason_code, escalation_pattern)
    """
    cust_texts = [t.get("text", "") for t in turns if t.get("speaker") == "customer"]
    agent_texts = [t.get("text", "") for t in turns if t.get("speaker") == "AppleSupport"]

    combined_cust = " ".join(cust_texts)
    combined_agent = " ".join(agent_texts)
    turn_count = len(turns)

    if has_dm or orig_status == "ESCALATED":
        status = "ESCALATED_TO_DM"
        if CREDENTIALS_RE.search(combined_cust) or CREDENTIALS_RE.search(combined_agent):
            reason = "PRIVATE_CREDENTIALS_AND_ACCOUNT_SECURITY"
            pattern = "Customer requested account/billing help -> AppleSupport shifted to DM to protect private info."
        elif HARDWARE_SERVICE_RE.search(combined_cust) or HARDWARE_SERVICE_RE.search(combined_agent):
            reason = "HARDWARE_SERVICE_AND_REPAIR_TRIAGE"
            pattern = "Hardware failure or repair need -> AppleSupport directed to private DM/Store triage."
        elif RECURRING_FAILURE_RE.search(combined_cust):
            reason = "FAILED_PUBLIC_TROUBLESHOOTING"
            pattern = "Standard troubleshooting failed in-channel -> Escalated to DM for advanced diagnostics."
        else:
            reason = "GENERAL_CHANNEL_DM_REDIRECT"
            pattern = "Direct message redirect initiated for individualized support."
    elif orig_status == "ABANDONED" or (turn_count == 2 and turns[-1].get("speaker") == "AppleSupport" and "?" in turns[-1].get("text", "")):
        status = "ABANDONED"
        reason = "CUSTOMER_UNRESPONSIVE_TO_DIAGNOSTIC"
        pattern = "AppleSupport asked diagnostic question -> Customer stopped responding before troubleshooting."
    elif has_inst and not has_dm:
        status = "PARTIAL_RESOLUTION"
        reason = "UNCONFIRMED_INSTRUCTION_PROVIDED"
        pattern = "Technical instructions provided but customer did not confirm outcome in-thread."
    elif turn_count <= 2:
        status = "UNRESOLVED"
        reason = "INCOMPLETE_CONVERSATION_THREAD"
        pattern = "Dialogue ended prematurely without diagnostic progression or resolution."
    else:
        status = "UNCLEAR"
        reason = "INCONCLUSIVE_MULTI_TURN_EXCHANGE"
        pattern = "Complex, noisy, or multi-party conversation without clear support outcome."

    return status, reason, pattern


def build_escalation_knowledge_layer() -> Dict[str, Any]:
    """
    Master builder for Layer 3: Escalation Knowledge.
    Processes the 72,325 excluded threads.
    """
    print("[Layer 3] Loading excluded threads...")
    excluded_threads = load_excluded_threads()

    print(f"[Layer 3] Processing {len(excluded_threads):,} excluded threads...")
    escalation_records = []
    category_counts = Counter()
    reason_counts = Counter()
    intent_counts = Counter()
    dm_count = 0

    pattern_frequency = Counter()

    for thread in excluded_threads:
        root_id = thread["thread_root_id"]
        orig_status = thread.get("resolution_status", "EXCLUDED")
        turns = thread.get("turns", [])
        turn_count = len(turns)

        cust_turns = [t for t in turns if t.get("speaker") == "customer"]
        agent_turns = [t for t in turns if t.get("speaker") == "AppleSupport"]

        first_cust = cust_turns[0].get("text", "") if cust_turns else ""
        intent = infer_quick_intent(first_cust)
        issues = extract_issues(first_cust, intent)
        primary_issue = issues[0] if issues else "unspecified_inquiry"

        has_dm = any(DM_RE.search(t.get("text", "")) for t in agent_turns)
        has_inst = any(INST_RE.search(t.get("text", "")) for t in agent_turns)

        if has_dm:
            dm_count += 1

        esc_status, reason_code, esc_pattern = categorize_escalation(
            orig_status=orig_status,
            turns=turns,
            has_dm=has_dm,
            has_inst=has_inst
        )

        category_counts[esc_status] += 1
        reason_counts[reason_code] += 1
        intent_counts[intent] += 1
        pattern_frequency[esc_pattern] += 1

        conv_complete = (esc_status in ["ESCALATED_TO_DM", "PARTIAL_RESOLUTION"])

        provenance = ProvenanceRecord(
            source_thread_id=root_id,
            source_message_id=cust_turns[0]["tweet_id"] if cust_turns else None,
            source_case_id=None,
            source_dataset="apple_support_excluded_threads.json",
            original_status=orig_status,
            derived_from_stage="STAGE_3_RESOLUTION_FILTER",
            split="unassigned_excluded"
        )

        record = {
            "thread_id": root_id,
            "intent": intent,
            "customer_issue": primary_issue,
            "escalation_status": esc_status,
            "escalation_reason": reason_code,
            "dm_requested": has_dm,
            "conversation_complete": conv_complete,
            "resolution_present": False,
            "evidence_quality": "ESCALATION_ONLY" if has_dm else "LOW",
            "escalation_pattern": esc_pattern,
            "conversation_length": turn_count,
            "customer_inquiry": first_cust[:300],
            "provenance": provenance.to_dict()
        }
        escalation_records.append(record)

    # Synthesize Escalation Patterns Catalog
    escalation_patterns = {
        "catalog_overview": "Empirical patterns explaining why AppleSupport ceased public troubleshooting.",
        "top_escalation_triggers": [
            {
                "trigger": "PRIVATE_CREDENTIALS_AND_ACCOUNT_SECURITY",
                "frequency": reason_counts["PRIVATE_CREDENTIALS_AND_ACCOUNT_SECURITY"],
                "description": "Apple ID, iCloud password, two-factor auth, and billing requests require private identity verification.",
                "policy_guidance": "Escalate immediately to secure DM or official Apple ID recovery flow."
            },
            {
                "trigger": "CUSTOMER_UNRESPONSIVE_TO_DIAGNOSTIC",
                "frequency": reason_counts["CUSTOMER_UNRESPONSIVE_TO_DIAGNOSTIC"],
                "description": "Customer did not answer diagnostic query (device model, iOS version).",
                "policy_guidance": "Do not hallucinate resolution; prompt for diagnostic parameter before advancing."
            },
            {
                "trigger": "HARDWARE_SERVICE_AND_REPAIR_TRIAGE",
                "frequency": reason_counts["HARDWARE_SERVICE_AND_REPAIR_TRIAGE"],
                "description": "Physical cracks, water damage, swollen battery, or component failure.",
                "policy_guidance": "Escalate to Apple Store Genius Bar or Authorized Service Provider."
            },
            {
                "trigger": "FAILED_PUBLIC_TROUBLESHOOTING",
                "frequency": reason_counts["FAILED_PUBLIC_TROUBLESHOOTING"],
                "description": "Customer already attempted basic restart/reset steps with no success.",
                "policy_guidance": "Avoid repeating failed instructions; escalate to human specialist."
            }
        ],
        "category_distribution": dict(category_counts),
        "frequent_patterns": dict(pattern_frequency.most_common(6))
    }

    stats = {
        "layer_name": "LAYER_3_ESCALATION_KNOWLEDGE",
        "total_source_threads": len(excluded_threads),
        "total_derived_records": len(escalation_records),
        "dm_redirect_count": dm_count,
        "dm_redirect_percentage": round(dm_count / len(excluded_threads) * 100, 2),
        "escalation_status_distribution": dict(category_counts),
        "escalation_reason_distribution": dict(reason_counts),
        "records_per_intent": dict(intent_counts)
    }

    print(f"[Layer 3] Exporting {len(escalation_records):,} records to {ESCALATION_DIR}...")
    write_jsonl(escalation_records, ESCALATION_DIR / "escalation_cases.jsonl")
    write_json(escalation_patterns, ESCALATION_DIR / "escalation_patterns.json")
    write_json(stats, ESCALATION_DIR / "escalation_statistics.json")

    print(f"[Layer 3] Finished. Exported {len(escalation_records):,} escalation records.")
    return stats


if __name__ == "__main__":
    build_escalation_knowledge_layer()
