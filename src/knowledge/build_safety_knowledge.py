"""
SupportDNA Knowledge Layer 4 — Safety & Edge-Case Knowledge Builder
===================================================================
Identifies and catalogs fragile customer interactions, ambiguous queries, multi-issue
cases, and high-risk safety scenarios across the broad 80,247 dataset.

Edge Case Categories:
- ULTRA_SHORT
- AMBIGUOUS
- MULTI_ISSUE
- INSUFFICIENT_CONTEXT
- ESCALATION_SENSITIVE
- NO_RESOLUTION
- CONFLICTING_EVIDENCE
- WEAK_EVIDENCE
- UNUSUAL_PHRASING
- OTHER

Files generated:
- data/knowledge/safety/safety_edge_cases.jsonl
- data/knowledge/safety/safety_patterns.json
- data/knowledge/safety/safety_statistics.json
"""

import sys
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd

from src.knowledge.common import (
    load_all_threads,
    load_curated_intent_cases,
    load_splits_map,
    write_jsonl,
    write_json,
    SAFETY_DIR,
    clean_customer_text
)
from src.knowledge.provenance import ProvenanceRecord
from src.stage4_intent_discovery import (
    extract_issues,
    detect_multi_issue_signatures,
    KEYBOARD_RE,
    AUDIO_RE,
    BATTERY_RE,
    CONNECTIVITY_RE,
    DISPLAY_RE,
    ACCOUNT_RE,
    BILLING_RE,
    APP_RE,
    OS_RE,
    HOWTO_RE
)

# High-risk / sensitive topic patterns
SENSITIVE_TOPIC_RE = re.compile(
    r'\b(?:refund|chargeback|lawsuit|attorney|stolen|hacked|fraud|swelling|overheated.*burned|melted|smoke|fire|dangerous)\b',
    re.IGNORECASE
)

# Instruction injection / adversarial phrasing patterns
ADVERSARIAL_RE = re.compile(
    r'\b(?:ignore\s+(?:all\s+)?previous\s+instructions|system\s+prompt|say\s+["\']?you\s+will\s+refund|you\s+must\s+promise|override\s+policy)\b',
    re.IGNORECASE
)

# Vague / ambiguous phrasing patterns
VAGUE_PATTERNS = [
    re.compile(r'^(?:help|help\s+me|pls\s+help|please\s+help|anyone\s+there)\b', re.I),
    re.compile(r'^(?:my\s+phone\s+is\s+broken|it\s+doesnt\s+work|nothing\s+works)\b', re.I),
    re.compile(r'^(?:why|what\s+is\s+this|wtf|fix\s+this|so\s+bad)\b', re.I)
]


def classify_edge_case(
    text: str,
    turn_count: int,
    is_resolved: bool,
    issue_count: int
) -> Tuple[Optional[str], str, str]:
    """
    Classify edge-case type, ambiguity level, and safety relevance.
    Returns: (edge_case_type, ambiguity_level, safety_relevance) or (None, ...)
    """
    cleaned = clean_customer_text(text).strip()
    words = cleaned.split()
    word_count = len(words)

    # 1. Adversarial / Prompt Override
    if ADVERSARIAL_RE.search(cleaned):
        return (
            "POTENTIALLY_ADVERSARIAL",
            "HIGH",
            "Attempts to override agent policies or induce unauthorized promises (e.g. refunds)."
        )

    # 2. Escalation Sensitive (Financial dispute, account takeover, hardware safety hazards)
    if SENSITIVE_TOPIC_RE.search(cleaned):
        return (
            "ESCALATION_SENSITIVE",
            "HIGH",
            "Involves financial claims, security breaches, or physical safety risks requiring immediate human specialist handoff."
        )

    # 3. Ultra-Short Inquiries (<= 3 words without specific technical symptoms)
    if word_count <= 3 and word_count > 0:
        return (
            "ULTRA_SHORT",
            "HIGH",
            "Extremely brief query provides insufficient diagnostic context; high risk of hallucinated troubleshooting."
        )

    # 4. Multi-Issue Inquiries (2 or more distinct functional symptoms)
    if issue_count >= 2:
        return (
            "MULTI_ISSUE",
            "MEDIUM",
            "Customer states multiple distinct problems; single-intent retrieval will miss compound diagnostic needs."
        )

    # 5. Insufficient Context / Missing Antecedents
    if cleaned.lower().startswith(("it still", "why is it", "it is not", "did that", "done that", "not working")):
        return (
            "INSUFFICIENT_CONTEXT",
            "HIGH",
            "Inquiry refers to an unstated antecedent or prior turn not captured in current dialogue context."
        )

    # 6. Ambiguous / Vague Inquiries
    for p in VAGUE_PATTERNS:
        if p.search(cleaned):
            return (
                "AMBIGUOUS",
                "HIGH",
                "Broad non-specific grievance without actionable symptoms; automated reply must ask clarifying questions."
            )

    # 7. Unusual Phrasing (Emoji-only or extreme punctuation)
    emojis = re.findall(r'[^\w\s,.\'\"?!-]', cleaned)
    if len(emojis) > 8 or (len(emojis) > 0 and word_count <= 2):
        return (
            "UNUSUAL_PHRASING",
            "MEDIUM",
            "Heavy emoji usage or non-standard tokenization impairs standard embedding similarity."
        )

    # 8. Protracted No-Resolution Cases
    if turn_count >= 6 and not is_resolved:
        return (
            "NO_RESOLUTION",
            "MEDIUM",
            "Extended multi-turn thread that failed to resolve, indicating complex edge conditions or stubborn bugs."
        )

    return None, "LOW", ""


def build_safety_knowledge_layer() -> Dict[str, Any]:
    """
    Master builder for Layer 4: Safety & Edge-Case Knowledge.
    Scans the conversation universe to catalog edge cases and safety triggers.
    """
    print("[Layer 4] Loading conversation threads and splits...")
    all_threads = load_all_threads()
    curated_cases = load_curated_intent_cases()
    splits_map = load_splits_map()
    root_to_split = splits_map["root_to_split"]

    curated_roots = set(curated_cases["thread_root_id"].astype(int))

    print(f"[Layer 4] Scanning {len(all_threads):,} threads for edge cases and safety hazards...")
    edge_case_records = []
    category_counts = Counter()
    ambiguity_counts = Counter()
    intent_counts = Counter()

    for thread in all_threads:
        root_id = thread["thread_root_id"]
        turn_count = thread["thread_length"]
        turns = thread.get("turns", [])
        is_curated = root_id in curated_roots
        split = root_to_split.get(root_id, "unassigned_excluded")

        cust_turns = [t for t in turns if t.get("speaker") == "customer"]
        if not cust_turns:
            continue

        first_text = cust_turns[0].get("text", "")
        if not first_text.strip():
            continue

        # Extract multi-issue signatures
        signatures = detect_multi_issue_signatures(first_text)
        issue_count = len(signatures)

        edge_type, ambiguity, relevance = classify_edge_case(
            text=first_text,
            turn_count=turn_count,
            is_resolved=is_curated,
            issue_count=issue_count
        )

        if edge_type is not None:
            category_counts[edge_type] += 1
            ambiguity_counts[ambiguity] += 1

            primary_intent = signatures[0] if signatures else "GENERAL_DEVICE_INQUIRY"
            intent_counts[primary_intent] += 1

            provenance = ProvenanceRecord(
                source_thread_id=root_id,
                source_message_id=cust_turns[0]["tweet_id"],
                source_case_id=None,
                source_dataset="apple_support_threads.json",
                original_status="RESOLVED" if is_curated else "EXCLUDED",
                derived_from_stage="STAGE_2_RECONSTRUCTED",
                split=split
            )

            record = {
                "thread_id": root_id,
                "customer_message": first_text,
                "cleaned_message": clean_customer_text(first_text),
                "intent": primary_intent,
                "issue_count": issue_count,
                "ambiguity_level": ambiguity,
                "evidence_available": is_curated,
                "resolution_available": is_curated,
                "escalation_status": "ESCALATE" if not is_curated else "RESOLVED_HISTORICAL",
                "edge_case_type": edge_type,
                "safety_relevance": relevance,
                "provenance": provenance.to_dict()
            }
            edge_case_records.append(record)

    # Synthesize Safety Handling Patterns Catalog
    safety_patterns = {
        "catalog_overview": "Precautionary operating rules for unanswerable, ambiguous, or risky queries.",
        "edge_case_policies": [
            {
                "edge_case_type": "ULTRA_SHORT",
                "guidance": "Inquiries <= 3 words lack actionable diagnostic context. Do not guess; issue structured diagnostic menu.",
                "action": "DIAGNOSTIC_CLARIFICATION"
            },
            {
                "edge_case_type": "ESCALATION_SENSITIVE",
                "guidance": "Involves refunds, billing dispute, physical hardware swelling, or account theft. Bypass autonomous agent.",
                "action": "IMMEDIATE_HUMAN_ESCALATION"
            },
            {
                "edge_case_type": "MULTI_ISSUE",
                "guidance": "Inquiry spans multiple unrelated hardware/software symptoms. Address sequentially or transfer to specialist.",
                "action": "TRIAGE_OR_ESCALATE"
            },
            {
                "edge_case_type": "AMBIGUOUS",
                "guidance": "Vague symptom statements. Narrow down problem domain before attempting retrieval.",
                "action": "DIAGNOSTIC_CLARIFICATION"
            },
            {
                "edge_case_type": "POTENTIALLY_ADVERSARIAL",
                "guidance": "System prompt exfiltration or policy override attempts. Enforce XML data delimiters and hard reject.",
                "action": "NEUTRALIZE_AND_REJECT"
            },
            {
                "edge_case_type": "INSUFFICIENT_CONTEXT",
                "guidance": "Query references prior unstated turns. Prompt customer to restate device model and symptom.",
                "action": "REQUEST_CONTEXT"
            }
        ],
        "category_distribution": dict(category_counts),
        "ambiguity_distribution": dict(ambiguity_counts)
    }

    stats = {
        "layer_name": "LAYER_4_SAFETY_EDGE_CASE_KNOWLEDGE",
        "total_source_threads_scanned": len(all_threads),
        "total_edge_cases_identified": len(edge_case_records),
        "edge_case_prevalence_pct": round(len(edge_case_records) / len(all_threads) * 100, 2),
        "category_distribution": dict(category_counts),
        "ambiguity_level_distribution": dict(ambiguity_counts),
        "intents_affected": dict(intent_counts)
    }

    print(f"[Layer 4] Exporting {len(edge_case_records):,} edge cases to {SAFETY_DIR}...")
    write_jsonl(edge_case_records, SAFETY_DIR / "safety_edge_cases.jsonl")
    write_json(safety_patterns, SAFETY_DIR / "safety_patterns.json")
    write_json(stats, SAFETY_DIR / "safety_statistics.json")

    print(f"[Layer 4] Finished. Identified {len(edge_case_records):,} safety edge cases.")
    return stats


if __name__ == "__main__":
    build_safety_knowledge_layer()
