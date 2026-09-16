"""
SupportDNA Knowledge Layer 1 — Intent & Language Knowledge Builder
==================================================================
Extracts customer problem descriptions, real-world phrasing, symptoms, and product
topics across the broad 80,247 conversation threads without requiring resolution.

Fields:
- thread_id, message_id, customer_message, conversation_context
- intent, issue, product_topic, conversation_length, source_type
- confidence, labeling_method, label_source (existing, inferred, uncertain)
- provenance: source_thread_id, source_message_id, source_case_id, source_dataset, original_status, derived_from_stage, split
"""

import sys
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple
import pandas as pd

from src.knowledge.common import (
    load_all_threads,
    load_curated_intent_cases,
    load_splits_map,
    load_resolved_threads,
    load_excluded_threads,
    write_jsonl,
    write_json,
    INTENT_DIR,
    clean_customer_text
)
from src.knowledge.provenance import ProvenanceRecord

# Import deterministic regex patterns from Stage 4
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

# Topic / Product Regex
PRODUCT_PATTERNS = [
    ("iPhone 8", re.compile(r'\biphone\s*8(?:\s*plus)?\b', re.I)),
    ("iPhone X", re.compile(r'\biphone\s*x(?:s|r|max)?\b', re.I)),
    ("iPhone 7", re.compile(r'\biphone\s*7(?:\s*plus)?\b', re.I)),
    ("iPhone 6s", re.compile(r'\biphone\s*6s(?:\s*plus)?\b', re.I)),
    ("iPhone 6", re.compile(r'\biphone\s*6(?:\s*plus)?\b', re.I)),
    ("iPhone SE", re.compile(r'\biphone\s*se\b', re.I)),
    ("iPhone", re.compile(r'\biphone\b', re.I)),
    ("iPad Pro", re.compile(r'\bipad\s*pro\b', re.I)),
    ("iPad Air", re.compile(r'\bipad\s*air\b', re.I)),
    ("iPad Mini", re.compile(r'\bipad\s*mini\b', re.I)),
    ("iPad", re.compile(r'\bipad\b', re.I)),
    ("MacBook Pro", re.compile(r'\bmacbook\s*pro\b', re.I)),
    ("MacBook Air", re.compile(r'\bmacbook\s*air\b', re.I)),
    ("MacBook", re.compile(r'\bmacbook\b', re.I)),
    ("Mac", re.compile(r'\b(?:imac|mac\s*mini|mac)\b', re.I)),
    ("Apple Watch", re.compile(r'\b(?:apple\s*watch|iwatch|watchos)\b', re.I)),
    ("AirPods", re.compile(r'\bairpods?\b', re.I)),
    ("iOS 11", re.compile(r'\bios\s*11(?:\.\d+)?\b', re.I)),
    ("iOS 10", re.compile(r'\bios\s*10(?:\.\d+)?\b', re.I)),
    ("App Store", re.compile(r'\bapp\s*store\b', re.I)),
    ("iCloud", re.compile(r'\bicloud\b', re.I)),
    ("Apple ID", re.compile(r'\bapple\s*id\b', re.I)),
    ("Apple Music", re.compile(r'\bapple\s*music\b', re.I)),
    ("iTunes", re.compile(r'\bitunes\b', re.I)),
]


def extract_product_topic(text: str) -> str:
    """Identify primary hardware or software topic mentioned."""
    for name, pattern in PRODUCT_PATTERNS:
        if pattern.search(text):
            return name
    return "General Apple Device/Service"


def infer_intent_from_text(text: str) -> Tuple[str, str, float]:
    """
    Infer intent using deterministic regex priority.
    Returns: (intent, label_source, confidence)
    """
    if KEYBOARD_RE.search(text):
        return "KEYBOARD_TYPING_AUTOCORRECT", "inferred", 0.85
    if BATTERY_RE.search(text):
        return "BATTERY_CHARGING_POWER", "inferred", 0.85
    if CONNECTIVITY_RE.search(text):
        return "CONNECTIVITY_WIFI_BLUETOOTH", "inferred", 0.85
    if DISPLAY_RE.search(text):
        return "DISPLAY_TOUCH_SCREEN", "inferred", 0.85
    if ACCOUNT_RE.search(text):
        return "ACCOUNT_APPLEID_ICLOUD", "inferred", 0.85
    if BILLING_RE.search(text):
        return "APP_STORE_PURCHASES_BILLING", "inferred", 0.85
    if APP_RE.search(text):
        return "APP_CRASH_AND_DOWNLOAD", "inferred", 0.85
    if AUDIO_RE.search(text):
        return "AUDIO_SOUND_SPEAKER", "inferred", 0.85
    if OS_RE.search(text):
        return "OS_UPDATE_SYSTEM_PERFORMANCE", "inferred", 0.85
    if HOWTO_RE.search(text):
        return "HOW_TO_SETTINGS_CONFIGURATION", "inferred", 0.85

    # If no specific technical rule fires, mark as uncertain
    return "GENERAL_DEVICE_INQUIRY", "uncertain", 0.50


def build_intent_language_layer() -> Dict[str, Any]:
    """
    Master builder for Layer 1: Intent & Language Knowledge.
    Processes all 80,247 conversation threads.
    """
    print("[Layer 1] Loading source datasets and splits...")
    all_threads = load_all_threads()
    splits_map = load_splits_map()
    root_to_split = splits_map["root_to_split"]
    case_to_split = splits_map["case_to_split"]

    # Load curated cases for golden labels
    curated_cases = load_curated_intent_cases()
    curated_lookup = {}
    for _, row in curated_cases.iterrows():
        root_id = int(row["thread_root_id"])
        curated_lookup[root_id] = {
            "case_id": str(row["case_id"]),
            "intent": str(row["intent_id"]),
            "resolution_status": str(row.get("resolution_status", "RESOLVED")),
        }

    # Load excluded threads map for original status
    excluded_threads = load_excluded_threads()
    excluded_status_map = {t["thread_root_id"]: t.get("resolution_status", "EXCLUDED") for t in excluded_threads}

    print(f"[Layer 1] Extracting customer language from {len(all_threads):,} threads...")
    examples = []
    text_freq = Counter()

    # Pre-pass for text frequency calculation (deduplication & frequency metrics)
    for thread in all_threads:
        for turn in thread.get("turns", []):
            if turn.get("speaker") == "customer":
                norm = clean_customer_text(turn.get("text", "")).lower()
                if norm:
                    text_freq[norm] += 1

    unique_messages_seen = set()
    threads_utilized = set()

    for thread in all_threads:
        root_id = thread["thread_root_id"]
        thread_len = thread["thread_length"]
        turns = thread.get("turns", [])

        # Check if thread is in curated set
        is_curated = root_id in curated_lookup
        curated_info = curated_lookup.get(root_id, {})
        split = root_to_split.get(root_id, "unassigned_excluded")

        if is_curated:
            original_status = curated_info["resolution_status"]
            case_id = curated_info["case_id"]
            gold_intent = curated_info["intent"]
        else:
            original_status = excluded_status_map.get(root_id, "EXCLUDED")
            case_id = None
            gold_intent = None

        customer_turn_idx = 0
        for turn in turns:
            if turn.get("speaker") == "customer":
                tweet_id = turn["tweet_id"]
                raw_text = turn.get("text", "")
                norm_text = clean_customer_text(raw_text).lower()

                if not raw_text.strip():
                    continue

                threads_utilized.add(root_id)
                unique_messages_seen.add(norm_text)

                source_type = "initial_inquiry" if customer_turn_idx == 0 else "follow_up_turn"
                customer_turn_idx += 1

                # Determine intent and label source
                if is_curated and customer_turn_idx == 1:
                    intent = gold_intent
                    label_source = "existing"
                    labeling_method = "curated_ground_truth"
                    confidence = 1.00
                else:
                    inferred_intent, l_source, conf = infer_intent_from_text(raw_text)
                    intent = inferred_intent
                    label_source = l_source
                    labeling_method = "symptom_regex_rule" if l_source == "inferred" else "fallback_default"
                    confidence = conf

                # Extract symptoms and topic
                detected_issues = extract_issues(raw_text, intent)
                issue = detected_issues[0] if detected_issues else "unspecified_issue"
                product_topic = extract_product_topic(raw_text)

                freq = text_freq.get(norm_text, 1)

                provenance = ProvenanceRecord(
                    source_thread_id=root_id,
                    source_message_id=tweet_id,
                    source_case_id=case_id,
                    source_dataset="apple_support_threads.json",
                    original_status=original_status,
                    derived_from_stage="STAGE_2_RECONSTRUCTED",
                    split=split
                )

                record = {
                    "thread_id": root_id,
                    "message_id": tweet_id,
                    "customer_message": raw_text,
                    "cleaned_message": clean_customer_text(raw_text),
                    "conversation_context": {
                        "turn_index": turn.get("turn_index", 0),
                        "total_turns": thread_len,
                        "is_initial": (customer_turn_idx == 1)
                    },
                    "intent": intent,
                    "issue": issue,
                    "all_issues": detected_issues,
                    "product_topic": product_topic,
                    "conversation_length": thread_len,
                    "source_type": source_type,
                    "confidence": confidence,
                    "labeling_method": labeling_method,
                    "label_source": label_source,
                    "phrasing_frequency": freq,
                    "provenance": provenance.to_dict()
                }
                examples.append(record)

    # Compute Statistics
    intent_counts = Counter(r["intent"] for r in examples)
    label_source_counts = Counter(r["label_source"] for r in examples)
    product_counts = Counter(r["product_topic"] for r in examples)
    split_counts = Counter(r["provenance"]["split"] for r in examples)

    stats = {
        "layer_name": "LAYER_1_INTENT_LANGUAGE",
        "source_threads_count": len(all_threads),
        "utilized_threads_count": len(threads_utilized),
        "thread_utilization_pct": round(len(threads_utilized) / len(all_threads) * 100, 2),
        "total_derived_records": len(examples),
        "unique_customer_messages": len(unique_messages_seen),
        "duplicate_messages_count": len(examples) - len(unique_messages_seen),
        "intents_represented_count": len(intent_counts),
        "records_per_intent": dict(intent_counts),
        "label_source_distribution": dict(label_source_counts),
        "split_distribution": dict(split_counts),
        "top_product_topics": dict(product_counts.most_common(10))
    }

    print(f"[Layer 1] Exporting {len(examples):,} records to {INTENT_DIR}...")
    jsonl_path = INTENT_DIR / "intent_language_examples.jsonl"
    write_jsonl(examples, jsonl_path)

    # Flatten for CSV representation
    flat_rows = []
    for r in examples:
        flat_rows.append({
            "thread_id": r["thread_id"],
            "message_id": r["message_id"],
            "customer_message": r["customer_message"][:200],  # preview for CSV
            "intent": r["intent"],
            "issue": r["issue"],
            "product_topic": r["product_topic"],
            "conversation_length": r["conversation_length"],
            "source_type": r["source_type"],
            "confidence": r["confidence"],
            "label_source": r["label_source"],
            "phrasing_frequency": r["phrasing_frequency"],
            "split": r["provenance"]["split"],
            "source_dataset": r["provenance"]["source_dataset"]
        })
    csv_path = INTENT_DIR / "intent_language_examples.csv"
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False, encoding="utf-8")

    stats_path = INTENT_DIR / "intent_language_statistics.json"
    write_json(stats, stats_path)

    print(f"[Layer 1] Finished. Derived {len(examples):,} records from {len(threads_utilized):,} threads.")
    return stats


if __name__ == "__main__":
    build_intent_language_layer()
