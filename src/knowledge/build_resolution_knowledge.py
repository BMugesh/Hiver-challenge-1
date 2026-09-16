"""
SupportDNA Knowledge Layer 2 — Resolution Knowledge Builder
===========================================================
Extracts grounded diagnostic steps, agent troubleshooting actions, verified resolution
steps, and empirical resolution patterns from the 7,922 curated historical cases (34,171 turns).

Files generated:
- data/knowledge/resolution/resolution_cases.jsonl
- data/knowledge/resolution/resolution_patterns.json
- data/knowledge/resolution/resolution_steps.jsonl
- data/knowledge/resolution/resolution_statistics.json
"""

import sys
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple
import pandas as pd

from src.knowledge.common import (
    load_resolved_threads,
    load_curated_intent_cases,
    load_splits_map,
    write_jsonl,
    write_json,
    RESOLUTION_DIR,
    clean_customer_text
)
from src.knowledge.provenance import ProvenanceRecord
from src.stage4_intent_discovery import extract_issues

# Diagnostic question patterns
DIAGNOSTIC_RE = re.compile(
    r'\b(?:what(?:[\'\’]s|\s+is)?\s+(?:the\s+)?(?:exact\s+)?(?:version|model|device)|'
    r'which\s+(?:version|ios|device|model|app)|'
    r'have\s+you\s+(?:tried|restarted|updated)|'
    r'does\s+this\s+happen|'
    r'when\s+did\s+this\s+start|'
    r'settings\s*(?:&gt;|>)\s*general\s*(?:&gt;|>)\s*about)\b',
    re.IGNORECASE
)

# Troubleshooting action patterns
STEP_PATTERNS = [
    ("Check iOS Version", re.compile(r'\bsettings\s*(?:&gt;|>)\s*general\s*(?:&gt;|>)\s*about\b', re.I)),
    ("Update iOS Software", re.compile(r'\b(?:backup\s*(?:and|&)\s*update|settings\s*(?:&gt;|>)\s*general\s*(?:&gt;|>)\s*software\s+update|update\s+to\s+ios)\b', re.I)),
    ("Check Battery Usage", re.compile(r'\bsettings\s*(?:&gt;|>)\s*battery\b', re.I)),
    ("Restart Device", re.compile(r'\b(?:restart|turn\s+(?:it\s+)?off\s+and\s+(?:back\s+)?on|power\s+down)\b', re.I)),
    ("Force Restart Device", re.compile(r'\bforce\s+restart\b', re.I)),
    ("Reset Network Settings", re.compile(r'\breset\s+network\s+settings\b', re.I)),
    ("Toggle Wi-Fi / Bluetooth", re.compile(r'\b(?:toggle|turn\s+(?:wifi|bluetooth)\s+off\s+and\s+on)\b', re.I)),
    ("Forget Wi-Fi Network", re.compile(r'\bforget\s+(?:this\s+)?network\b', re.I)),
    ("Text Replacement Workaround", re.compile(r'\bsettings\s*(?:&gt;|>)\s*general\s*(?:&gt;|>)\s*keyboard\s*(?:&gt;|>)\s*text\s+replacement\b', re.I)),
    ("Reset Apple ID Password", re.compile(r'\b(?:iforgot\.apple\.com|reset\s+your\s+password)\b', re.I)),
    ("Manage iCloud Storage", re.compile(r'\b(?:manage\s+storage|icloud\s+storage|settings\s*(?:&gt;|>)\s*.*icloud)\b', re.I)),
    ("Force Close and Reopen App", re.compile(r'\b(?:force\s+close|close\s+the\s+app\s+and\s+reopen)\b', re.I)),
    ("Reinstall App", re.compile(r'\b(?:delete\s+(?:and|&)\s*reinstall|reinstall\s+the\s+app)\b', re.I)),
    ("Review Official Support Article", re.compile(r'\bhttps?://(?:support\.apple\.com|apple\.co)\S+\b', re.I)),
]


def extract_diagnostic_questions(agent_turns: List[str]) -> List[str]:
    """Extract diagnostic questions asked by AppleSupport."""
    diagnostics = []
    for text in agent_turns:
        if "?" in text or DIAGNOSTIC_RE.search(text):
            cleaned = clean_customer_text(text)
            if len(cleaned) > 10:
                diagnostics.append(cleaned)
    return diagnostics


def extract_actionable_steps(agent_turns: List[str]) -> Tuple[List[str], List[str]]:
    """
    Extract discrete actionable instructions and canonical action names.
    Returns: (canonical_actions, raw_instruction_turns)
    """
    canonical_actions = []
    instruction_turns = []

    for text in agent_turns:
        matched_any = False
        for action_name, pattern in STEP_PATTERNS:
            if pattern.search(text):
                if action_name not in canonical_actions:
                    canonical_actions.append(action_name)
                matched_any = True

        if matched_any or "settings" in text.lower() or "try" in text.lower() or "please" in text.lower():
            cleaned = clean_customer_text(text)
            if len(cleaned) > 15:
                instruction_turns.append(cleaned)

    return canonical_actions, instruction_turns


def build_resolution_knowledge_layer() -> Dict[str, Any]:
    """
    Master builder for Layer 2: Resolution Knowledge.
    Processes the 7,922 curated resolved cases.
    """
    print("[Layer 2] Loading resolved threads and intent annotations...")
    resolved_threads = load_resolved_threads()
    curated_cases = load_curated_intent_cases()
    splits_map = load_splits_map()
    case_to_split = splits_map["case_to_split"]

    case_intent_map = {row["case_id"]: row["intent_id"] for _, row in curated_cases.iterrows()}

    print(f"[Layer 2] Processing {len(resolved_threads):,} curated resolved cases...")
    cases_records = []
    discrete_steps_records = []
    pattern_steps_by_intent = defaultdict(list)
    action_frequency = Counter()

    for thread in resolved_threads:
        case_id = thread["case_id"]
        root_id = thread["thread_root_id"]
        res_status = thread["resolution_status"]
        turns = thread.get("turns", [])

        split = case_to_split.get(case_id, "train")
        intent = case_intent_map.get(case_id, "GENERAL_DEVICE_INQUIRY")

        cust_turns = [t for t in turns if t.get("speaker") == "customer"]
        agent_turns = [t for t in turns if t.get("speaker") == "AppleSupport"]

        initial_problem = cust_turns[0].get("text", "") if cust_turns else ""
        symptoms = extract_issues(initial_problem, intent)

        agent_texts = [t.get("text", "") for t in agent_turns]
        diagnostic_qs = extract_diagnostic_questions(agent_texts)
        canonical_actions, instruction_turns = extract_actionable_steps(agent_texts)

        for act in canonical_actions:
            action_frequency[act] += 1
            pattern_steps_by_intent[intent].append(act)

        # Final resolution representation
        final_resolution = agent_texts[-1] if agent_texts else ""
        if cust_turns and res_status == "CLEARLY_RESOLVED":
            # Append confirmed fix quote if available
            final_resolution += f" | Customer Confirmation: {cust_turns[-1].get('text', '')}"

        quality = "HIGH" if res_status == "CLEARLY_RESOLVED" else "MEDIUM"
        turn_ids = [t["tweet_id"] for t in turns]

        provenance = ProvenanceRecord(
            source_thread_id=root_id,
            source_message_id=cust_turns[0]["tweet_id"] if cust_turns else None,
            source_case_id=case_id,
            source_dataset="apple_support_resolved_threads.json",
            original_status=res_status,
            derived_from_stage="STAGE_3_RESOLUTION_FILTER",
            split=split
        )

        case_record = {
            "case_id": case_id,
            "thread_id": root_id,
            "intent": intent,
            "customer_problem": initial_problem,
            "customer_symptoms": symptoms,
            "diagnostic_steps": diagnostic_qs[:3],  # top relevant diagnostic questions
            "support_actions": canonical_actions,
            "resolution_steps": instruction_turns[:5],  # top instructions provided
            "final_resolution": clean_customer_text(final_resolution),
            "resolution_status": res_status,
            "escalation_status": "NOT_ESCALATED",
            "evidence_turn_ids": turn_ids,
            "source_turn_count": len(turns),
            "resolution_quality": quality,
            "provenance": provenance.to_dict()
        }
        cases_records.append(case_record)

        # Discrete step records
        for idx, step_text in enumerate(instruction_turns[:5]):
            discrete_steps_records.append({
                "case_id": case_id,
                "step_index": idx + 1,
                "intent": intent,
                "instruction": step_text,
                "split": split,
                "provenance": provenance.to_dict()
            })

    # Synthesize Empirical Resolution Patterns per Intent
    resolution_patterns = {}
    for it, actions in pattern_steps_by_intent.items():
        act_counts = Counter(actions)
        sorted_steps = [act for act, _ in act_counts.most_common(6)]
        resolution_patterns[it] = {
            "intent": it,
            "total_resolved_cases": sum(1 for c in cases_records if c["intent"] == it),
            "canonical_resolution_workflow": sorted_steps,
            "common_action_frequencies": dict(act_counts.most_common(8))
        }

    # Statistics
    status_counts = Counter(c["resolution_status"] for c in cases_records)
    quality_counts = Counter(c["resolution_quality"] for c in cases_records)
    intent_counts = Counter(c["intent"] for c in cases_records)
    split_counts = Counter(c["provenance"]["split"] for c in cases_records)

    stats = {
        "layer_name": "LAYER_2_RESOLUTION_KNOWLEDGE",
        "total_resolution_cases": len(cases_records),
        "total_resolution_steps": len(discrete_steps_records),
        "resolution_status_distribution": dict(status_counts),
        "resolution_quality_distribution": dict(quality_counts),
        "split_distribution": dict(split_counts),
        "cases_per_intent": dict(intent_counts),
        "patterns_synthesized_count": len(resolution_patterns),
        "top_canonical_support_actions": dict(action_frequency.most_common(10))
    }

    print(f"[Layer 2] Exporting {len(cases_records):,} cases to {RESOLUTION_DIR}...")
    write_jsonl(cases_records, RESOLUTION_DIR / "resolution_cases.jsonl")
    write_jsonl(discrete_steps_records, RESOLUTION_DIR / "resolution_steps.jsonl")
    write_json(resolution_patterns, RESOLUTION_DIR / "resolution_patterns.json")
    write_json(stats, RESOLUTION_DIR / "resolution_statistics.json")

    print(f"[Layer 2] Finished. Exported {len(cases_records):,} resolved cases and {len(resolution_patterns)} intent patterns.")
    return stats


if __name__ == "__main__":
    build_resolution_knowledge_layer()
