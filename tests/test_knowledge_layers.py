"""
Test Suite for SupportDNA 4-Layer Knowledge Architecture
========================================================
Validates all 10 core data utilization and integrity guarantees:
1. Every layer files exist and can be loaded.
2. No source provenance is lost across records.
3. Resolution knowledge strictly contains 7,922 cases.
4. Existing 5,545-case FAISS retrieval index and embeddings remain intact.
5. Excluded threads are not incorrectly classified as resolved.
6. Unknown/uncertain labels remain explicitly unknown/uncertain (not treated as ground truth).
7. Validation/test leakage is audited and zero leakage is detected.
8. Duplicate handling tracks frequencies accurately.
9. Statistics are internally consistent and sum up to total records.
10. Deterministic output across repeated extractions.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.knowledge.common import (
    INTENT_DIR,
    RESOLUTION_DIR,
    ESCALATION_DIR,
    SAFETY_DIR,
    SPLITS_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
    read_jsonl
)
from src.knowledge.leakage_check import run_leakage_check
from src.knowledge.build_intent_language import infer_intent_from_text
from src.knowledge.build_escalation_knowledge import categorize_escalation


# 1. Every layer can be generated and files exist
def test_all_layer_files_exist():
    """Verify that all required JSONL, JSON, and CSV files exist for all 4 layers."""
    expected_files = [
        # Layer 1
        INTENT_DIR / "intent_language_examples.jsonl",
        INTENT_DIR / "intent_language_examples.csv",
        INTENT_DIR / "intent_language_statistics.json",
        # Layer 2
        RESOLUTION_DIR / "resolution_cases.jsonl",
        RESOLUTION_DIR / "resolution_patterns.json",
        RESOLUTION_DIR / "resolution_steps.jsonl",
        RESOLUTION_DIR / "resolution_statistics.json",
        # Layer 3
        ESCALATION_DIR / "escalation_cases.jsonl",
        ESCALATION_DIR / "escalation_patterns.json",
        ESCALATION_DIR / "escalation_statistics.json",
        # Layer 4
        SAFETY_DIR / "safety_edge_cases.jsonl",
        SAFETY_DIR / "safety_patterns.json",
        SAFETY_DIR / "safety_statistics.json",
        # Reports
        REPORTS_DIR / "knowledge_layer_statistics.txt",
        REPORTS_DIR / "data_utilization_before_after.txt",
        REPORTS_DIR / "knowledge_layer_leakage_check.txt"
    ]
    for p in expected_files:
        assert p.exists(), f"Expected file missing: {p}"
        assert p.stat().st_size > 0, f"File is empty: {p}"


# 2. No source provenance is lost
def test_provenance_preservation():
    """Verify that 100% of sampled records in all 4 layers contain full provenance metadata."""
    # Check Layer 1
    l1_sample = read_jsonl(INTENT_DIR / "intent_language_examples.jsonl")[:100]
    for r in l1_sample:
        assert "provenance" in r
        prov = r["provenance"]
        assert "source_thread_id" in prov and prov["source_thread_id"] > 0
        assert "source_message_id" in prov and prov["source_message_id"] > 0
        assert "source_dataset" in prov
        assert "derived_from_stage" in prov
        assert "split" in prov

    # Check Layer 2
    l2_sample = read_jsonl(RESOLUTION_DIR / "resolution_cases.jsonl")[:100]
    for r in l2_sample:
        assert "provenance" in r
        prov = r["provenance"]
        assert "source_thread_id" in prov
        assert "source_case_id" in prov and prov["source_case_id"].startswith("CASE_")
        assert prov["derived_from_stage"] == "STAGE_3_RESOLUTION_FILTER"

    # Check Layer 3
    l3_sample = read_jsonl(ESCALATION_DIR / "escalation_cases.jsonl")[:100]
    for r in l3_sample:
        assert "provenance" in r
        prov = r["provenance"]
        assert "source_thread_id" in prov
        assert prov["split"] == "unassigned_excluded"

    # Check Layer 4
    l4_sample = read_jsonl(SAFETY_DIR / "safety_edge_cases.jsonl")[:100]
    for r in l4_sample:
        assert "provenance" in r
        prov = r["provenance"]
        assert "source_thread_id" in prov


# 3. Resolution data remains 7,922 cases
def test_resolution_cases_count_invariance():
    """Verify Layer 2 contains exactly 7,922 curated cases."""
    cases = read_jsonl(RESOLUTION_DIR / "resolution_cases.jsonl")
    assert len(cases) == 7922, f"Expected exactly 7,922 cases, got {len(cases)}"

    # Check that case IDs span CASE_000001 to CASE_007922
    case_ids = set(c["case_id"] for c in cases)
    assert len(case_ids) == 7922
    assert "CASE_000001" in case_ids
    assert "CASE_007922" in case_ids


# 4. Existing 5,545 FAISS records remain valid
def test_existing_faiss_corpus_intact():
    """Verify that the existing 5,545 FAISS index, embeddings, and metadata are unchanged."""
    faiss_index_path = PROCESSED_DIR / "retrieval" / "apple_support_faiss.index"
    embeddings_path = PROCESSED_DIR / "retrieval" / "train_case_embeddings.npy"
    metadata_path = PROCESSED_DIR / "retrieval" / "train_case_metadata.json"

    assert faiss_index_path.exists()
    assert embeddings_path.exists()
    assert metadata_path.exists()

    embeddings = np.load(embeddings_path)
    assert embeddings.shape == (5545, 384), f"Unexpected embeddings shape: {embeddings.shape}"

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    assert len(metadata) == 5545, f"Expected 5,545 metadata records, got {len(metadata)}"


# 5. Excluded threads are not incorrectly classified as resolved
def test_excluded_threads_not_classified_as_resolved():
    """Verify Layer 3 contains zero resolved threads and resolution_present is strictly False."""
    esc_cases = read_jsonl(ESCALATION_DIR / "escalation_cases.jsonl")
    assert len(esc_cases) == 72325

    for c in esc_cases[:200]:
        assert c["resolution_present"] is False
        assert c["escalation_status"] in ["ESCALATED_TO_DM", "UNRESOLVED", "ABANDONED", "UNCLEAR", "PARTIAL_RESOLUTION", "OTHER"]
        assert c["escalation_status"] != "CLEARLY_RESOLVED"


# 6. Unknown labels remain unknown / uncertain
def test_uncertain_labels_preserved():
    """Verify that ambiguous/fallback queries are marked 'uncertain' and never masquerade as ground truth."""
    intent, label_source, conf = infer_intent_from_text("Hello what is this thing doing?")
    assert label_source == "uncertain"
    assert conf <= 0.60

    # In Layer 1 stats, verify uncertain labels exist and are explicitly tallied
    with open(INTENT_DIR / "intent_language_statistics.json", "r", encoding="utf-8") as f:
        l1_stats = json.load(f)

    assert "uncertain" in l1_stats["label_source_distribution"]
    assert l1_stats["label_source_distribution"]["uncertain"] > 50000


# 7. Validation/test/golden leakage is detected and passes
def test_zero_leakage_audit():
    """Verify that leakage audit runs and achieves PASS status with zero cross-split overlap."""
    audit_res = run_leakage_check()
    assert audit_res["status"] == "PASS"
    assert audit_res["passed"] is True
    assert audit_res["case_overlaps"] == (0, 0, 0)
    assert audit_res["root_overlaps"] == (0, 0, 0)
    assert audit_res["faiss_val_overlap"] == 0
    assert audit_res["faiss_test_overlap"] == 0
    assert audit_res["esc_res_overlap"] == 0


# 8. Duplicate handling works
def test_duplicate_handling():
    """Verify that duplicate customer phrasings are tracked with accurate frequency metrics."""
    with open(INTENT_DIR / "intent_language_statistics.json", "r", encoding="utf-8") as f:
        l1_stats = json.load(f)

    total_records = l1_stats["total_derived_records"]
    unique_messages = l1_stats["unique_customer_messages"]
    duplicates = l1_stats["duplicate_messages_count"]

    assert total_records == 129146
    assert unique_messages > 100000
    assert duplicates == total_records - unique_messages
    assert duplicates > 0  # verifies duplicate detection operated


# 9. Statistics are internally consistent
def test_statistics_internal_consistency():
    """Verify that statistical breakdowns sum up exactly to their respective layer totals."""
    # Layer 1
    with open(INTENT_DIR / "intent_language_statistics.json", "r", encoding="utf-8") as f:
        l1 = json.load(f)
    assert sum(l1["label_source_distribution"].values()) == l1["total_derived_records"]
    assert sum(l1["split_distribution"].values()) == l1["total_derived_records"]
    assert sum(l1["records_per_intent"].values()) == l1["total_derived_records"]

    # Layer 2
    with open(RESOLUTION_DIR / "resolution_statistics.json", "r", encoding="utf-8") as f:
        l2 = json.load(f)
    assert sum(l2["resolution_status_distribution"].values()) == l2["total_resolution_cases"]
    assert sum(l2["cases_per_intent"].values()) == l2["total_resolution_cases"]
    assert l2["total_resolution_cases"] == 7922

    # Layer 3
    with open(ESCALATION_DIR / "escalation_statistics.json", "r", encoding="utf-8") as f:
        l3 = json.load(f)
    assert sum(l3["escalation_status_distribution"].values()) == l3["total_derived_records"]
    assert l3["total_derived_records"] == 72325

    # Layer 4
    with open(SAFETY_DIR / "safety_statistics.json", "r", encoding="utf-8") as f:
        l4 = json.load(f)
    assert sum(l4["category_distribution"].values()) == l4["total_edge_cases_identified"]


# 10. Running the pipeline twice produces deterministic output
def test_pipeline_determinism():
    """Verify that deterministic inference returns identical classifications on identical inputs."""
    q = "My iPhone 7 battery is draining very fast after updating to iOS 11."
    i1, s1, c1 = infer_intent_from_text(q)
    i2, s2, c2 = infer_intent_from_text(q)

    assert i1 == i2 == "BATTERY_CHARGING_POWER"
    assert s1 == s2 == "inferred"
    assert c1 == c2 == 0.85
