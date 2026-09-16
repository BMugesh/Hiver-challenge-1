"""
Unit Tests for SupportDNA Golden Evaluation Set (N=200)
======================================================
Validates that the frozen Golden Evaluation Set adheres to schema rules,
contains no data leakage against train/val/test/FAISS corpora, covers
all 11 intents and 6 action classes, and preserves CSV/JSON consistency.
"""

import json
import sys
from pathlib import Path
import pytest
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"

from scripts.validate_golden_set import (
    validate_schema,
    validate_uniqueness,
    validate_taxonomies,
    validate_zero_leakage,
    validate_csv_parity,
    TAXONOMY_INTENTS,
    VALID_ACTIONS,
    VALID_EVIDENCE_EXPECTATIONS
)


@pytest.fixture(scope="module")
def golden_data():
    json_path = EVALUATION_DIR / "golden_set.json"
    csv_path = EVALUATION_DIR / "golden_set.csv"
    schema_path = EVALUATION_DIR / "golden_set_schema.json"

    assert json_path.exists(), f"Missing {json_path}"
    assert csv_path.exists(), f"Missing {csv_path}"
    assert schema_path.exists(), f"Missing {schema_path}"

    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    csv_df = pd.read_csv(csv_path)

    return records, schema, csv_df


def test_golden_set_size(golden_data):
    """Verify Golden Set contains exactly 200 records (allowed 150-250)."""
    records, _, _ = golden_data
    assert len(records) == 200


def test_json_schema_compliance(golden_data):
    """Verify 100% conformance against golden_set_schema.json draft-07 schema."""
    records, schema, _ = golden_data
    errors = validate_schema(records, schema)
    assert len(errors) == 0, f"Schema validation failed: {errors}"


def test_identifier_and_text_uniqueness(golden_data):
    """Verify zero duplicate golden_id, source_id, or customer_message."""
    records, _, _ = golden_data
    errors = validate_uniqueness(records)
    assert len(errors) == 0, f"Uniqueness violation: {errors}"


def test_taxonomy_and_action_coverage(golden_data):
    """Verify coverage across all 11 intents, 6 actions, and 6 evidence profiles."""
    records, _, _ = golden_data
    errors = validate_taxonomies(records)
    assert len(errors) == 0, f"Taxonomy coverage violation: {errors}"

    intents = set(r["ground_truth_intent"] for r in records)
    actions = set(r["ground_truth_action"] for r in records)
    evidences = set(r["evidence_expectation"] for r in records)

    assert intents == TAXONOMY_INTENTS
    assert actions == VALID_ACTIONS
    assert evidences == VALID_EVIDENCE_EXPECTATIONS


def test_zero_leakage_against_training_and_splits(golden_data):
    """Verify 0 overlap with train/val/test splits and FAISS index metadata."""
    records, _, _ = golden_data
    errors = validate_zero_leakage(records)
    assert len(errors) == 0, f"Data leakage detected: {errors}"


def test_csv_json_synchronization(golden_data):
    """Verify perfect 1:1 synchronization between JSON and CSV exports."""
    records, _, csv_df = golden_data
    errors = validate_csv_parity(records, csv_df)
    assert len(errors) == 0, f"CSV/JSON synchronization failure: {errors}"


def test_expected_reply_requirements_structure(golden_data):
    """Verify expected reply requirements are criteria lists, not exact reference answers."""
    records, _, _ = golden_data
    for r in records:
        reqs = r["expected_reply_requirements"]
        assert isinstance(reqs, list)
        assert len(reqs) >= 2, f"{r['golden_id']}: Expected at least 2 reply requirements"
        for req in reqs:
            assert isinstance(req, str)
            assert len(req) >= 10, f"{r['golden_id']}: Requirement text too short"


def test_human_reasoning_completeness(golden_data):
    """Verify human reasoning field is present, substantive, and non-empty for every record."""
    records, _, _ = golden_data
    for r in records:
        reasoning = r["human_reasoning"]
        assert isinstance(reasoning, str)
        assert len(reasoning.strip()) >= 15, f"{r['golden_id']}: Reasoning too short ({reasoning})"
