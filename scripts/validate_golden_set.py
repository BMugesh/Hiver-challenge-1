"""
SupportDNA Golden Evaluation Set — Quality & Integrity Validator
================================================================
Performs comprehensive quality, schema, consistency, and zero-leakage
validations on data/evaluation/golden_set.json and data/evaluation/golden_set.csv.

Validation Checks:
1. JSON Schema conformance against data/evaluation/golden_set_schema.json
2. Field completeness (all 11 required fields populated for every record)
3. Uniqueness (100% unique golden_id, source_id, customer_message)
4. Intent taxonomy validity (all 11 verified business intents present)
5. Action labels validity (all 6 allowed operational actions present)
6. Evidence expectation validity (all 6 profiles present)
7. Edge-case category validity (all tags belong to allowed set)
8. Zero leakage against train.csv, validation.csv, and test.csv (exact texts and roots)
9. Zero leakage against FAISS retrieval index metadata
10. CSV and JSON parity (row counts, field values, list serializations)
"""

import sys
import json
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Set
import pandas as pd
import jsonschema

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EVALUATION_DIR = DATA_DIR / "evaluation"
SPLITS_DIR = DATA_DIR / "processed" / "splits"
RETRIEVAL_DIR = DATA_DIR / "processed" / "retrieval"

GOLDEN_JSON = EVALUATION_DIR / "golden_set.json"
GOLDEN_CSV = EVALUATION_DIR / "golden_set.csv"
GOLDEN_SCHEMA = EVALUATION_DIR / "golden_set_schema.json"

TAXONOMY_INTENTS = {
    "KEYBOARD_TYPING_AUTOCORRECT",
    "BATTERY_CHARGING_POWER",
    "CONNECTIVITY_WIFI_BLUETOOTH",
    "DISPLAY_TOUCH_SCREEN",
    "ACCOUNT_APPLEID_ICLOUD",
    "APP_STORE_PURCHASES_BILLING",
    "APP_CRASH_AND_DOWNLOAD",
    "AUDIO_SOUND_SPEAKER",
    "OS_UPDATE_SYSTEM_PERFORMANCE",
    "HOW_TO_SETTINGS_CONFIGURATION",
    "GENERAL_DEVICE_INQUIRY"
}

VALID_ACTIONS = {
    "ANSWER",
    "GUIDE",
    "CLARIFY",
    "ESCALATE",
    "SAFE_REFUSAL",
    "SAFE_REFUSAL_AND_ESCALATE"
}

VALID_EVIDENCE_EXPECTATIONS = {
    "STRONG_HISTORICAL_EVIDENCE",
    "MODERATE_HISTORICAL_EVIDENCE",
    "WEAK_HISTORICAL_EVIDENCE",
    "INSUFFICIENT_HISTORICAL_EVIDENCE",
    "CONFLICTING_HISTORICAL_EVIDENCE",
    "NO_HISTORICAL_EVIDENCE_REQUIRED"
}

VALID_EDGE_CASES = {
    "CLEAR_INTENT",
    "AMBIGUOUS",
    "SHORT_VAGUE",
    "MULTI_ISSUE",
    "FOLLOW_UP",
    "UNRESOLVED_FOLLOW_UP",
    "STRONG_EVIDENCE",
    "WEAK_EVIDENCE",
    "INSUFFICIENT_EVIDENCE",
    "CONFLICTING_EVIDENCE",
    "ESCALATION_SENSITIVE",
    "SAFETY_SENSITIVE",
    "UNUSUAL_WORDING"
}


def validate_schema(records: List[Dict[str, Any]], schema: Dict[str, Any]) -> List[str]:
    errors = []
    for idx, record in enumerate(records):
        try:
            jsonschema.validate(instance=record, schema=schema["items"])
        except jsonschema.ValidationError as e:
            errors.append(f"Record {idx} ({record.get('golden_id', 'UNKNOWN')}): Schema error - {e.message}")
    return errors


def validate_uniqueness(records: List[Dict[str, Any]]) -> List[str]:
    errors = []
    golden_ids = [r["golden_id"] for r in records]
    source_ids = [r["source_id"] for r in records]
    messages = [r["customer_message"].strip().lower() for r in records]

    if len(golden_ids) != len(set(golden_ids)):
        dups = [item for item, count in pd.Series(golden_ids).value_counts().items() if count > 1]
        errors.append(f"Duplicate golden_id found: {dups}")

    if len(source_ids) != len(set(source_ids)):
        dups = [item for item, count in pd.Series(source_ids).value_counts().items() if count > 1]
        errors.append(f"Duplicate source_id found: {dups}")

    if len(messages) != len(set(messages)):
        dups = [item for item, count in pd.Series(messages).value_counts().items() if count > 1]
        errors.append(f"Duplicate customer_message found: {dups}")

    return errors


def validate_taxonomies(records: List[Dict[str, Any]]) -> List[str]:
    errors = []
    intents = set(r["ground_truth_intent"] for r in records)
    actions = set(r["ground_truth_action"] for r in records)
    evidences = set(r["evidence_expectation"] for r in records)

    missing_intents = TAXONOMY_INTENTS - intents
    if missing_intents:
        errors.append(f"Missing business intents in golden set: {missing_intents}")

    missing_actions = VALID_ACTIONS - actions
    if missing_actions:
        errors.append(f"Missing action types in golden set: {missing_actions}")

    missing_evidences = VALID_EVIDENCE_EXPECTATIONS - evidences
    if missing_evidences:
        errors.append(f"Missing evidence expectations in golden set: {missing_evidences}")

    # Validate individual field values
    for r in records:
        gid = r["golden_id"]
        if r["ground_truth_intent"] not in TAXONOMY_INTENTS:
            errors.append(f"{gid}: Invalid intent '{r['ground_truth_intent']}'")
        if r["ground_truth_action"] not in VALID_ACTIONS:
            errors.append(f"{gid}: Invalid action '{r['ground_truth_action']}'")
        if r["evidence_expectation"] not in VALID_EVIDENCE_EXPECTATIONS:
            errors.append(f"{gid}: Invalid evidence expectation '{r['evidence_expectation']}'")
        for ec in r.get("edge_case_category", []):
            if ec not in VALID_EDGE_CASES:
                errors.append(f"{gid}: Invalid edge case '{ec}'")

    return errors


def validate_zero_leakage(records: List[Dict[str, Any]]) -> List[str]:
    errors = []
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "validation.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")

    with open(RETRIEVAL_DIR / "train_case_metadata.json", "r", encoding="utf-8") as f:
        faiss_meta = json.load(f)

    train_roots = set(train_df["thread_root_id"].dropna().astype(int))
    val_roots = set(val_df["thread_root_id"].dropna().astype(int))
    test_roots = set(test_df["thread_root_id"].dropna().astype(int))
    faiss_roots = set(int(m["thread_root_id"]) for m in faiss_meta)
    all_excluded_roots = train_roots | val_roots | test_roots | faiss_roots

    train_texts = set(train_df["customer_text"].dropna().str.strip().str.lower())
    val_texts = set(val_df["customer_text"].dropna().str.strip().str.lower())
    test_texts = set(test_df["customer_text"].dropna().str.strip().str.lower())
    faiss_texts = set(m["customer_problem"].strip().lower() for m in faiss_meta)

    for r in records:
        gid = r["golden_id"]
        sid = r["source_id"]
        msg = r["customer_message"].strip().lower()

        # Check exact message leakage
        if msg in train_texts:
            errors.append(f"{gid}: Exact message leakage in train.csv!")
        if msg in val_texts:
            errors.append(f"{gid}: Exact message leakage in validation.csv!")
        if msg in test_texts:
            errors.append(f"{gid}: Exact message leakage in test.csv!")
        if msg in faiss_texts:
            errors.append(f"{gid}: Exact message leakage in FAISS index metadata!")

        # Check source_id thread root overlap
        match = re.search(r"\d+", sid)
        if match:
            rid = int(match.group(0))
            if rid in all_excluded_roots:
                errors.append(f"{gid}: Source thread root {rid} ({sid}) exists in excluded train/val/test/FAISS roots!")

    return errors


def validate_csv_parity(records: List[Dict[str, Any]], csv_df: pd.DataFrame) -> List[str]:
    errors = []
    if len(records) != len(csv_df):
        errors.append(f"Row count mismatch: JSON has {len(records)} records, CSV has {len(csv_df)} rows")
        return errors

    for idx, r in enumerate(records):
        row = csv_df.iloc[idx]
        gid = r["golden_id"]
        if row["golden_id"] != gid:
            errors.append(f"Row {idx} golden_id mismatch: JSON={gid}, CSV={row['golden_id']}")
        if row["source_id"] != r["source_id"]:
            errors.append(f"{gid}: source_id mismatch between CSV and JSON")
        if row["customer_message"] != r["customer_message"]:
            errors.append(f"{gid}: customer_message mismatch between CSV and JSON")
        if row["ground_truth_intent"] != r["ground_truth_intent"]:
            errors.append(f"{gid}: intent mismatch between CSV and JSON")
        if row["ground_truth_action"] != r["ground_truth_action"]:
            errors.append(f"{gid}: action mismatch between CSV and JSON")
        if bool(row["escalation_required"]) != bool(r["escalation_required"]):
            errors.append(f"{gid}: escalation_required mismatch between CSV and JSON")

        # Check parsed JSON fields in CSV
        try:
            csv_issues = json.loads(row["ground_truth_issues"])
            if csv_issues != r["ground_truth_issues"]:
                errors.append(f"{gid}: ground_truth_issues list mismatch")
        except Exception as e:
            errors.append(f"{gid}: CSV ground_truth_issues JSON parse failure: {e}")

        try:
            csv_reqs = json.loads(row["expected_reply_requirements"])
            if csv_reqs != r["expected_reply_requirements"]:
                errors.append(f"{gid}: expected_reply_requirements list mismatch")
        except Exception as e:
            errors.append(f"{gid}: CSV expected_reply_requirements JSON parse failure: {e}")

        try:
            csv_edge = json.loads(row["edge_case_category"])
            if csv_edge != r["edge_case_category"]:
                errors.append(f"{gid}: edge_case_category list mismatch")
        except Exception as e:
            errors.append(f"{gid}: CSV edge_case_category JSON parse failure: {e}")

    return errors


def run_all_validations() -> Dict[str, Any]:
    print("=" * 70)
    print("RUNNING GOLDEN EVALUATION SET INTEGRITY AUDIT")
    print("=" * 70)

    # 1. Load Files
    if not GOLDEN_JSON.exists():
        raise FileNotFoundError(f"Missing {GOLDEN_JSON}")
    if not GOLDEN_CSV.exists():
        raise FileNotFoundError(f"Missing {GOLDEN_CSV}")
    if not GOLDEN_SCHEMA.exists():
        raise FileNotFoundError(f"Missing {GOLDEN_SCHEMA}")

    with open(GOLDEN_JSON, "r", encoding="utf-8") as f:
        records = json.load(f)
    with open(GOLDEN_SCHEMA, "r", encoding="utf-8") as f:
        schema = json.load(f)
    csv_df = pd.read_csv(GOLDEN_CSV)

    print(f"Loaded {len(records)} records from {GOLDEN_JSON.name}")
    print(f"Loaded {len(csv_df)} rows from {GOLDEN_CSV.name}")

    results = {}

    # Check 1: Record Count
    count_passed = len(records) == 200
    results["record_count_valid"] = count_passed
    print(f"[{'PASS' if count_passed else 'FAIL'}] Exactly 200 Records: count={len(records)}")

    # Check 2: Schema Conformance
    schema_errors = validate_schema(records, schema)
    results["schema_conformance_valid"] = len(schema_errors) == 0
    print(f"[{'PASS' if len(schema_errors) == 0 else 'FAIL'}] JSON Schema Conformance: {len(schema_errors)} errors")
    for err in schema_errors[:5]:
        print(f"       {err}")

    # Check 3: Uniqueness
    unique_errors = validate_uniqueness(records)
    results["uniqueness_valid"] = len(unique_errors) == 0
    print(f"[{'PASS' if len(unique_errors) == 0 else 'FAIL'}] IDs & Messages Uniqueness: {len(unique_errors)} errors")
    for err in unique_errors[:5]:
        print(f"       {err}")

    # Check 4: Taxonomies & Label Coverage
    tax_errors = validate_taxonomies(records)
    results["taxonomies_valid"] = len(tax_errors) == 0
    print(f"[{'PASS' if len(tax_errors) == 0 else 'FAIL'}] Taxonomies & Label Coverage: {len(tax_errors)} errors")
    for err in tax_errors[:5]:
        print(f"       {err}")

    # Check 5: Zero Data Leakage
    leak_errors = validate_zero_leakage(records)
    results["zero_leakage_valid"] = len(leak_errors) == 0
    print(f"[{'PASS' if len(leak_errors) == 0 else 'FAIL'}] Zero Leakage (Train/Val/Test/FAISS): {len(leak_errors)} errors")
    for err in leak_errors[:5]:
        print(f"       {err}")

    # Check 6: CSV/JSON Parity
    parity_errors = validate_csv_parity(records, csv_df)
    results["csv_parity_valid"] = len(parity_errors) == 0
    print(f"[{'PASS' if len(parity_errors) == 0 else 'FAIL'}] CSV / JSON Parity: {len(parity_errors)} errors")
    for err in parity_errors[:5]:
        print(f"       {err}")

    all_passed = all(results.values())
    results["all_passed"] = all_passed
    print("=" * 70)
    print(f"OVERALL AUDIT RESULT: {'ALL CHECKS PASSED (100% CLEAN)' if all_passed else 'FAILED'}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    res = run_all_validations()
    if not res["all_passed"]:
        sys.exit(1)
    sys.exit(0)
