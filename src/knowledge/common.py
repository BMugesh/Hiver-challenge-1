"""
SupportDNA Knowledge Layer — Common Utilities & Paths
=====================================================
Centralized paths, data loaders, and serialization utilities for all 4 knowledge layers.
"""

import sys
import os
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

# Reconfigure stdout for utf-8 on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = PROCESSED_DIR / "splits"
RETRIEVAL_DIR = PROCESSED_DIR / "retrieval"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Knowledge layer subdirectories
INTENT_DIR = KNOWLEDGE_DIR / "intent_language"
RESOLUTION_DIR = KNOWLEDGE_DIR / "resolution"
ESCALATION_DIR = KNOWLEDGE_DIR / "escalation"
SAFETY_DIR = KNOWLEDGE_DIR / "safety"

# Ensure directories exist
for d in [INTENT_DIR, RESOLUTION_DIR, ESCALATION_DIR, SAFETY_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def load_all_threads() -> List[Dict[str, Any]]:
    """Load all 80,247 reconstructed conversation threads."""
    path = PROCESSED_DIR / "apple_support_threads.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing reconstructed threads: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_resolved_threads() -> List[Dict[str, Any]]:
    """Load the 7,922 curated resolved cases."""
    path = PROCESSED_DIR / "apple_support_resolved_threads.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing resolved threads: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_excluded_threads() -> List[Dict[str, Any]]:
    """Load the 72,325 excluded/unclear/escalated threads."""
    path = PROCESSED_DIR / "apple_support_excluded_threads.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing excluded threads: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_splits_map() -> Dict[str, Dict[str, str]]:
    """
    Load train/val/test splits and construct lookup dictionaries:
    - case_id_to_split: e.g. "CASE_000001" -> "train"
    - root_id_to_split: e.g. 747 -> "train"
    """
    case_to_split = {}
    root_to_split = {}

    train_path = SPLITS_DIR / "train.csv"
    val_path = SPLITS_DIR / "validation.csv"
    test_path = SPLITS_DIR / "test.csv"

    for path, split_name in [(train_path, "train"), (val_path, "validation"), (test_path, "test")]:
        if path.exists():
            df = pd.read_csv(path)
            if "case_id" in df.columns:
                for cid in df["case_id"]:
                    case_to_split[str(cid)] = split_name
            if "thread_root_id" in df.columns:
                for rid in df["thread_root_id"]:
                    root_to_split[int(rid)] = split_name

    return {
        "case_to_split": case_to_split,
        "root_to_split": root_to_split
    }


def load_intent_taxonomy() -> List[Dict[str, Any]]:
    """Load 11-intent taxonomy schema."""
    path = PROCESSED_DIR / "apple_support_intent_taxonomy.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing intent taxonomy: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_curated_intent_cases() -> pd.DataFrame:
    """Load labeled historical cases."""
    path = PROCESSED_DIR / "apple_support_intent_cases.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing intent cases CSV: {path}")
    return pd.read_csv(path)


def write_jsonl(records: List[Dict[str, Any]], filepath: Path) -> None:
    """Write list of dictionaries to JSONL format."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    """Read JSONL file into list of dictionaries."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_json(data: Any, filepath: Path) -> None:
    """Write object to formatted JSON."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def clean_customer_text(text: str) -> str:
    """Normalize twitter handles, excessive whitespace, and basic formatting."""
    cleaned = re.sub(r'@\w+', '', text)
    cleaned = re.sub(r'https?://\S+', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned
