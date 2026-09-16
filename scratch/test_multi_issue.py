"""
Scratch script to test multi-issue detection and Stage 7 safety compatibility on validation set.
"""
import os
import sys
import json
import re
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"c:\Hiver")
sys.path.insert(0, str(PROJECT_ROOT))

# Configure offline / CPU environment
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.stage5_retrieval import (
    load_data,
    build_case_representations,
    init_embedding_model,
    build_or_load_faiss_index,
    SemanticRetriever,
)
from src.stage6_grounded_reply import (
    GroundedReplyGenerator,
    IndependentGroundingVerifier,
    run_stage6,
    clean_twitter_noise
)
from src.stage7_decision import (
    DecisionEngine,
    DecisionPolicy,
    CalibratedIntentClassifier,
    assign_ground_truth_safety_label
)

def detect_multi_issue(query_text, prototypes):
    q_lower = query_text.lower()
    
    # 1. Identify context mentions vs actual symptoms
    context_patterns = [r"\bafter update\b", r"\bsince update\b", r"\bafter updating\b", r"\bsince updating\b", r"\bafter installing\b", r"\bfollowing the update\b", r"\bupdated to\b", r"\bnew ios\b", r"\bupdate got my\b", r"\bupdate made my\b"]
    has_context = any(re.search(p, q_lower) for p in context_patterns)
    
    # 2. Identify symptom domains
    symptom_domains = []
    for intent_name, proto in prototypes.items():
        if intent_name in ["OS_UPDATE_SYSTEM_PERFORMANCE", "GENERAL_DEVICE_INQUIRY"]:
            continue
        # Check distinguishing terms
        matches = [dt for dt in proto["distinguishing_terms"] if re.search(r'\b' + re.escape(dt) + r'\b', q_lower)]
        if matches:
            symptom_domains.append((intent_name, matches))
            
    # Conjunction indicators connecting separate issues
    conjunction_patterns = [r"\band\b", r"\b&\b", r"\balso\b", r"\bplus\b", r"\bas well\b", r"\bboth\b", r"\balong with\b", r"\bnot only\b", r"\btogether with\b"]
    has_conjunction = any(re.search(p, q_lower) for p in conjunction_patterns)
    
    is_multi_issue = (len(symptom_domains) >= 2 and has_conjunction)
    
    decision_hint = "ESCALATE_MULTI_ISSUE" if is_multi_issue else "PROCEED"
    
    return {
        "multi_issue": is_multi_issue,
        "symptom_domains": [sd[0] for sd in symptom_domains],
        "symptom_keywords": {sd[0]: sd[1] for sd in symptom_domains},
        "has_context": has_context,
        "decision_hint": decision_hint,
        "confidence": 0.85 if is_multi_issue else 0.0
    }

def main():
    train_df, val_df, test_df, resolved_threads, taxonomy = load_data()
    threads_by_case = {t["case_id"]: t for t in resolved_threads}
    train_cases = build_case_representations(train_df, threads_by_case)
    val_cases = build_case_representations(val_df, threads_by_case)

    # Let's import build_fast_prototypes
    from scratch.tune_fast_validation import build_fast_prototypes
    embeddings = np.load("data/processed/retrieval/train_case_embeddings.npy")
    prototypes = build_fast_prototypes(train_cases, taxonomy, embeddings)

    val_multi_count = 0
    for vc in val_cases:
        res = detect_multi_issue(vc["customer_problem"], prototypes)
        if res["multi_issue"]:
            val_multi_count += 1

    print(f"Validation Multi-Issue count: {val_multi_count} / {len(val_cases)} ({val_multi_count / len(val_cases) * 100:.2f}%)")

if __name__ == "__main__":
    main()
