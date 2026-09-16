"""
Analyze the 495 errors and develop the comprehensive categorization logic.
"""
import os
import sys
import json
import re
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

def categorize_error(row, train_counts_by_intent):
    query = str(row["query"]).strip()
    query_lower = query.lower()
    exp_intent = row["expected_intent"]
    pred_intent = row["predicted_top1_intent"]
    top1_sim = float(row["top1_similarity"])
    correct_rank = row["correct_case_rank_num"] if pd.notna(row["correct_case_rank_num"]) else None
    
    # Word count (excluding mentions and urls)
    words = [w for w in query_lower.split() if not w.startswith("@") and not w.startswith("http")]
    word_count = len(words)
    
    # Check related intent pairs (semantically adjacent intents)
    semantically_adjacent = {
        ("GENERAL_DEVICE_INQUIRY", "OS_UPDATE_SYSTEM_PERFORMANCE"),
        ("OS_UPDATE_SYSTEM_PERFORMANCE", "GENERAL_DEVICE_INQUIRY"),
        ("HOW_TO_SETTINGS_CONFIGURATION", "GENERAL_DEVICE_INQUIRY"),
        ("GENERAL_DEVICE_INQUIRY", "HOW_TO_SETTINGS_CONFIGURATION"),
        ("HOW_TO_SETTINGS_CONFIGURATION", "OS_UPDATE_SYSTEM_PERFORMANCE"),
        ("OS_UPDATE_SYSTEM_PERFORMANCE", "HOW_TO_SETTINGS_CONFIGURATION"),
        ("APP_CRASH_AND_DOWNLOAD", "APP_STORE_PURCHASES_BILLING"),
        ("APP_STORE_PURCHASES_BILLING", "APP_CRASH_AND_DOWNLOAD"),
        ("APP_CRASH_AND_DOWNLOAD", "OS_UPDATE_SYSTEM_PERFORMANCE"),
        ("OS_UPDATE_SYSTEM_PERFORMANCE", "APP_CRASH_AND_DOWNLOAD"),
        ("DISPLAY_TOUCH_SCREEN", "GENERAL_DEVICE_INQUIRY"),
        ("GENERAL_DEVICE_INQUIRY", "DISPLAY_TOUCH_SCREEN"),
        ("AUDIO_SOUND_SPEAKER", "GENERAL_DEVICE_INQUIRY"),
        ("GENERAL_DEVICE_INQUIRY", "AUDIO_SOUND_SPEAKER"),
        ("CONNECTIVITY_WIFI_BLUETOOTH", "GENERAL_DEVICE_INQUIRY"),
        ("GENERAL_DEVICE_INQUIRY", "CONNECTIVITY_WIFI_BLUETOOTH"),
        ("ACCOUNT_APPLEID_ICLOUD", "APP_STORE_PURCHASES_BILLING"),
        ("APP_STORE_PURCHASES_BILLING", "ACCOUNT_APPLEID_ICLOUD"),
        ("BATTERY_CHARGING_POWER", "OS_UPDATE_SYSTEM_PERFORMANCE"),
        ("OS_UPDATE_SYSTEM_PERFORMANCE", "BATTERY_CHARGING_POWER"),
    }
    
    # Check for Post-update framing (query mentions update/ios 11 but expected intent is a specific hardware/functional domain)
    has_update_mention = any(w in query_lower for w in ["update", "updated", "updating", "ios 11", "ios11", "ios 11.0", "ios 11.1", "ios 11.2", "11.0.3", "11.1.2", "new ios", "latest update", "upgrade"])
    is_post_update_framing = (
        has_update_mention and 
        exp_intent != "OS_UPDATE_SYSTEM_PERFORMANCE" and 
        pred_intent == "OS_UPDATE_SYSTEM_PERFORMANCE"
    )
    
    # Check for Multi-issue query
    multi_issue_indicators = [
        " and ", " & ", "also", "plus", "as well", "both", "along with", "not only"
    ]
    # Check if multiple symptoms are mentioned
    symptom_domains = 0
    if any(w in query_lower for w in ["battery", "drain", "charge", "power", "dying"]): symptom_domains += 1
    if any(w in query_lower for w in ["wifi", "wi-fi", "bluetooth", "cellular", "data", "lte", "signal", "connect"]): symptom_domains += 1
    if any(w in query_lower for w in ["screen", "display", "touch", "freeze", "black screen", "frozen"]): symptom_domains += 1
    if any(w in query_lower for w in ["keyboard", "typing", "autocorrect", "type", "letter i", "predictive"]): symptom_domains += 1
    if any(w in query_lower for w in ["sound", "volume", "speaker", "audio", "mic", "earpiece"]): symptom_domains += 1
    if any(w in query_lower for w in ["app", "crash", "download", "install", "store", "music"]): symptom_domains += 1
    
    is_multi_issue = (symptom_domains >= 2 and any(ind in query_lower for ind in multi_issue_indicators))
    
    # Short or ambiguous query
    is_short_or_ambiguous = (
        word_count <= 5 or
        query_lower in ["help", "need help", "fix this", "why is this happening", "what is wrong", "hello please help", "@applesupport help", "@applesupport please help"] or
        (word_count <= 8 and any(w in query_lower for w in ["broken", "not working", "help me", "fix it", "what happened", "issue with phone", "glitch"]))
    )
    
    # Terminology mismatch / slang / foreign language / atypical phrasing
    is_foreign_or_slang = (
        any(w in query_lower for w in ["hola", "necesito", "batería", "ayuda", "olá", "não", "por favor", "merci", "svp", "bonjour"]) or
        any(w in query_lower for w in ["yo", "wth", "wtf", "smh", "bruh", "y'all", "yall", "af", "sucks", "trash", "screwed", "bricked", "pos", "fuck", "shit"]) or
        top1_sim < 0.52
    )
    
    # Insufficient training examples (rare intent classes with <200 training examples)
    is_sparse_intent = (train_counts_by_intent.get(exp_intent, 0) < 200)
    
    # Decision hierarchy based on root cause:
    if is_short_or_ambiguous:
        return "SHORT_OR_AMBIGUOUS_QUERY", "Query is extremely brief (<6 words or vague phrasing) lacking specific diagnostic keywords."
    elif is_post_update_framing:
        return "POST_UPDATE_OR_CONTEXT_MISMATCH", "Customer framed a specific functional/hardware issue around an iOS update event, biasing dense retrieval toward general OS update cases."
    elif is_multi_issue:
        return "MULTI_ISSUE_QUERY", "Query describes multiple distinct problems simultaneously, causing the embedding to blend disparate semantic centroids."
    elif (exp_intent, pred_intent) in semantically_adjacent:
        return "SEMANTICALLY_SIMILAR_INTENTS", f"Expected intent ({exp_intent}) and predicted intent ({pred_intent}) share dense semantic overlap in customer problem descriptions."
    elif is_foreign_or_slang:
        return "TERMINOLOGY_MISMATCH", "Query contains non-English phrasing, heavy slang, emotional exclamations, or atypical vocabulary divergent from corpus cases."
    elif is_sparse_intent and correct_rank is None:
        return "INSUFFICIENT_TRAINING_EXAMPLES", f"Expected intent ({exp_intent}) has low training volume (<200 cases) resulting in sparse corpus coverage for this specific phrasing."
    elif correct_rank is not None and correct_rank > 1:
        return "WRONG_KNOWLEDGE_REPRESENTATION", "Relevant historical cases exist in the top-10 but the single customer-problem bi-encoder representation ranked a distractor case higher."
    else:
        return "OTHER", "Uncategorized edge case with divergent lexical and semantic features."

def main():
    df_err = pd.read_csv("scratch/all_495_errors.csv")
    print(f"Loaded {len(df_err)} errors.")
    
    train_df = pd.read_csv("data/processed/splits/train.csv")
    train_counts = dict(train_df["intent_id"].value_counts())
    
    categories = []
    descriptions = []
    for _, row in df_err.iterrows():
        cat, desc = categorize_error(row, train_counts)
        categories.append(cat)
        descriptions.append(desc)
        
    df_err["failure_category"] = categories
    df_err["category_explanation"] = descriptions
    
    cat_counts = df_err["failure_category"].value_counts()
    print("\n=== FAILURE CATEGORIES DISTRIBUTION ===")
    for cat, cnt in cat_counts.items():
        pct_err = cnt / len(df_err) * 100
        pct_test = cnt / 1189 * 100
        print(f"{cat:<35} : {cnt:>4} ({pct_err:>6.2f}% of errors | {pct_test:>5.2f}% of test set)")

if __name__ == "__main__":
    main()
