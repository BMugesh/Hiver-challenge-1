"""
Refined categorization and validation for all 495 errors.
"""
import sys
import json
import re
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

def classify_error(row, train_counts_by_intent):
    query = str(row["query"]).strip()
    query_lower = query.lower()
    exp_intent = row["expected_intent"]
    pred_intent = row["predicted_top1_intent"]
    top1_sim = float(row["top1_similarity"])
    correct_rank = row["correct_case_rank_num"] if pd.notna(row["correct_case_rank_num"]) else None
    
    total_tokens = len(query.split())
    words = [w for w in query_lower.split() if not w.startswith("@") and not w.startswith("http")]
    content_word_count = len(words)
    
    # 1. Foreign language / non-English
    non_english_tokens = ["hola", "necesito", "batería", "ayuda", "olá", "não", "por favor", "merci", "svp", "bonjour", "actualización", "dura", "tá", "acontecendo", "guloso", "bienvenida", "multitarea", "gracias"]
    has_foreign = any(w in query_lower for w in non_english_tokens)
    
    # Slang & Colloquialisms
    slang_tokens = ["wtf", "wth", "smh", "bruh", "y'all", "yall", "af", "tf", "sucks", "trash", "bricked", "pos", "pissing", "screwed", "shit", "fuck", "damn"]
    has_slang = any(re.search(r'\b' + re.escape(s) + r'\b', query_lower) for s in slang_tokens)
    
    # Update framing: Query mentions update / iOS 11, but expected intent is a specific functional/hardware issue, and predicted is OS_UPDATE
    update_keywords = ["update", "updated", "updating", "ios 11", "ios11", "ios 11.0", "ios 11.1", "ios 11.2", "11.0.3", "11.1.2", "11.2", "new ios", "latest update", "upgrade"]
    has_update_mention = any(w in query_lower for w in update_keywords)
    is_post_update_framing = (
        has_update_mention and 
        exp_intent != "OS_UPDATE_SYSTEM_PERFORMANCE" and 
        pred_intent == "OS_UPDATE_SYSTEM_PERFORMANCE"
    )
    
    # Short or ambiguous: <= 8 total tokens or very vague content words
    vague_phrases = ["help", "help me", "fix this", "fix it", "broken", "not working", "why is this happening", "what is wrong", "please assist", "what happened"]
    is_short_or_ambiguous = (
        total_tokens <= 8 or 
        content_word_count <= 6 or
        (content_word_count <= 9 and any(v in query_lower for v in vague_phrases))
    )
    
    # Multi-issue query: describes two distinct problem domains connected with conjunctions
    symptom_domains = 0
    if any(w in query_lower for w in ["battery", "drain", "charge", "power", "overheating"]): symptom_domains += 1
    if any(w in query_lower for w in ["wifi", "wi-fi", "bluetooth", "cellular", "data", "lte", "signal", "connect"]): symptom_domains += 1
    if any(w in query_lower for w in ["screen", "display", "touch", "freez", "black screen", "brightness"]): symptom_domains += 1
    if any(w in query_lower for w in ["keyboard", "typing", "autocorrect", "type", "letter i", "predictive"]): symptom_domains += 1
    if any(w in query_lower for w in ["sound", "volume", "speaker", "audio", "mic", "earpiece", "headphone"]): symptom_domains += 1
    if any(w in query_lower for w in ["app", "crash", "download", "install", "itunes", "music", "app store"]): symptom_domains += 1
    if any(w in query_lower for w in ["icloud", "apple id", "password", "lock", "account", "storage full"]): symptom_domains += 1
    
    has_conjunction = any(ind in query_lower for ind in [" and ", " & ", "also", "plus", "as well", "both", "along with", "not only"])
    is_multi_issue = (symptom_domains >= 2 and has_conjunction)
    
    # Semantically similar / adjacent intents
    semantically_similar_pairs = {
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
        ("KEYBOARD_TYPING_AUTOCORRECT", "GENERAL_DEVICE_INQUIRY"),
        ("GENERAL_DEVICE_INQUIRY", "KEYBOARD_TYPING_AUTOCORRECT"),
        ("KEYBOARD_TYPING_AUTOCORRECT", "OS_UPDATE_SYSTEM_PERFORMANCE"),
        ("OS_UPDATE_SYSTEM_PERFORMANCE", "KEYBOARD_TYPING_AUTOCORRECT")
    }
    is_sem_similar = (exp_intent, pred_intent) in semantically_similar_pairs
    
    # Sparse training data (<200 cases in training corpus for this intent)
    train_count = train_counts_by_intent.get(exp_intent, 0)
    is_sparse_data = (train_count < 200)
    
    # Priority classification tree
    if is_short_or_ambiguous:
        category = "SHORT_OR_AMBIGUOUS_QUERY"
        description = "The query is too short or lacks enough diagnostic information, placing its embedding near generic dense clusters."
    elif is_post_update_framing:
        category = "POST_UPDATE_OR_CONTEXT_MISMATCH"
        description = "The customer framed a functional symptom around an iOS update event, causing the dense retriever to falsely match OS update cases."
    elif is_multi_issue:
        category = "MULTI_ISSUE_QUERY"
        description = "The query contains multiple distinct problems and dense vector averaging focuses on the wrong issue."
    elif has_foreign or has_slang or (top1_sim < 0.55 and not is_sem_similar):
        category = "TERMINOLOGY_MISMATCH"
        description = "The query uses non-English words, heavy slang, or phrasing that differs significantly from historical training examples."
    elif is_sparse_data and (correct_rank is None or correct_rank > 5):
        category = "INSUFFICIENT_TRAINING_EXAMPLES"
        description = f"Very few relevant historical training cases exist for '{exp_intent}' ({train_count} cases in training corpus)."
    elif is_sem_similar:
        category = "SEMANTICALLY_SIMILAR_INTENTS"
        description = f"The predicted intent ({pred_intent}) is closely related and semantically adjacent to the expected intent ({exp_intent})."
    elif correct_rank is not None and correct_rank > 1:
        category = "WRONG_KNOWLEDGE_REPRESENTATION"
        description = "Relevant historical information exists in the index, but single customer-turn problem text representation retrieved a distractor case."
    else:
        category = "OTHER"
        description = "Unique edge case failure where symptom description does not align with standard error clusters."
        
    return category, description

def main():
    df_err = pd.read_csv("scratch/all_495_errors.csv")
    train_df = pd.read_csv("data/processed/splits/train.csv")
    train_counts = dict(train_df["intent_id"].value_counts())
    
    categories = []
    descriptions = []
    for _, row in df_err.iterrows():
        cat, desc = classify_error(row, train_counts)
        categories.append(cat)
        descriptions.append(desc)
        
    df_err["category"] = categories
    df_err["description"] = descriptions
    
    print("=== CATEGORY DISTRIBUTION ===")
    summary = []
    for cat, cnt in df_err["category"].value_counts().items():
        pct_err = cnt / len(df_err) * 100
        pct_test = cnt / 1189 * 100
        desc = df_err[df_err["category"] == cat]["description"].iloc[0]
        print(f"{cat:<35} | Count: {cnt:>3} | % of Errors: {pct_err:>6.2f}% | % of Test: {pct_test:>5.2f}%")
        summary.append({
            "category": cat,
            "count": cnt,
            "percentage_of_errors": round(pct_err, 2),
            "percentage_of_test_set": round(pct_test, 2),
            "description": desc
        })
        
    print("\nTotal errors categorized:", len(df_err))

if __name__ == "__main__":
    main()
