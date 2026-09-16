"""
Fast tuning script loading cached embeddings.
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
    RETRIEVAL_DIR
)

def build_fast_prototypes(train_cases, taxonomy, embeddings):
    intents = [item["intent_id"] for item in taxonomy]
    taxonomy_by_intent = {item["intent_id"]: item for item in taxonomy}

    cases_by_intent = defaultdict(list)
    for c in train_cases:
        cases_by_intent[c["intent_id"]].append(c)

    # Compute centroids using precomputed embeddings
    train_labels = [c["intent_id"] for c in train_cases]
    intent_centroids = {}
    for intent in intents:
        idxs = [i for i, l in enumerate(train_labels) if l == intent]
        if idxs:
            centroid = np.mean(embeddings[idxs], axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
            intent_centroids[intent] = centroid

    domain_knowledge = {
        "KEYBOARD_TYPING_AUTOCORRECT": {
            "symptoms": ["keyboard lag", "predictive text bug", "letter I glitch", "autocorrect symbol box", "typing slow", "keyboard unresponsive"],
            "positive_terms": ["keyboard", "autocorrect", "letter i", "predictive", "typing", "type", "keypad", "auto correct", "capital i", "question box", "i bug", "symbol box"],
            "distinguishing_terms": ["letter i", "autocorrect", "keyboard", "predictive text", "typing", "capital i"],
            "negative_terms": ["wifi", "battery", "charge", "screen brightness", "touch screen", "sound", "volume", "itunes"],
            "confusable_intents": ["OS_UPDATE_SYSTEM_PERFORMANCE", "GENERAL_DEVICE_INQUIRY"]
        },
        "BATTERY_CHARGING_POWER": {
            "symptoms": ["rapid battery drain", "overheating", "slow charging", "phone dying fast", "battery percentage drop", "won't charge"],
            "positive_terms": ["battery", "drain", "draining", "charge", "charging", "charger", "overheating", "hot", "dying fast", "battery life", "percentage drop", "power down", "shut off"],
            "distinguishing_terms": ["battery", "drain", "charge", "charger", "overheat", "battery life"],
            "negative_terms": ["wifi", "keyboard", "autocorrect", "app store", "sound", "speaker", "bluetooth"],
            "confusable_intents": ["OS_UPDATE_SYSTEM_PERFORMANCE", "GENERAL_DEVICE_INQUIRY"]
        },
        "CONNECTIVITY_WIFI_BLUETOOTH": {
            "symptoms": ["wifi disconnects", "bluetooth pairing failure", "no cellular data", "lte dropping", "airdrop not working", "no service"],
            "positive_terms": ["wifi", "wi-fi", "bluetooth", "cellular", "data", "lte", "airdrop", "hotspot", "pairing", "connect", "connection", "no service", "carrier", "network", "signal"],
            "distinguishing_terms": ["wifi", "wi-fi", "bluetooth", "cellular", "airdrop", "hotspot", "pairing"],
            "negative_terms": ["battery", "keyboard", "autocorrect", "screen brightness", "touch id"],
            "confusable_intents": ["GENERAL_DEVICE_INQUIRY", "OS_UPDATE_SYSTEM_PERFORMANCE"]
        },
        "DISPLAY_TOUCH_SCREEN": {
            "symptoms": ["touchscreen unresponsive", "screen freezing", "black screen", "auto brightness bug", "screen flickering", "touch id fail", "face id fail"],
            "positive_terms": ["screen", "touch", "touchscreen", "display", "unresponsive", "frozen screen", "freeze", "freezing", "black screen", "auto brightness", "brightness", "flicker", "3d touch", "touch id", "face id", "lock screen"],
            "distinguishing_terms": ["screen", "touchscreen", "display", "auto brightness", "3d touch", "touch id", "face id"],
            "negative_terms": ["battery", "keyboard", "wifi", "sound", "speaker", "itunes", "app store"],
            "confusable_intents": ["GENERAL_DEVICE_INQUIRY", "OS_UPDATE_SYSTEM_PERFORMANCE"]
        },
        "ACCOUNT_APPLEID_ICLOUD": {
            "symptoms": ["apple id disabled", "forgot password", "two factor verification", "icloud storage full", "activation lock", "cant log in"],
            "positive_terms": ["apple id", "icloud", "password", "appleid", "account", "login", "log in", "disabled", "activation lock", "two-factor", "2fa", "verification code", "storage full", "icloud backup"],
            "distinguishing_terms": ["apple id", "icloud", "password", "activation lock", "two-factor", "2fa", "appleid"],
            "negative_terms": ["battery", "screen", "keyboard", "wifi", "sound", "volume"],
            "confusable_intents": ["APP_STORE_PURCHASES_BILLING", "GENERAL_DEVICE_INQUIRY"]
        },
        "APP_STORE_PURCHASES_BILLING": {
            "symptoms": ["accidental purchase", "refund request", "subscription charged", "apple music billing", "itunes receipt", "card declined"],
            "positive_terms": ["app store", "itunes", "apple music", "purchase", "billing", "refund", "subscription", "charged", "receipt", "credit card", "payment", "buy", "renew", "in-app", "order"],
            "distinguishing_terms": ["app store", "itunes", "apple music", "refund", "billing", "subscription", "charged"],
            "negative_terms": ["battery", "screen", "keyboard", "touchscreen", "wifi", "bluetooth"],
            "confusable_intents": ["APP_CRASH_AND_DOWNLOAD", "ACCOUNT_APPLEID_ICLOUD", "GENERAL_DEVICE_INQUIRY"]
        },
        "APP_CRASH_AND_DOWNLOAD": {
            "symptoms": ["app crashing on launch", "app won't download", "download spinning circle", "can't install app", "apps keep closing"],
            "positive_terms": ["app crash", "crashing", "crashes", "crash", "wont download", "can't download", "spinning", "cant install", "apps close", "app freezes", "app update fail"],
            "distinguishing_terms": ["crashing", "crashes", "crash", "cant download", "wont download", "spinning icon", "apps close"],
            "negative_terms": ["battery", "screen brightness", "keyboard", "autocorrect", "billing", "refund"],
            "confusable_intents": ["APP_STORE_PURCHASES_BILLING", "OS_UPDATE_SYSTEM_PERFORMANCE"]
        },
        "AUDIO_SOUND_SPEAKER": {
            "symptoms": ["no sound from speaker", "microphone not working", "alarm too quiet", "headphone crackling", "call volume low", "airpods audio drop"],
            "positive_terms": ["sound", "speaker", "audio", "volume", "mic", "microphone", "earpiece", "headphones", "airpods", "quiet alarm", "crackling", "ringer", "hear"],
            "distinguishing_terms": ["sound", "speaker", "audio", "volume", "microphone", "mic", "earpiece", "headphones"],
            "negative_terms": ["battery", "keyboard", "screen", "wifi", "itunes", "app store"],
            "confusable_intents": ["GENERAL_DEVICE_INQUIRY", "OS_UPDATE_SYSTEM_PERFORMANCE"]
        },
        "OS_UPDATE_SYSTEM_PERFORMANCE": {
            "symptoms": ["ios update failed", "stuck on update bar", "restarting boot loop after update", "general system lag", "phone bricked during update"],
            "positive_terms": ["ios 11", "ios update", "update failed", "installing update", "stuck on apple logo", "boot loop", "restarting", "laggy", "system lag", "update bricked", "11.0.3", "11.1", "11.2", "os update"],
            "distinguishing_terms": ["ios update", "update failed", "boot loop", "stuck on apple logo", "update bar"],
            "negative_terms": ["battery", "keyboard", "autocorrect", "screen", "touch", "wifi", "speaker", "sound", "apple music"],
            "confusable_intents": ["GENERAL_DEVICE_INQUIRY", "HOW_TO_SETTINGS_CONFIGURATION"]
        },
        "HOW_TO_SETTINGS_CONFIGURATION": {
            "symptoms": ["how to change settings", "how to delete app", "how to disable feature", "where is the setting", "customize control center"],
            "positive_terms": ["how to", "how do i", "how can i", "settings", "disable", "turn off", "turn on", "customize", "configure", "uninstall", "delete app", "change wallpaper"],
            "distinguishing_terms": ["how to", "how do i", "how can i", "settings", "uninstall", "customize"],
            "negative_terms": ["broken", "crashed", "bricked", "refund", "charged", "stolen"],
            "confusable_intents": ["GENERAL_DEVICE_INQUIRY", "OS_UPDATE_SYSTEM_PERFORMANCE"]
        },
        "GENERAL_DEVICE_INQUIRY": {
            "symptoms": ["general hardware inquiry", "specs", "apple support assistance", "store inquiry", "device repair status"],
            "positive_terms": ["iphone", "apple", "device", "apple support", "store", "serial number", "model", "repair", "service", "warranty", "genius bar"],
            "distinguishing_terms": ["apple support", "store", "repair", "warranty", "genius bar"],
            "negative_terms": ["battery", "keyboard", "screen", "wifi", "sound", "volume", "autocorrect"],
            "confusable_intents": ["OS_UPDATE_SYSTEM_PERFORMANCE", "HOW_TO_SETTINGS_CONFIGURATION"]
        }
    }

    prototypes = {}
    for intent in intents:
        tax_info = taxonomy_by_intent.get(intent, {})
        dom_info = domain_knowledge.get(intent, {})
        examples = [c["customer_problem"] for c in cases_by_intent.get(intent, [])[:5]]

        prototypes[intent] = {
            "intent_name": intent,
            "definition": tax_info.get("description", ""),
            "typical_symptoms": dom_info.get("symptoms", []),
            "positive_terms": dom_info.get("positive_terms", []),
            "distinguishing_terms": dom_info.get("distinguishing_terms", []),
            "negative_terms": dom_info.get("negative_terms", []),
            "confusable_intents": dom_info.get("confusable_intents", []),
            "representative_historical_examples": examples,
            "centroid": intent_centroids.get(intent)
        }

    return prototypes

def evaluate_val_set(val_cases, batch_retrieved, val_q_embs, prototypes, w_sem, w_kw, w_proto, w_conf, debias_context=True):
    context_patterns = [r"\bafter update\b", r"\bsince update\b", r"\bafter updating\b", r"\bsince updating\b", r"\bafter installing\b", r"\bfollowing the update\b", r"\bupdated to\b", r"\bnew ios\b", r"\bupdate got my\b", r"\bupdate made my\b"]

    top1_correct = 0
    top3_correct = 0
    rr_list = []

    for idx, (val_case, retrieved, q_emb) in enumerate(zip(val_cases, batch_retrieved, val_q_embs)):
        query_text = val_case["customer_problem"]
        true_intent = val_case["intent_id"]
        q_lower = query_text.lower()
        has_context = any(re.search(p, q_lower) for p in context_patterns)

        # Candidate cases from Top-10
        candidate_cases = retrieved[:10]

        # Detect symptoms present in query
        detected_symptoms = []
        for intent_name, proto in prototypes.items():
            if intent_name in ["OS_UPDATE_SYSTEM_PERFORMANCE", "GENERAL_DEVICE_INQUIRY"]:
                continue
            for dt in proto["distinguishing_terms"]:
                if re.search(r'\b' + re.escape(dt) + r'\b', q_lower):
                    detected_symptoms.append(intent_name)
                    break

        has_specific_symptoms = len(detected_symptoms) > 0

        scored_candidates = []
        for c in candidate_cases:
            c_intent = c["intent_id"]
            c_sim = c["similarity"]
            proto = prototypes.get(c_intent, {})

            sem_score = c_sim

            kw_matches = 0
            for pt in proto.get("positive_terms", []):
                if re.search(r'\b' + re.escape(pt) + r'\b', q_lower):
                    kw_matches += 1
            for dt in proto.get("distinguishing_terms", []):
                if re.search(r'\b' + re.escape(dt) + r'\b', q_lower):
                    kw_matches += 2

            kw_score = min(1.0, kw_matches / 3.0)

            proto_sim = 0.0
            if proto.get("centroid") is not None:
                proto_sim = float(np.dot(q_emb, proto["centroid"]))

            conf_penalty = 0.0
            if debias_context and has_specific_symptoms:
                if c_intent == "OS_UPDATE_SYSTEM_PERFORMANCE" and has_context:
                    conf_penalty += 0.25
                elif c_intent == "GENERAL_DEVICE_INQUIRY":
                    conf_penalty += 0.20

            final_score = (w_sem * sem_score) + (w_kw * kw_score) + (w_proto * proto_sim) - (w_conf * conf_penalty)
            scored_candidates.append((final_score, c))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        new_top1_intent = scored_candidates[0][1]["intent_id"]
        new_top3_intents = [x[1]["intent_id"] for x in scored_candidates[:3]]

        if new_top1_intent == true_intent:
            top1_correct += 1

        if true_intent in new_top3_intents:
            top3_correct += 1

        rr = 0.0
        for r_idx, (_, c) in enumerate(scored_candidates, 1):
            if c["intent_id"] == true_intent:
                rr = 1.0 / r_idx
                break
        rr_list.append(rr)

    n = len(val_cases)
    r1 = top1_correct / n * 100
    r3 = top3_correct / n * 100
    mrr = np.mean(rr_list)
    return r1, r3, mrr

def main():
    train_df, val_df, test_df, resolved_threads, taxonomy = load_data()
    threads_by_case = {t["case_id"]: t for t in resolved_threads}
    train_cases = build_case_representations(train_df, threads_by_case)
    val_cases = build_case_representations(val_df, threads_by_case)

    model = init_embedding_model()
    index, embeddings, metadata = build_or_load_faiss_index(model, train_cases)
    retriever = SemanticRetriever(model, index, metadata)

    prototypes = build_fast_prototypes(train_cases, taxonomy, embeddings)

    val_texts = [q["customer_problem"] for q in val_cases]
    batch_retrieved_val = retriever.retrieve_batch(val_texts, top_k=10)
    val_q_embs = model.encode(val_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

    # Baseline on Validation
    b_r1, b_r3, b_mrr = evaluate_val_set(val_cases, batch_retrieved_val, val_q_embs, prototypes, w_sem=1.0, w_kw=0.0, w_proto=0.0, w_conf=0.0, debias_context=False)
    print(f"BASELINE (Validation N={len(val_cases)}): Recall@1 = {b_r1:.2f}%, Recall@3 = {b_r3:.2f}%, MRR = {b_mrr:.4f}")

    best_config = None
    best_r1 = b_r1
    best_mrr = b_mrr

    for w_kw in [0.10, 0.20, 0.30, 0.40, 0.50]:
        for w_proto in [0.0, 0.05, 0.10, 0.15, 0.20]:
            for w_conf in [0.0, 0.10, 0.20, 0.30, 0.40]:
                r1, r3, mrr = evaluate_val_set(val_cases, batch_retrieved_val, val_q_embs, prototypes, w_sem=1.0, w_kw=w_kw, w_proto=w_proto, w_conf=w_conf, debias_context=True)
                if r1 > best_r1 or (r1 == best_r1 and mrr > best_mrr):
                    best_r1 = r1
                    best_mrr = mrr
                    best_config = (w_kw, w_proto, w_conf)
                    print(f"New Best Val: Recall@1 = {r1:.2f}%, Recall@3 = {r3:.2f}%, MRR = {mrr:.4f} with w_kw={w_kw}, w_proto={w_proto}, w_conf={w_conf}")

    print(f"\nOPTIMAL VALIDATION CONFIG: w_kw={best_config[0]}, w_proto={best_config[1]}, w_conf={best_config[2]}")
    print(f"Validation Gain: Recall@1: {b_r1:.2f}% -> {best_r1:.2f}% (+{best_r1 - b_r1:.2f}%), MRR: {b_mrr:.4f} -> {best_mrr:.4f} (+{best_mrr - b_mrr:.4f})")

if __name__ == "__main__":
    main()
