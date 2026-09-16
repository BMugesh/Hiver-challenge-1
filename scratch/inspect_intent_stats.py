"""
Deep inspection of the 495 errors to design rigorous categorization and statistics.
"""
import os
import sys
import json
from pathlib import Path
from collections import Counter
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

def main():
    train_df, val_df, test_df, resolved_threads, taxonomy = load_data()
    threads_by_case = {t["case_id"]: t for t in resolved_threads}
    train_cases = build_case_representations(train_df, threads_by_case)
    test_cases = build_case_representations(test_df, threads_by_case)

    # Intent taxonomy
    intents = [item["intent_id"] for item in taxonomy]

    # Training counts per intent
    train_intent_counts = Counter(c["intent_id"] for c in train_cases)
    test_intent_counts = Counter(c["intent_id"] for c in test_cases)

    model = init_embedding_model()
    index, embeddings, metadata = build_or_load_faiss_index(model, train_cases)
    retriever = SemanticRetriever(model, index, metadata)

    query_texts = [q["customer_problem"] for q in test_cases]
    top10_results = retriever.retrieve_batch(query_texts, top_k=10)

    # Intent level performance
    intent_stats = {intent: {"train_count": train_intent_counts[intent], "test_count": test_intent_counts[intent], "correct": 0, "wrong": 0} for intent in intents}

    errors = []

    for idx, (query_case, retrieved, expected_intent) in enumerate(zip(test_cases, top10_results, [q["intent_id"] for q in test_cases])):
        top1 = retrieved[0]
        top1_intent = top1["intent_id"]
        top1_sim = top1["similarity"]

        if top1_intent == expected_intent:
            intent_stats[expected_intent]["correct"] += 1
        else:
            intent_stats[expected_intent]["wrong"] += 1

            match_rank = None
            match_case_id = None
            for r in retrieved:
                if r["intent_id"] == expected_intent:
                    match_rank = r["rank"]
                    match_case_id = r["case_id"]
                    break

            top3_intents = [r["intent_id"] for r in retrieved[:3]]
            top5_intents = [r["intent_id"] for r in retrieved[:5]]
            top3_cases = [r["case_id"] for r in retrieved[:3]]
            top5_cases = [r["case_id"] for r in retrieved[:5]]

            errors.append({
                "test_idx": idx,
                "case_id": query_case["case_id"],
                "thread_root_id": query_case["thread_root_id"],
                "query": query_case["customer_problem"],
                "expected_intent": expected_intent,
                "predicted_top1_intent": top1_intent,
                "top1_similarity": top1_sim,
                "correct_case_id": match_case_id if match_rank is not None else "N/A",
                "correct_case_rank": match_rank if match_rank is not None else "Missing (>10)",
                "correct_case_rank_num": match_rank,
                "top3_intents": top3_intents,
                "top5_intents": top5_intents,
                "top3_case_ids": top3_cases,
                "top5_case_ids": top5_cases,
                "retrieved_top1_problem": top1["customer_problem"],
                "retrieved_top1_resolution": top1["support_response"]
            })

    print("=== INTENT LEVEL PERFORMANCE ===")
    for intent, s in sorted(intent_stats.items(), key=lambda x: x[1]["correct"] / max(1, x[1]["test_count"])):
        total = s["test_count"]
        corr = s["correct"]
        wrg = s["wrong"]
        recall1 = (corr / total * 100) if total > 0 else 0
        error_rate = (wrg / total * 100) if total > 0 else 0
        print(f"{intent:<35} | Train: {s['train_count']:<5} | Test: {total:<4} | Corr: {corr:<4} | Wrg: {wrg:<4} | Recall@1: {recall1:>6.2f}% | ErrRate: {error_rate:>6.2f}%")

    # Let's inspect text characteristics of the 495 errors to refine categorization
    # Look at lengths, keywords, semantic relationships
    # Let's save a sample of errors for inspection
    df_err = pd.DataFrame(errors)
    df_err.to_csv("scratch/all_495_errors.csv", index=False)
    print(f"\nSaved {len(df_err)} errors to scratch/all_495_errors.csv")

if __name__ == "__main__":
    main()
