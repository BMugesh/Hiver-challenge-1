"""
Scratch script to perform initial extraction and exploration of the 495 errors.
"""
import os
import sys
import json
from pathlib import Path
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
    print("Loading data...")
    train_df, val_df, test_df, resolved_threads, taxonomy = load_data()
    threads_by_case = {t["case_id"]: t for t in resolved_threads}
    train_cases = build_case_representations(train_df, threads_by_case)
    test_cases = build_case_representations(test_df, threads_by_case)

    print(f"Train cases: {len(train_cases)}, Test cases: {len(test_cases)}")

    model = init_embedding_model()
    index, embeddings, metadata = build_or_load_faiss_index(model, train_cases)
    retriever = SemanticRetriever(model, index, metadata)

    query_texts = [q["customer_problem"] for q in test_cases]
    expected_intents = [q["intent_id"] for q in test_cases]

    # Retrieve top 10
    top10_results = retriever.retrieve_batch(query_texts, top_k=10)

    correct_count = 0
    wrong_count = 0
    errors = []
    correct_scores = []
    wrong_scores = []

    for idx, (query_case, retrieved, expected_intent) in enumerate(zip(test_cases, top10_results, expected_intents)):
        top1 = retrieved[0]
        top1_intent = top1["intent_id"]
        top1_sim = top1["similarity"]

        if top1_intent == expected_intent:
            correct_count += 1
            correct_scores.append(top1_sim)
        else:
            wrong_count += 1
            wrong_scores.append(top1_sim)

            # Find rank of first matching intent in top-10
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
                "correct_case_rank_num": match_rank, # None if missing
                "top3_intents": top3_intents,
                "top5_intents": top5_intents,
                "top3_case_ids": top3_cases,
                "top5_case_ids": top5_cases,
                "retrieved_top1_problem": top1["customer_problem"],
                "retrieved_top1_resolution": top1["support_response"]
            })

    print(f"Total: {len(test_cases)}, Correct: {correct_count}, Wrong: {wrong_count}")
    print(f"Recall@1: {correct_count / len(test_cases) * 100:.2f}%")

    # Correct rank analysis among errors
    in_top3 = sum(1 for e in errors if e["correct_case_rank_num"] is not None and e["correct_case_rank_num"] <= 3)
    in_top5 = sum(1 for e in errors if e["correct_case_rank_num"] is not None and e["correct_case_rank_num"] <= 5)
    in_top10 = sum(1 for e in errors if e["correct_case_rank_num"] is not None and e["correct_case_rank_num"] <= 10)
    missing = sum(1 for e in errors if e["correct_case_rank_num"] is None)

    print("\n--- Correct Case Rank Analysis Among 495 Errors ---")
    print(f"In Top 3 (rank 2 or 3): {in_top3} ({in_top3 / len(errors) * 100:.2f}%)")
    print(f"In Top 5 (rank 2 to 5): {in_top5} ({in_top5 / len(errors) * 100:.2f}%)")
    print(f"In Top 10 (rank 2 to 10): {in_top10} ({in_top10 / len(errors) * 100:.2f}%)")
    print(f"Missing from Top 10: {missing} ({missing / len(errors) * 100:.2f}%)")

    # Score stats
    print("\n--- Similarity Score Distribution ---")
    print(f"Correct Top-1: Mean={np.mean(correct_scores):.4f}, Median={np.median(correct_scores):.4f}, Min={np.min(correct_scores):.4f}, Max={np.max(correct_scores):.4f}, P25={np.percentile(correct_scores, 25):.4f}, P75={np.percentile(correct_scores, 75):.4f}")
    print(f"Wrong Top-1:   Mean={np.mean(wrong_scores):.4f}, Median={np.median(wrong_scores):.4f}, Min={np.min(wrong_scores):.4f}, Max={np.max(wrong_scores):.4f}, P25={np.percentile(wrong_scores, 25):.4f}, P75={np.percentile(wrong_scores, 75):.4f}")

    # Top confusions
    from collections import Counter
    confusions = Counter((e["expected_intent"], e["predicted_top1_intent"]) for e in errors)
    print("\n--- Top 10 Confusion Pairs ---")
    for (exp, pred), cnt in confusions.most_common(10):
        print(f"{exp} -> {pred}: {cnt} ({cnt / len(errors) * 100:.2f}%)")

if __name__ == "__main__":
    main()
