"""
Scratch script to research training data for intent prototypes and test scoring on validation set.
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

def inspect_training_vocab():
    train_df, val_df, test_df, resolved_threads, taxonomy = load_data()
    threads_by_case = {t["case_id"]: t for t in resolved_threads}
    train_cases = build_case_representations(train_df, threads_by_case)

    print(f"Loaded {len(train_cases)} train cases.")
    intents = [item["intent_id"] for item in taxonomy]
    print(f"Taxonomy intents ({len(intents)}): {intents}")

    # For each intent, find top distinctive words vs other intents using TF-IDF or log odds
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=2000, stop_words="english", ngram_range=(1, 2))
    texts = [c["customer_problem"] for c in train_cases]
    labels = [c["intent_id"] for c in train_cases]
    X = vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names_out()

    print("\nTop distinctive ngrams per intent in train set:")
    for intent in intents:
        idx = [i for i, l in enumerate(labels) if l == intent]
        sub_mean = np.asarray(X[idx].mean(axis=0)).flatten()
        other_idx = [i for i, l in enumerate(labels) if l != intent]
        other_mean = np.asarray(X[other_idx].mean(axis=0)).flatten()
        diff = sub_mean - other_mean
        top_terms = [vocab[k] for k in np.argsort(diff)[::-1][:10]]
        print(f"Intent: {intent:<35} | Top terms: {', '.join(top_terms)}")

if __name__ == "__main__":
    inspect_training_vocab()
