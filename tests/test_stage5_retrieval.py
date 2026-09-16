"""
Unit Tests for Stage 5: Semantic Retrieval & Evidence Packaging
==============================================================
Verifies:
1. FAISS index construction, embedding dimensions (384-d), and unit normalization.
2. Cosine similarity ranking and top-K evidence retrieval.
3. Zero-leakage verification (no test/validation cases in retrieval index).
4. Evidence package format and required schema fields for Stage 6 consumption.
5. TF-IDF lexical retrieval baseline functionality.
6. Top-K ranking stability and score bounds [0.0, 1.0].
"""

import os
import sys
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure offline / CPU environment
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.stage5_retrieval import (
    load_data,
    verify_split_leakage,
    build_case_representations,
    init_embedding_model,
    build_or_load_faiss_index,
    SemanticRetriever,
    TFIDFRetrievalBaseline,
    retrieve_similar_cases,
    EMBEDDING_DIM
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RETRIEVAL_DIR = DATA_DIR / "retrieval"


@pytest.fixture(scope="module")
def setup_data():
    """Load splits, threads, and metadata once for test suite."""
    train_df, val_df, test_df, resolved_threads, taxonomy = load_data()
    threads_by_case = {t["case_id"]: t for t in resolved_threads}
    train_cases = build_case_representations(train_df, threads_by_case)
    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_cases": train_cases,
        "taxonomy": taxonomy
    }


class TestStage5SemanticRetrieval:

    def test_split_leakage_strict_check(self, setup_data):
        """Verify zero overlap in case_id, thread_root_id, and texts across splits."""
        train_df = setup_data["train_df"]
        val_df = setup_data["val_df"]
        test_df = setup_data["test_df"]

        leakage = verify_split_leakage(train_df, val_df, test_df)
        assert leakage["passed"] is True
        assert leakage["case_overlap"] == 0
        assert leakage["root_overlap"] == 0
        assert leakage["train_size"] == 5545
        assert leakage["val_size"] == 1188
        assert leakage["test_size"] == 1189

    def test_embedding_dimensions_and_norm(self, setup_data):
        """Verify embeddings are 384-dimensional and strictly L2-normalized."""
        model = init_embedding_model()
        test_texts = [
            "My iPhone battery dies so fast on iOS 11",
            "How do I reset my Apple ID password?",
            "WiFi disconnects constantly when locked"
        ]
        embs = model.encode(test_texts, convert_to_numpy=True, normalize_embeddings=True)
        assert embs.shape == (3, EMBEDDING_DIM)
        norms = np.linalg.norm(embs, axis=1)
        np.testing.assert_allclose(norms, np.ones(3), rtol=1e-5)

    def test_faiss_index_integrity(self, setup_data):
        """Verify FAISS IndexFlatIP count, dimensionality, and non-empty vector matrix."""
        model = init_embedding_model()
        train_cases = setup_data["train_cases"]
        index, embeddings, metadata = build_or_load_faiss_index(model, train_cases)

        assert index.ntotal == len(train_cases)
        assert index.d == EMBEDDING_DIM
        assert embeddings.shape == (len(train_cases), EMBEDDING_DIM)
        assert len(metadata) == len(train_cases)

    def test_semantic_retrieval_evidence_packaging(self, setup_data):
        """Verify retrieved historical cases contain all structured evidence fields required for Stage 6."""
        query = "My battery drains extremely fast after installing the new update"
        results = retrieve_similar_cases(query, top_k=3)

        assert len(results) == 3
        for r in results:
            assert "rank" in r
            assert "case_id" in r
            assert "thread_root_id" in r
            assert "intent_id" in r
            assert "similarity" in r
            assert "customer_problem" in r
            assert "support_response" in r
            assert "resolution_status" in r
            assert "outcome" in r
            assert 0.0 <= r["similarity"] <= 1.0
            assert r["resolution_status"] in ["CLEARLY_RESOLVED", "PARTIALLY_RESOLVED"]
            assert len(r["customer_problem"]) > 0

        # Verify ranking order (descending similarity)
        assert results[0]["similarity"] >= results[1]["similarity"] >= results[2]["similarity"]

    def test_tfidf_baseline_retrieval(self, setup_data):
        """Verify lexical TF-IDF baseline retrieves ranked cases."""
        train_cases = setup_data["train_cases"]
        tfidf = TFIDFRetrievalBaseline(train_cases)
        results = tfidf.retrieve("autocorrect letter I bug", top_k=3)

        assert len(results) == 3
        for r in results:
            assert "rank" in r
            assert "case_id" in r
            assert "similarity" in r
            assert 0.0 <= r["similarity"] <= 1.0

    def test_domain_semantic_alignment(self, setup_data):
        """Verify battery and WiFi queries retrieve matching domain intents with high similarity."""
        battery_results = retrieve_similar_cases("iPhone battery percentage dropping from 80% to 10% in minutes", top_k=1)
        assert len(battery_results) == 1
        assert battery_results[0]["intent_id"] in ["BATTERY_CHARGING_POWER", "OS_UPDATE_SYSTEM_PERFORMANCE"]
        assert battery_results[0]["similarity"] >= 0.60

        wifi_results = retrieve_similar_cases("Unable to connect to home WiFi network or Bluetooth", top_k=1)
        assert len(wifi_results) == 1
        assert wifi_results[0]["intent_id"] in ["CONNECTIVITY_WIFI_BLUETOOTH", "GENERAL_DEVICE_INQUIRY"]
        assert wifi_results[0]["similarity"] >= 0.60
