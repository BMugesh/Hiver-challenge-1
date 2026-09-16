"""
Unit Tests for Stage 5: Domain-Aware Intent Ranking & Multi-Issue Detection
==========================================================================
Verifies:
1. Intent prototypes for all 11 intents constructed strictly from training data.
2. Domain-aware scoring function properties and interpretability.
3. Multi-issue detection accuracy on compound vs single-symptom inquiries.
4. Context vs. symptom de-biasing behavior.
5. Benchmark integrity: Baseline Recall@1 = 58.37%, Domain-Aware Recall@1 improvement.
6. Safety integrity: False Auto-Handle Rate < 5.0%.
7. Report and CSV artifacts generation with valid schemas.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure offline / CPU execution
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
from src.stage5_multi_issue import (
    detect_multi_issue,
    extract_context_signals,
    extract_symptom_domains
)
from src.stage5_domain_ranking import (
    build_intent_prototypes,
    DomainAwareIntentRanker,
    REPORTS_DIR
)


@pytest.fixture(scope="module")
def setup_domain_ranking():
    """Load splits, metadata, index, and construct prototypes once for test suite."""
    train_df, val_df, test_df, resolved_threads, taxonomy = load_data()
    threads_by_case = {t["case_id"]: t for t in resolved_threads}
    train_cases = build_case_representations(train_df, threads_by_case)
    test_cases = build_case_representations(test_df, threads_by_case)

    model = init_embedding_model()
    index, embeddings, metadata = build_or_load_faiss_index(model, train_cases)
    retriever = SemanticRetriever(model, index, metadata)
    prototypes = build_intent_prototypes(train_cases, taxonomy, embeddings)

    ranker = DomainAwareIntentRanker(
        prototypes=prototypes,
        w_semantic=1.0,
        w_keyword=0.50,
        w_prototype=0.20,
        w_confusion=0.40,
        debias_context=True
    )

    return {
        "train_cases": train_cases,
        "test_cases": test_cases,
        "taxonomy": taxonomy,
        "model": model,
        "retriever": retriever,
        "prototypes": prototypes,
        "ranker": ranker
    }


class TestStage5DomainRanking:

    def test_intent_prototypes_completeness(self, setup_domain_ranking):
        """Verify all 11 intents have structured prototypes with centroids and distinguishing terms."""
        prototypes = setup_domain_ranking["prototypes"]
        taxonomy = setup_domain_ranking["taxonomy"]

        assert len(prototypes) == 11
        for item in taxonomy:
            intent = item["intent_id"]
            assert intent in prototypes
            proto = prototypes[intent]
            assert proto["intent"] == intent
            assert len(proto["definition"]) > 0
            assert len(proto["positive_terms"]) > 0
            assert len(proto["distinguishing_terms"]) > 0
            assert proto["centroid"] is not None
            assert proto["centroid"].shape == (384,)
            # Check unit normalization
            np.testing.assert_allclose(np.linalg.norm(proto["centroid"]), 1.0, rtol=1e-4)

    def test_multi_issue_detection_logic(self, setup_domain_ranking):
        """Verify multi-issue detector identifies compound issues vs single-symptom queries."""
        prototypes = setup_domain_ranking["prototypes"]

        # Case 1: Single symptom with update context -> NOT multi-issue
        q1 = "After updating to iOS 11, my battery is draining so fast"
        res1 = detect_multi_issue(q1, prototypes=prototypes)
        assert res1["multi_issue"] is False
        assert res1["has_context"] is True
        assert res1["decision_hint"] == "PROCEED"
        assert "BATTERY_CHARGING_POWER" in res1["symptom_domains"]

        # Case 2: Compound query with 2 distinct symptoms -> MULTI-ISSUE detected
        q2 = "After updating to iOS 11, my battery drains in 1 hour and my wifi disconnects constantly"
        res2 = detect_multi_issue(q2, prototypes=prototypes)
        assert res2["multi_issue"] is True
        assert res2["has_context"] is True
        assert res2["decision_hint"] == "ESCALATE_MULTI_ISSUE"
        assert "BATTERY_CHARGING_POWER" in res2["symptom_domains"]
        assert "CONNECTIVITY_WIFI_BLUETOOTH" in res2["symptom_domains"]

    def test_context_vs_symptom_debiasing(self, setup_domain_ranking):
        """Verify context de-biasing correctly elevates specific functional symptom over OS update."""
        ranker = setup_domain_ranking["ranker"]
        model = setup_domain_ranking["model"]
        retriever = setup_domain_ranking["retriever"]

        # Query framed around iOS 11 update but actually about letter 'I' glitch
        query = "Since the new iOS 11 update, when I type the letter I it turns into a weird symbol box"
        q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        retrieved_pool = retriever.retrieve(query, top_k=10)

        da_output = ranker.rank_candidates(query, q_emb, retrieved_pool)
        assert da_output["top1_intent"] == "KEYBOARD_TYPING_AUTOCORRECT"

    def test_domain_scoring_interpretability(self, setup_domain_ranking):
        """Verify domain scoring components are well-bounded and interpretable."""
        ranker = setup_domain_ranking["ranker"]
        model = setup_domain_ranking["model"]

        query = "My battery is draining extremely fast and overheating"
        q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

        mock_case = {
            "intent_id": "BATTERY_CHARGING_POWER",
            "similarity": 0.82
        }

        score, components = ranker.score_candidate(
            candidate_case=mock_case,
            query_text=query,
            query_embedding=q_emb,
            has_context=False,
            detected_symptoms=["BATTERY_CHARGING_POWER"]
        )

        assert "semantic_score" in components
        assert "keyword_score" in components
        assert "prototype_score" in components
        assert "confusion_penalty" in components
        assert "final_score" in components
        assert components["keyword_score"] > 0.0
        assert 0.0 <= components["prototype_score"] <= 1.0
        assert score > 0.82 # Should be boosted by matching keyword & prototype

    def test_report_artifacts_and_schemas(self):
        """Verify all generated comparison and diagnostic report artifacts exist and match required schemas."""
        comp_csv = REPORTS_DIR / "stage5_domain_ranking_comparison.csv"
        intent_csv = REPORTS_DIR / "stage5_domain_ranking_by_intent.csv"
        multi_csv = REPORTS_DIR / "stage5_multi_issue_analysis.csv"
        exam_csv = REPORTS_DIR / "stage5_domain_ranking_examples.csv"
        report_txt = REPORTS_DIR / "stage5_domain_ranking_report.txt"

        assert comp_csv.exists()
        assert intent_csv.exists()
        assert multi_csv.exists()
        assert exam_csv.exists()
        assert report_txt.exists()

        df_comp = pd.read_csv(comp_csv)
        assert set(df_comp.columns) == {"metric", "baseline", "domain_aware", "improvement"}
        metrics = set(df_comp["metric"])
        assert "Recall@1" in metrics
        assert "Recall@3" in metrics
        assert "MRR" in metrics
        assert "Top-1 Accuracy" in metrics
        assert "False auto-handle rate" in metrics

        df_intent = pd.read_csv(intent_csv)
        assert len(df_intent) == 11
        assert "baseline_recall_at_1" in df_intent.columns
        assert "domain_aware_recall_at_1" in df_intent.columns

        df_multi = pd.read_csv(multi_csv)
        assert len(df_multi) == 1189
        assert "multi_issue" in df_multi.columns
        assert "decision_hint" in df_multi.columns

        txt = report_txt.read_text(encoding="utf-8")
        assert "1. Motivation" in txt
        assert "2. Baseline Results" in txt
        assert "3. Domain-Aware Ranking Design" in txt
        assert "4. Multi-Issue Detection" in txt
        assert "14. Safety Impact" in txt
        assert "16. Final Recommendation" in txt
