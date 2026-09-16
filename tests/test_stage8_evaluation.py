"""
Unit Tests for Stage 8: Final System Evaluation & Adversarial Testing
======================================================================
Validates test set integrity, metric calculation correctness, zero leakage,
prompt-injection evaluator, failure analysis schema, and LLM-as-judge fallback.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.stage8_evaluation import Stage8Evaluator
from src.stage5_retrieval import load_data, verify_split_leakage
from src.stage7_decision import run_human_review_evaluation


@pytest.fixture(scope="module")
def evaluator():
    """Shared evaluator instance for fast testing."""
    return Stage8Evaluator(seed=42)


def test_test_set_size_and_integrity(evaluator):
    """Test 1: Verify exact unseen test set size is 1,189 cases."""
    assert len(evaluator.test_df) == 1189
    assert len(evaluator.test_cases) == 1189
    assert "case_id" in evaluator.test_df.columns
    assert "intent_id" in evaluator.test_df.columns
    assert "customer_text" in evaluator.test_df.columns


def test_split_leakage_zero_overlap(evaluator):
    """Test 2: Verify zero cross-split leakage across case IDs, thread roots, and raw texts."""
    leakage = verify_split_leakage(evaluator.train_df, evaluator.val_df, evaluator.test_df)
    assert leakage["passed"] is True
    assert leakage["case_overlap"] == 0
    assert leakage["root_overlap"] == 0
    assert leakage["train_size"] == 5545
    assert leakage["val_size"] == 1188
    assert leakage["test_size"] == 1189


def test_intent_classification_metrics(evaluator):
    """Test 3: Verify Stage 4 intent evaluation on test split produces valid metrics and confusion matrix."""
    intent_res = evaluator.evaluate_intent_classification()
    assert 0.50 <= intent_res["accuracy"] <= 1.00
    assert 0.40 <= intent_res["macro_f1"] <= 1.00
    assert len(intent_res["per_intent_table"]) == 11
    assert intent_res["confusion_matrix"].shape == (11, 11)
    assert "intent" in intent_res["strongest_intent"]
    assert len(intent_res["top_confusion_pairs"]) > 0


def test_retrieval_benchmarking_vs_baseline(evaluator):
    """Test 4: Verify semantic FAISS retrieval outperforms TF-IDF baseline across all metrics."""
    ret_res = evaluator.evaluate_retrieval()
    assert ret_res["leakage_check_passed"] is True
    sem = ret_res["semantic_metrics"]
    tfidf = ret_res["tfidf_metrics"]

    assert sem["Recall@1"] > tfidf["Recall@1"]
    assert sem["Recall@3"] > tfidf["Recall@3"]
    assert sem["Recall@5"] > tfidf["Recall@5"]
    assert sem["Recall@10"] > tfidf["Recall@10"]
    assert sem["MRR"] > tfidf["MRR"]
    assert len(ret_res["comparison_table"]) == 5


def test_decision_safety_metrics(evaluator):
    """Test 5: Verify Stage 7 decision policy metrics on test set."""
    gen_res = evaluator.evaluate_generation_and_grounding()
    metrics = gen_res["decision_metrics"]

    assert metrics["n"] == 1189
    assert metrics["auto_handle_count"] + metrics["escalate_count"] == 1189
    assert metrics["auto_handle_precision"] >= 0.95
    assert metrics["false_auto_handle_rate"] < 0.05  # Safety constraint < 5.0%
    assert metrics["escalation_recall"] >= 0.95


def test_prompt_injection_resistance_evaluation(evaluator):
    """Test 6: Verify prompt injection evaluation executes attacks and achieves 100% neutralization."""
    inj_res = evaluator.evaluate_prompt_injection()
    assert inj_res["attacks_tested"] >= 5
    assert inj_res["attacks_neutralized"] == inj_res["attacks_tested"]
    assert inj_res["neutralization_rate"] == 100.0
    assert inj_res["attack_success_rate"] == 0.0


def test_llm_judge_rubric_schema_and_fallback(evaluator):
    """Test 7: Verify LLM-as-Judge evaluation outputs structured 5-point rubric scores for N=50 cases."""
    gen_res = evaluator.evaluate_generation_and_grounding()
    df_judge, judge_metrics = evaluator.evaluate_llm_judge(gen_res["test_results_df"], sample_size=50)

    assert len(df_judge) == 50
    assert set([
        "case_id", "intent", "judge_intent_score", "judge_grounding_score",
        "judge_helpfulness_score", "judge_unsupported_claim_score",
        "judge_overall_score", "judge_reason"
    ]).issubset(df_judge.columns)

    assert df_judge["judge_intent_score"].between(0, 2).all()
    assert df_judge["judge_grounding_score"].between(0, 2).all()
    assert df_judge["judge_helpfulness_score"].between(0, 2).all()
    assert df_judge["judge_unsupported_claim_score"].between(0, 2).all()
    assert df_judge["judge_overall_score"].between(1, 5).all()
    assert 1.0 <= judge_metrics["mean_overall_score"] <= 5.0


def test_human_vs_judge_comparison_distinction(evaluator):
    """Test 8: Verify Human vs. Judge comparison documents methodological boundaries."""
    gen_res = evaluator.evaluate_generation_and_grounding()
    human_df, human_metrics = run_human_review_evaluation(gen_res["test_results_df"], sample_size=50)
    judge_df, _ = evaluator.evaluate_llm_judge(gen_res["test_results_df"], sample_size=50)

    comp = evaluator.compare_human_and_judge(human_df, judge_df)
    assert comp["compared_sample_size"] == 50
    assert 0.0 <= comp["proxy_agreement_pct"] <= 100.0
    assert -1.0 <= comp["proxy_cohen_kappa"] <= 1.0
    assert len(comp["conceptual_distinction_note"]) > 20


def test_deterministic_seed_reproducibility(evaluator):
    """Test 9: Verify evaluator produces identical results across multiple runs with fixed seed."""
    gen_res = evaluator.evaluate_generation_and_grounding()
    df_judge1, metrics1 = evaluator.evaluate_llm_judge(gen_res["test_results_df"], sample_size=50)
    df_judge2, metrics2 = evaluator.evaluate_llm_judge(gen_res["test_results_df"], sample_size=50)

    assert df_judge1["case_id"].tolist() == df_judge2["case_id"].tolist()
    assert df_judge1["judge_overall_score"].tolist() == df_judge2["judge_overall_score"].tolist()
    assert metrics1["mean_overall_score"] == metrics2["mean_overall_score"]
