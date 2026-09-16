"""
Unit Tests for Stage 5: Top-1 Retrieval Error Analysis
======================================================
Verifies:
1. Exactly 495 Top-1 errors are identified from 1,189 test queries (Recall@1 = 58.37%).
2. Category counts sum to exactly 495 (100.0% of errors).
3. Percentages of errors and percentage of test set are calculated correctly.
4. Confusion pairs are calculated accurately with zero self-confusions.
5. No correct Top-1 prediction is included in the error dataset.
6. All required output files are generated and non-empty:
   - reports/stage5_error_categories.csv
   - reports/stage5_top_confusions.csv
   - reports/stage5_error_examples.csv
   - reports/stage5_intent_error_analysis.csv
   - reports/stage5_top1_error_analysis.txt
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
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
from src.stage5_error_analysis import (
    extract_test_errors,
    classify_failure,
    FAILURE_CATEGORIES,
    run_error_analysis,
    REPORTS_DIR
)


@pytest.fixture(scope="module")
def error_analysis_data():
    """Load test dataset, FAISS index, and perform error extraction once."""
    train_df, val_df, test_df, resolved_threads, taxonomy = load_data()
    threads_by_case = {t["case_id"]: t for t in resolved_threads}
    train_cases = build_case_representations(train_df, threads_by_case)
    test_cases = build_case_representations(test_df, threads_by_case)

    model = init_embedding_model()
    index, embeddings, metadata = build_or_load_faiss_index(model, train_cases)
    retriever = SemanticRetriever(model, index, metadata)

    errors, correct_scores, wrong_scores = extract_test_errors(retriever, test_cases, top_k=10)

    train_counts_by_intent = dict(train_df["intent_id"].value_counts())
    categories = [classify_failure(err, train_counts_by_intent)[0] for err in errors]

    return {
        "test_cases": test_cases,
        "errors": errors,
        "correct_scores": correct_scores,
        "wrong_scores": wrong_scores,
        "categories": categories,
        "taxonomy": taxonomy
    }


class TestStage5ErrorAnalysis:

    def test_exact_error_count_and_recall(self, error_analysis_data):
        """Verify exactly 495 errors and 694 correct predictions out of 1,189 queries."""
        test_cases = error_analysis_data["test_cases"]
        errors = error_analysis_data["errors"]
        correct_scores = error_analysis_data["correct_scores"]
        wrong_scores = error_analysis_data["wrong_scores"]

        assert len(test_cases) == 1189
        assert len(errors) == 495
        assert len(wrong_scores) == 495
        assert len(correct_scores) == 694
        assert len(errors) + len(correct_scores) == 1189

        recall_1 = len(correct_scores) / len(test_cases) * 100.0
        assert round(recall_1, 2) == 58.37

    def test_no_correct_predictions_in_errors(self, error_analysis_data):
        """Verify zero correct Top-1 matches are included in the error set."""
        errors = error_analysis_data["errors"]
        for err in errors:
            assert err["predicted_top1_intent"] != err["expected_intent"], (
                f"Error set contains matching prediction: {err['case_id']} with intent {err['expected_intent']}"
            )

    def test_category_counts_sum_to_495(self, error_analysis_data):
        """Verify failure categories cover all 495 errors without omissions or duplicates."""
        categories = error_analysis_data["categories"]
        assert len(categories) == 495

        for cat in categories:
            assert cat in FAILURE_CATEGORIES, f"Unknown category: {cat}"

    def test_percentage_calculations(self, error_analysis_data):
        """Verify category percentage arithmetic across 495 errors and 1,189 total queries."""
        categories = error_analysis_data["categories"]
        cat_counts = pd.Series(categories).value_counts()

        total_err_pct = sum(cnt / 495.0 * 100.0 for cnt in cat_counts)
        total_test_pct = sum(cnt / 1189.0 * 100.0 for cnt in cat_counts)

        np.testing.assert_allclose(total_err_pct, 100.0, rtol=1e-4)
        np.testing.assert_allclose(total_test_pct, 495.0 / 1189.0 * 100.0, rtol=1e-4)

    def test_confusion_pairs_integrity(self, error_analysis_data):
        """Verify confusion pairs do not have expected == predicted and sum to 495."""
        errors = error_analysis_data["errors"]
        confusions = pd.Series(
            [(e["expected_intent"], e["predicted_top1_intent"]) for e in errors]
        ).value_counts()

        assert confusions.sum() == 495
        for (exp, pred), count in confusions.items():
            assert exp != pred
            assert count > 0

    def test_report_artifacts_generated(self):
        """Verify all 4 CSV reports and 1 TXT diagnostic report exist and have valid schemas."""
        cat_csv = REPORTS_DIR / "stage5_error_categories.csv"
        conf_csv = REPORTS_DIR / "stage5_top_confusions.csv"
        exam_csv = REPORTS_DIR / "stage5_error_examples.csv"
        intent_csv = REPORTS_DIR / "stage5_intent_error_analysis.csv"
        report_txt = REPORTS_DIR / "stage5_top1_error_analysis.txt"

        assert cat_csv.exists()
        assert conf_csv.exists()
        assert exam_csv.exists()
        assert intent_csv.exists()
        assert report_txt.exists()

        df_cat = pd.read_csv(cat_csv)
        assert set(df_cat.columns) == {"category", "count", "percentage_of_errors", "percentage_of_test_set", "description"}
        assert df_cat["count"].sum() == 495

        df_conf = pd.read_csv(conf_csv)
        assert set(df_conf.columns) == {"expected_intent", "predicted_intent", "count", "percentage_of_errors"}
        assert df_conf["count"].sum() == 495

        df_exam = pd.read_csv(exam_csv)
        assert set(df_exam.columns) == {
            "query", "expected_intent", "predicted_intent", "top1_similarity",
            "correct_case_rank", "top3_intents", "failure_category", "why_retrieval_failed"
        }
        assert len(df_exam) == 20

        df_intent = pd.read_csv(intent_csv)
        assert set(df_intent.columns) == {
            "intent", "training_count", "test_count", "correct_top1",
            "wrong_top1", "recall_at_1", "error_rate", "common_failure_category"
        }
        assert len(df_intent) == 11
        assert df_intent["wrong_top1"].sum() == 495
        assert df_intent["correct_top1"].sum() == 694

        txt_content = report_txt.read_text(encoding="utf-8")
        assert "1. Overview" in txt_content
        assert "2. Error Category Distribution" in txt_content
        assert "3. Top Intent Confusions" in txt_content
        assert "4. Similarity Score Analysis" in txt_content
        assert "5. Correct Case Rank Analysis" in txt_content
        assert "6. Training Data / Intent Analysis" in txt_content
        assert "7. Top 20 Representative Failures" in txt_content
        assert "8. Root Cause Analysis" in txt_content
        assert "9. What This Suggests We Should Improve" in txt_content
        assert "10. Conclusion" in txt_content
        assert "Why is Recall@1 only 58.37%?" in txt_content
        assert "Is the correct answer actually being retrieved but ranked incorrectly?" in txt_content
