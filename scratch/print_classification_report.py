"""
Print exact classification report and metrics for the current system.
"""
import os
import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

PROJECT_ROOT = Path(r"c:\Hiver")
sys.path.insert(0, str(PROJECT_ROOT))

# Configure offline / CPU environment
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

from src.stage7_decision import CalibratedIntentClassifier, load_data

def main():
    train_df, val_df, test_df, resolved_threads, taxonomy = load_data()
    classifier = CalibratedIntentClassifier()
    classifier.fit(train_df)

    y_true = test_df["intent_id"].astype(str).tolist()
    texts = test_df["customer_text"].fillna("").astype(str).tolist()

    y_pred = []
    for t in texts:
        pred_intent, _ = classifier.predict(t)
        y_pred.append(pred_intent)

    labels = sorted(list(set(y_true)))
    report = classification_report(y_true, y_pred, labels=labels, digits=4)
    print("=== CLASSIFICATION REPORT ===")
    print(report)

    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro")
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted")

    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Macro Precision: {p_macro:.4f} ({p_macro*100:.2f}%)")
    print(f"Macro Recall: {r_macro:.4f} ({r_macro*100:.2f}%)")
    print(f"Macro F1: {f1_macro:.4f}")
    print(f"Weighted F1: {f1_weighted:.4f}")

if __name__ == "__main__":
    main()
