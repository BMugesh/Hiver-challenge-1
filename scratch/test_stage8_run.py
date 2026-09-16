import os
import sys
from pathlib import Path

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.stage8_evaluation import Stage8Evaluator

evaluator = Stage8Evaluator(seed=42)
intent_res = evaluator.evaluate_intent_classification()

print(f"Overall Accuracy: {intent_res['accuracy'] * 100:.2f}% ({int(intent_res['accuracy'] * 1189)} / 1189)")
print(f"Macro Precision:  {intent_res['macro_precision'] * 100:.2f}%")
print(f"Macro Recall:     {intent_res['macro_recall'] * 100:.2f}%")
print(f"Macro F1:         {intent_res['macro_f1']:.4f}")
print("\nPer-Intent Table:")
print(intent_res["per_intent_table"].to_string(index=False))

print("\nTop 5 Confusion Pairs:")
for p in intent_res["top_confusion_pairs"]:
    print(f"  {p['true_intent']} -> {p['pred_intent']}: {p['count']}")
