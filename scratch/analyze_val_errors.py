import os
import sys
from pathlib import Path
import pandas as pd

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.stage5_retrieval import load_data
from src.stage7_decision import CalibratedIntentClassifier

train_df, val_df, test_df, _, _ = load_data()
clf = CalibratedIntentClassifier()
clf.fit(train_df)

val_texts = val_df["customer_text"].fillna("").astype(str).tolist()
val_true = val_df["intent_id"].astype(str).tolist()

correct = 0
val_errors = []
for t, y in zip(val_texts, val_true):
    p, conf = clf.predict(t)
    if p == y:
        correct += 1
    else:
        val_errors.append({"text": t, "true": y, "pred": p, "conf": conf})

print(f"Validation Accuracy: {correct / len(val_texts):.4f} ({correct} / {len(val_texts)})")

df_err = pd.DataFrame(val_errors)
print("\nTop 10 Validation Confusions:")
print(df_err.groupby(["true", "pred"]).size().sort_values(ascending=False).head(10))

print("\nSample APP_STORE_PURCHASES_BILLING errors in validation:")
for _, r in df_err[df_err["true"] == "APP_STORE_PURCHASES_BILLING"].head(5).iterrows():
    print(f"  Query: {r['text'][:80]} | Pred: {r['pred']}")

print("\nSample APP_CRASH_AND_DOWNLOAD errors in validation:")
for _, r in df_err[df_err["true"] == "APP_CRASH_AND_DOWNLOAD"].head(5).iterrows():
    print(f"  Query: {r['text'][:80]} | Pred: {r['pred']}")

print("\nSample KEYBOARD_TYPING_AUTOCORRECT errors in validation:")
for _, r in df_err[df_err["true"] == "KEYBOARD_TYPING_AUTOCORRECT"].head(5).iterrows():
    print(f"  Query: {r['text'][:80]} | Pred: {r['pred']}")
