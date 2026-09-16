"""
Inspect queries that are short or ambiguous in the 495 errors.
"""
import sys
import pandas as pd

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

df = pd.read_csv("scratch/all_495_errors.csv")
for idx, row in df.iterrows():
    q = str(row["query"]).strip()
    words = [w for w in q.split() if not w.startswith("@") and not w.startswith("http")]
    if len(words) <= 12:
        print(f"Len: {len(words):<2} | Exp: {row['expected_intent']:<30} | Pred: {row['predicted_top1_intent']:<30} | Q: {q}")
