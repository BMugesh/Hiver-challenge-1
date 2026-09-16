"""
Inspect query lengths, short queries, and categorization nuances.
"""
import sys
import pandas as pd

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

df = pd.read_csv("scratch/all_495_errors.csv")
lengths = [len([w for w in str(q).split() if not w.startswith("@") and not w.startswith("http")]) for q in df["query"]]
df["word_len"] = lengths

print("Shortest 20 queries in errors:")
for idx, row in df.sort_values("word_len").head(20).iterrows():
    print(f"Len: {row['word_len']:<2} | Exp: {row['expected_intent']:<30} | Pred: {row['predicted_top1_intent']:<30} | Query: {row['query']}")

print(f"\nCount of queries with word_len <= 6: {(df['word_len'] <= 6).sum()}")
print(f"Count of queries with word_len <= 8: {(df['word_len'] <= 8).sum()}")
print(f"Count of queries with word_len <= 10: {(df['word_len'] <= 10).sum()}")
print(f"Count of queries with word_len <= 12: {(df['word_len'] <= 12).sum()}")
