import json
import re
import pandas as pd

with open('data/evaluation/golden_set.json', 'r', encoding='utf-8') as f:
    golden = json.load(f)

train = pd.read_csv('data/processed/splits/train.csv')
val = pd.read_csv('data/processed/splits/validation.csv')
test = pd.read_csv('data/processed/splits/test.csv')

with open('data/processed/retrieval/train_case_metadata.json', 'r', encoding='utf-8') as f:
    faiss_meta = json.load(f)

train_roots = set(train['thread_root_id'].dropna().astype(int))
val_roots = set(val['thread_root_id'].dropna().astype(int))
test_roots = set(test['thread_root_id'].dropna().astype(int))
faiss_roots = set(int(m['thread_root_id']) for m in faiss_meta)
all_excluded_roots = train_roots | val_roots | test_roots | faiss_roots

train_texts = set(train['customer_text'].dropna().str.strip().str.lower())
val_texts = set(val['customer_text'].dropna().str.strip().str.lower())
test_texts = set(test['customer_text'].dropna().str.strip().str.lower())
faiss_texts = set(m['customer_problem'].strip().lower() for m in faiss_meta)

golden_messages = [g['customer_message'].strip().lower() for g in golden]
golden_sources = [g['source_id'] for g in golden]

exact_train = sum(1 for m in golden_messages if m in train_texts)
exact_val = sum(1 for m in golden_messages if m in val_texts)
exact_test = sum(1 for m in golden_messages if m in test_texts)
exact_faiss = sum(1 for m in golden_messages if m in faiss_texts)

overlap_roots = []
for s in golden_sources:
    match = re.search(r'\d+', s)
    if match:
        rid = int(match.group(0))
        if rid in all_excluded_roots:
            overlap_roots.append((s, rid))

print("Audit Results:")
print(f"  Exact Train Overlap: {exact_train}")
print(f"  Exact Val Overlap:   {exact_val}")
print(f"  Exact Test Overlap:  {exact_test}")
print(f"  Exact FAISS Overlap: {exact_faiss}")
print(f"  Thread Root Overlap: {len(overlap_roots)}")
if overlap_roots:
    print(f"  Overlaps: {overlap_roots}")
