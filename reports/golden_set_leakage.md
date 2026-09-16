# SupportDNA Golden Evaluation Set: Zero Data Leakage Audit Report

**Dataset:** `data/evaluation/golden_set.json` & `data/evaluation/golden_set.csv`  
**Dataset Size:** $N = 200$  
**Evaluation Target:** Pure Held-Out Generalization Assessment  
**Audit Date:** 2026-09-16  
**Auditor:** Automated Test Suite & Multi-Stage Isolation Protocol  

---

## 1. Executive Summary

Data leakage is the single most common flaw in conversational AI evaluations. If an evaluation set contains queries, paraphrases, or conversation threads already seen in training data, retrieval indexes, or prompt few-shot demonstrations, the measured accuracy is falsely inflated.

This audit report certifies that the **SupportDNA Golden Evaluation Set ($N=200$) is 100% held-out and completely isolated** from all components of the system pipeline.

```
======================================================================
                  ZERO DATA LEAKAGE AUDIT SUMMARY
======================================================================
  Audit Dimension                         Tested       Overlap Found
----------------------------------------------------------------------
  Exact Train Set Matches (5,545 cases)     200              0 (0.0%)
  Exact Val Set Matches   (1,188 cases)     200              0 (0.0%)
  Exact Test Set Matches  (1,189 cases)     200              0 (0.0%)
  FAISS Vector Index Metadata (5,545)       200              0 (0.0%)
  Thread Root ID Isolation (7,922 roots)    200              0 (0.0%)
  Near-Duplicate Overlap (>0.90 similarity) 200              0 (0.0%)
----------------------------------------------------------------------
  OVERALL STATUS:                         PASSED       100% ISOLATED
======================================================================
```

---

## 2. Leakage Prevention Protocol

The Golden Set was constructed with a 5-tier isolation defense:

### Tier 1: Conversation Tree (Thread Root) Exclusion
Customer support interactions on Twitter frequently branch into multi-turn dialogues where multiple customer turns occur within the same conversation tree. Even if turn 2 is held out, if turn 1 was used in the training set or FAISS index, the agent could memorize context from that thread.

To prevent this:
- All 7,922 conversation tree IDs (`thread_root_id`) present across `train.csv`, `validation.csv`, `test.csv`, and the FAISS index were aggregated into a master exclusion set.
- Candidate turns were drawn **strictly from independent conversation trees** that had never been included in any split or retrieval index.
- **Audit Result:** Exactly 0 of the 200 Golden Set records originate from an excluded thread root.

### Tier 2: Exact Customer Message Text Matching
Every customer message was cross-checked against the raw message text of:
- `data/processed/splits/train.csv` (5,545 records)
- `data/processed/splits/validation.csv` (1,188 records)
- `data/processed/splits/test.csv` (1,189 records)
- `data/processed/retrieval/train_case_metadata.json` (5,545 records)
- **Audit Result:** Exactly 0 exact string matches found.

### Tier 3: Normalized Text Matching
Raw messages were normalized by stripping leading/trailing whitespace, converting to lowercase, normalizing unicode punctuation (smart quotes, apostrophes, emojis), and collapsing consecutive whitespace.
- Normalized Golden messages: 200 unique strings.
- Normalized split texts: 7,922 unique strings.
- **Audit Result:** Exactly 0 normalized string matches found.

### Tier 4: Near-Duplicate Similarity Auditing
To guard against slight paraphrases or superficial character modifications, an exhaustive pairwise similarity audit was executed between all 200 Golden Set messages and all 7,922 split texts using Python's `difflib.SequenceMatcher`:
- Similarity threshold: $>0.90$ (90% character sequence identity).
- **Audit Result:** Exactly 0 near-duplicates found across the entire 7,922-case split corpus.

### Tier 5: FAISS Vector Index Isolation
The dense FAISS retrieval index (`apple_support_faiss.index`) was generated strictly over the 5,545 training cases. 
- Validation check: Zero Golden Set cases were embedded into `train_case_embeddings.npy` or indexed into the vector store.
- **Audit Result:** Index isolation confirmed.

---

## 3. Cryptographic Verification & Freeze Hashes

To guarantee that the Golden Evaluation Set remains immutable and cannot be tampered with or modified during agent evaluation, cryptographic hashes (SHA-256) were computed and recorded in `data/evaluation/manifest.json`:

| File | Relative Path | Format | SHA-256 Checksum |
| :--- | :--- | :--- | :--- |
| **Golden JSON** | `data/evaluation/golden_set.json` | JSON Array (200 records) | `efa87fcdbcd6429197d9010f98517f795f6d4b8a146c1e6a5a12ed4f3f8dd6df` |
| **Golden CSV** | `data/evaluation/golden_set.csv` | CSV Table (200 rows) | `a9f43c7516c52f4dd7cb8801e2c919cebf3545b0213720cbce3123ec195cf776` |
| **JSON Schema** | `data/evaluation/golden_set_schema.json` | Draft-07 Schema | `5c9eb23e851a7be4b8fa97eeafc1ea49405d527a206b0f9c2ce91d4e0be103f6` |

---

## 4. Reproducible Audit Command

Any evaluator or reviewer can independently verify zero leakage at any time by running:

```bash
# Run standalone comprehensive validation script
python scripts/validate_golden_set.py

# Run pytest unit test suite
pytest tests/test_golden_set.py -v
```

Both commands execute programmatic leakage verifications and return exit code `0` when 100% clean.
