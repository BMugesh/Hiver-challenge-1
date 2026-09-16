# SupportDNA Golden Evaluation Set: Sampling & Labelling Methodology

**Dataset Version:** `1.0.0 (FROZEN)`  
**Brand:** `AppleSupport`  
**Size:** 200 Hand-Labelled held-out customer inquiries  
**Target Window:** 150–250 (Target: ~200)  
**Corpus:** Customer Support on Twitter (`AppleSupport` subset)  
**Integrity Status:** 100% Held-Out (Zero Train / Validation / Test / FAISS Leakage)  

---

## 1. Executive Summary & Assignment Context

The Hiver SDE Intern take-home assignment requires building a **Golden Evaluation Set** of 150–250 hand-labelled customer queries with a comprehensive note documenting how they were sampled, labelled, and verified. 

Crucially, the assignment emphasizes that:
> *"Proving your system works is as important as building it. An agent that generates great replies 70% of the time but hallucinate dangerous steps 5% of the time is worse than an agent that answers 40% and safely escalates 60%."*

To satisfy this standard, the Golden Set was designed not merely as an intent classification test, but as a **comprehensive behavioral benchmark** for the entire support-agent pipeline:
1. **Customer Intent Understanding:** Classifying raw inquiries into the 11 verified business intents.
2. **Operational Decision Policy:** Determining whether the agent should directly `ANSWER`, `GUIDE`, `CLARIFY`, `ESCALATE`, `SAFE_REFUSAL`, or `SAFE_REFUSAL_AND_ESCALATE`.
3. **Evidence-Grounded Drafting:** Testing whether the agent appropriately relies on, rejects, or clarifies against historical resolution evidence.
4. **Safety & Defensibility:** Ensuring the agent never gives unsupported troubleshooting, ignores customer follow-ups, or fails to escalate hazardous situations.

> [!IMPORTANT]
> **Strict Anti-Tuning Principle:**
> This Golden Evaluation Set is **strictly held-out**. It has never been and will never be used to train intent classifiers, tune prompts, calibrate retrieval thresholds, fit logistic regressions, or build vector indexes. It serves exclusively as a permanent, blind evaluation benchmark.

---

## 2. Source Population & Sampling Frame

### 2.1 Population Profiling
The raw AppleSupport corpus comprises:
- **Total Flat Tweets:** 233,977 tweets (`data/raw/apple_support_filtered_threads.csv`).
- **Reconstructed Multi-Turn Trees:** 80,247 conversation trees (`data/processed/apple_support_threads.json`).
- **Curated Resolved Historical Cases:** 7,922 cases (`data/processed/apple_support_resolved_threads.json`).
- **Existing Splits:**
  - `train.csv`: 5,545 cases (70.0%) — indexed in dense FAISS corpus.
  - `validation.csv`: 1,188 cases (15.0%) — used for Stage 7 threshold tuning.
  - `test.csv`: 1,189 cases (15.0%) — used for Stage 8 offline benchmarking.
- **Excluded Thread Roots:** 7,922 unique conversation trees across train, validation, test, and FAISS.

### 2.2 Eligible Held-Out Sampling Pool
After filtering out all 7,922 conversation trees present in any split or retrieval index, the remaining corpus contained **90,505 eligible customer turns** from independent, unindexed conversation trees (`apple_support_turn_structure.csv`).

### 2.3 Stratification Dimensions
Rather than taking 200 random rows (which would over-represent simple battery complaints and under-represent complex edge cases), a multi-dimensional stratified sampling protocol was executed across five dimensions:

| Dimension | Stratification Categories | Target Distribution Strategy |
| :--- | :--- | :--- |
| **1. Business Intent** | All 11 Verified Business Intents | Balanced coverage (14 to 24 cases per intent, 7.0% – 12.0%) |
| **2. Operational Action** | `ANSWER`, `GUIDE`, `CLARIFY`, `ESCALATE`, `SAFE_REFUSAL`, `SAFE_REFUSAL_AND_ESCALATE` | Reflect realistic support operations (majority auto-handle, 11% escalation, refusal for security/piracy) |
| **3. Query Difficulty** | Clear, Short/Vague, Ambiguous, Multi-Issue, Unusual Wording, High Frustration | Include real conversational friction, slang, emotional outbursts, and typos |
| **4. Conversation State** | First Inquiries, Follow-Up Inquiries, Unresolved Repeated Failures | Test conversational memory and avoidance of repeating failed steps |
| **5. Evidence Expectation** | `STRONG`, `MODERATE`, `WEAK`, `INSUFFICIENT`, `CONFLICTING`, `NO_HISTORICAL_EVIDENCE_REQUIRED` | Test agent's ability to recognize when historical evidence is inadequate or contradictory |

---

## 3. Human Annotation & Labelling Protocol

Every example in the 200-case dataset was subjected to human review and annotated against a strict 11-field schema.

### 3.1 Field Schema & Semantics

1. **`golden_id`:** Unique sequential identifier (`GOLDEN_0001` to `GOLDEN_0200`).
2. **`source_id`:** Held-out originating thread root or turn ID (e.g. `SRC_ROOT_1573015`).
3. **`customer_message`:** The **exact, raw, unedited customer message**. No grammatical correction, punctuation cleanup, or synthetic smoothing.
4. **`ground_truth_intent`:** Exactly one of the 11 verified business intents from `apple_support_intent_taxonomy.json`.
5. **`ground_truth_customer_goal`:** Clear, objective statement of what the customer is trying to accomplish.
6. **`ground_truth_issues`:** Array of specific technical symptoms or issues raised (e.g. `["autocorrect_letter_i_bug", "keyboard_substitution_glitch"]`).
7. **`ground_truth_action`:**
   - `ANSWER`: Evidence supports a direct, complete, factual answer without multi-step troubleshooting.
   - `GUIDE`: Customer requires procedural troubleshooting grounded in historical resolution paths.
   - `CLARIFY`: Essential diagnostic information is missing before safe guidance can be offered.
   - `ESCALATE`: Case requires human intervention (hardware hazards, repeated failures, legal/account ownership issues).
   - `SAFE_REFUSAL`: Request violates policy/terms (e.g. jailbreaking, pirating apps) and must be politely refused.
   - `SAFE_REFUSAL_AND_ESCALATE`: Request contains a security/bypass concern (e.g. activation lock bypass, stolen device access) that must be refused and transferred to specialized human review.
8. **`escalation_required`:** Boolean flag (`True` for human handoff, `False` for auto-handling).
9. **`evidence_expectation`:** Expected retrieval profile:
   - `STRONG_HISTORICAL_EVIDENCE`: High-similarity, unambiguous historical precedent exists.
   - `MODERATE_HISTORICAL_EVIDENCE`: Partial precedent exists; requires careful adaptation.
   - `WEAK_HISTORICAL_EVIDENCE`: Low similarity or tangential historical cases.
   - `INSUFFICIENT_HISTORICAL_EVIDENCE`: Zero or near-zero relevant historical precedent.
   - `CONFLICTING_HISTORICAL_EVIDENCE`: Historical cases offer contradictory remediation patterns (e.g., reset network settings vs DFU restore).
   - `NO_HISTORICAL_EVIDENCE_REQUIRED`: Self-contained policy/safety determination.
10. **`expected_reply_requirements`:** Array of minimum evaluation criteria that a generated reply must fulfill. **No exact reference answers are written**, avoiding artificial n-gram bias in LLM-as-judge scoring.
11. **`edge_case_category`:** Applicable difficulty tags (`CLEAR_INTENT`, `SHORT_VAGUE`, `AMBIGUOUS`, `MULTI_ISSUE`, `FOLLOW_UP`, `UNRESOLVED_FOLLOW_UP`, `CONFLICTING_EVIDENCE`, `ESCALATION_SENSITIVE`, `SAFETY_SENSITIVE`, `UNUSUAL_WORDING`).
12. **`human_reasoning`:** Concise, explicit rationale explaining why the intent, action, and evidence expectations were assigned.

---

## 4. Policy on Clarification vs Direct Answering

A common failure mode in customer-support agents is **over-clarification** (asking unnecessary questions when the issue is already clear) or **under-clarification** (guessing troubleshooting steps on vague queries).

The Golden Set establishes a principled boundary:
- **Direct Guidance (`GUIDE` / `ANSWER`):** If a customer reports a known signature symptom (e.g., *"Why does autocorrect change I to an exclamation mark?"* or *"How do I force restart my iPhone 8?"*), the agent must answer directly without demanding device model or iOS version.
- **Clarification (`CLARIFY`):** If a customer sends an ambiguous or single-word query (e.g., *"@AppleSupport update"* or *"My phone is broken"*), the agent must ask targeted diagnostic questions before prescribing fixes.
- **Escalation (`ESCALATE`):** If the customer has already attempted standard troubleshooting (e.g., *"I already restarted and reset network settings, still no Wi-Fi"*), the agent must **not** repeat the same steps; it must escalate.

---

## 5. Dedicated Security Separation

Following assignment guidelines, adversarial prompt-injection testing is kept separate from the Golden Set:
- **Golden Evaluation Set ($N=200$):** Real held-out customer support inquiries reflecting authentic user behavior, frustrations, and edge cases.
- **Security Regression Benchmark (`data/security/prompt_injection/`):** Dedicated adversarial test suite containing prompt injection, system prompt extraction, delimiter evasion, and jailbreak vectors.
- Genuine security-sensitive customer requests from the wild (such as Activation Lock bypass inquiries or suspicious phishing alerts) are included in the Golden Set with `SAFE_REFUSAL` and `SAFE_REFUSAL_AND_ESCALATE` labels.

---

## 6. Zero Data Leakage Protocol

Data leakage renders evaluation invalid. The Golden Set implements five layers of programmatic isolation:
1. **Thread Root Isolation:** Zero records share a conversation tree ID (`thread_root_id`) with any case in `train.csv`, `validation.csv`, `test.csv`, or the FAISS index.
2. **Exact Text Isolation:** Zero customer messages match any turn text in the training, validation, or test sets.
3. **Normalized Text Isolation:** Zero customer messages match lowercase, stripped split texts.
4. **Near-Duplicate Exclusion:** SequenceMatcher similarity check across all 7,922 split texts confirms 0 cases with $>0.90$ similarity.
5. **Retrieval Index Isolation:** The FAISS index (`apple_support_faiss.index`) was built strictly over 5,545 training cases and never contains Golden Set vectors.

The leakage audit results are permanently verified via `scripts/validate_golden_set.py` and recorded in `reports/golden_set_leakage.md`.
