# SupportDNA Golden Evaluation Set: Comprehensive Statistical Profile

**Dataset Identifier:** `data/evaluation/golden_set.json` / `data/evaluation/golden_set.csv`  
**Dataset Version:** `1.0.0 (FROZEN)`  
**Total Evaluation Cases:** $N = 200$  
**Evaluation Scope:** Complete AI Support Agent Behavioral Assessment  
**Isolation Guarantee:** 100% Held-Out from all Training, Validation, Test, and FAISS Indexes  

---

## 1. Primary Dataset Overview

| Metric | Value | Description / Constraint |
| :--- | :--- | :--- |
| **Total Evaluation Cases** | **200** | Strict adherence to 150–250 range specification |
| **Intents Represented** | **11 / 11 (100.0%)** | All business intents from `apple_support_intent_taxonomy.json` |
| **Operational Actions** | **6 / 6 (100.0%)** | `ANSWER`, `GUIDE`, `CLARIFY`, `ESCALATE`, `SAFE_REFUSAL`, `SAFE_REFUSAL_AND_ESCALATE` |
| **Evidence Profiles** | **6 / 6 (100.0%)** | All 6 historical evidence expectation classifications present |
| **Unique Customer Messages** | **200 / 200 (100.0%)** | Zero duplicate customer messages across the set |
| **Unique Source Turn Roots** | **200 / 200 (100.0%)** | Zero shared threads within Golden Set or across splits |
| **Average Character Length** | **112.14 ± 31.77** | Realistic conversational text (min: 18, max: 169 chars) |
| **Average Word Count** | **19.31 ± 5.92** | Typical Twitter support customer brevity (min: 2, max: 31 words) |
| **Mean Reply Criteria** | **3.05 criteria / case** | High-precision multi-dimensional verification criteria |

---

## 2. Intent Taxonomy Distribution

The Golden Set deliberately avoids severe imbalance while respecting authentic customer inquiry prevalence in the real world:

```
                                  Intent Distribution (N=200)
OS_UPDATE_SYSTEM_PERFORMANCE     [====================] 24 (12.0%)
BATTERY_CHARGING_POWER           [==================] 22 (11.0%)
CONNECTIVITY_WIFI_BLUETOOTH      [================] 20 (10.0%)
DISPLAY_TOUCH_SCREEN             [================] 20 (10.0%)
KEYBOARD_TYPING_AUTOCORRECT      [==============] 18 (9.0%)
ACCOUNT_APPLEID_ICLOUD           [==============] 18 (9.0%)
APP_STORE_PURCHASES_BILLING      [==============] 18 (9.0%)
AUDIO_SOUND_SPEAKER              [============] 16 (8.0%)
HOW_TO_SETTINGS_CONFIGURATION    [============] 16 (8.0%)
APP_CRASH_AND_DOWNLOAD           [==========] 14 (7.0%)
GENERAL_DEVICE_INQUIRY           [==========] 14 (7.0%)
```

### Tabular Intent Breakdown

| Intent Class | Case Count | Proportion | Primary Customer Problem Focus |
| :--- | :---: | :---: | :--- |
| `OS_UPDATE_SYSTEM_PERFORMANCE` | 24 | 12.0% | Storage bloat, stuck updates, boot loops, lag after updating |
| `BATTERY_CHARGING_POWER` | 22 | 11.0% | Overnight drain, fast discharge, cable/adapter heating, degradation |
| `CONNECTIVITY_WIFI_BLUETOOTH` | 20 | 10.0% | Wi-Fi drops, Bluetooth accessory pairing, public hotspot lockouts |
| `DISPLAY_TOUCH_SCREEN` | 20 | 10.0% | Ghost touch, unresponsive digitizers, screen burn, black screen |
| `KEYBOARD_TYPING_AUTOCORRECT` | 18 | 9.0% | Predictive text bar missing, iOS 11 letter 'I' glyph glitch, keyboard lag |
| `ACCOUNT_APPLEID_ICLOUD` | 18 | 9.0% | Two-factor SMS recovery, disabled ID, storage quota, Family Sharing |
| `APP_STORE_PURCHASES_BILLING` | 18 | 9.0% | Accidental in-app purchases, subscription cancellations, refund routing |
| `AUDIO_SOUND_SPEAKER` | 16 | 8.0% | Crackling earpiece, microphone blockage, muted calls, volume limiters |
| `HOW_TO_SETTINGS_CONFIGURATION` | 16 | 8.0% | Control Center customization, Do Not Disturb, AirDrop configuration |
| `APP_CRASH_AND_DOWNLOAD` | 14 | 7.0% | Stuck waiting app icons, crash upon launch, server 503 timeouts |
| `GENERAL_DEVICE_INQUIRY` | 14 | 7.0% | Trade-in trade-offs, serial number lookups, water resistance ratings |
| **Total** | **200** | **100.0%** | **Comprehensive, defensible domain representation** |

---

## 3. Operational Action & Escalation Distribution

In real customer support operations, forcing every query into a canned response or forcing every vague inquiry into a generic clarification is a critical flaw. The Golden Set establishes clear operational standards:

```
                            Operational Action Distribution (N=200)
GUIDE                         [==========================================] 85 (42.5%)
ANSWER                        [======================================] 77 (38.5%)
ESCALATE                      [==========] 21 (10.5%)
CLARIFY                       [=======] 14 (7.0%)
SAFE_REFUSAL                  [=] 2 (1.0%)
SAFE_REFUSAL_AND_ESCALATE     [#] 1 (0.5%)
```

### Tabular Action Breakdown

| Operational Action | Count | Percentage | Mandatory Grounded Agent Behavior |
| :--- | :---: | :---: | :--- |
| `GUIDE` | 85 | 42.5% | Provide step-by-step procedural troubleshooting grounded in historical resolution data. |
| `ANSWER` | 77 | 38.5% | Provide direct factual or policy answer without unnecessary diagnostic hurdles. |
| `ESCALATE` | 21 | 10.5% | Safely route to human queue (repeated failure, hardware failure, legal/security issue). |
| `CLARIFY` | 14 | 7.0% | Query is too vague or ambiguous to safely answer without clarifying diagnostic details. |
| `SAFE_REFUSAL` | 2 | 1.0% | Request violates policy (jailbreaking, piracy); politely refuse with educational rationale. |
| `SAFE_REFUSAL_AND_ESCALATE` | 1 | 0.5% | Security bypass attempt (Activation Lock override); refuse and transfer to verification. |
| **Total** | **200** | **100.0%** | **Balanced operational distribution reflecting production reality** |

### Escalation Breakdown
- **Auto-Handle Target (`escalation_required: False`):** 178 cases (89.0%)
- **Escalation Target (`escalation_required: True`):** 22 cases (11.0%)
- **Safety Philosophy:** The agent is evaluated on whether it safely triggers escalation for all 22 sensitive cases without excessively abandoning the 178 resolvable inquiries.

---

## 4. Evidence Expectation Distribution

The Golden Set evaluates whether the agent understands **when retrieved historical evidence is truly reliable**:

| Evidence Expectation | Count | Percentage | Agent Evaluation Criterion |
| :--- | :---: | :---: | :--- |
| `STRONG_HISTORICAL_EVIDENCE` | 161 | 80.5% | Agent must ground reply strictly in top retrieved historical case. |
| `INSUFFICIENT_HISTORICAL_EVIDENCE` | 13 | 6.5% | Agent must trigger diagnostic fallback rather than hallucinating steps. |
| `MODERATE_HISTORICAL_EVIDENCE` | 12 | 6.0% | Agent must adapt historical pattern while accommodating query variance. |
| `WEAK_HISTORICAL_EVIDENCE` | 7 | 3.5% | Agent must recognize low relevance and refrain from over-confident claims. |
| `NO_HISTORICAL_EVIDENCE_REQUIRED` | 5 | 2.5% | Case is governed by self-contained safety/policy rules (refusal/emergency). |
| `CONFLICTING_HISTORICAL_EVIDENCE` | 2 | 1.0% | Agent must acknowledge conflicting patterns or guide cautious diagnostic steps. |
| **Total** | **200** | **100.0%** | **Evaluates agent's evidence discernment beyond raw vector similarity** |

---

## 5. Query Difficulty & Edge Case Categorization

Real customer support inquiries contain typos, emotional frustration, vague one-word messages, and multi-part questions. Every Golden Set example is categorized:

| Edge Case Category | Count | Percentage of Set | Example Customer Manifestation |
| :--- | :---: | :---: | :--- |
| `CLEAR_INTENT` | 159 | 79.5% | *"How do I force restart my iPhone 8?"* |
| `STRONG_EVIDENCE` | 156 | 78.0% | Well-documented issues with clear Apple Support precedents. |
| `ESCALATION_SENSITIVE` | 32 | 16.0% | Unresolved repeated failures, churn threats, hardware risks. |
| `SHORT_VAGUE` | 16 | 8.0% | *"@AppleSupport update"*, *"@AppleSupport sound"* |
| `INSUFFICIENT_EVIDENCE` | 13 | 6.5% | Incomplete inquiries with no relevant historical match. |
| `FOLLOW_UP` | 12 | 6.0% | *"Yes I tried restarting, still dead by noon."* |
| `SAFETY_SENSITIVE` | 11 | 5.5% | Swollen batteries, electrical sparks, account takeover. |
| `UNRESOLVED_FOLLOW_UP` | 10 | 5.0% | Customer previously followed instructions that failed to resolve issue. |
| `MULTI_ISSUE` | 9 | 4.5% | *"Wi-Fi drops, Bluetooth disconnects, and battery dying in 2 hours."* |
| `UNUSUAL_WORDING` | 7 | 3.5% | Sarcastic phrasing, slang, emoji-heavy frustration, profanity. |
| `CONFLICTING_EVIDENCE` | 2 | 1.0% | Contradictory community advice vs official Apple documentation. |
| `WEAK_EVIDENCE` | 2 | 1.0% | Rare accessory or niche legacy behavior. |

---

## 6. Query Length & Structural Characteristics

| Statistic | Character Length | Word Count | Ground Truth Issues | Expected Reply Requirements |
| :--- | :---: | :---: | :---: | :---: |
| **Minimum** | 18 chars | 2 words | 1 issue | 2 criteria |
| **Maximum** | 169 chars | 31 words | 4 issues | 4 criteria |
| **Mean** | 112.14 chars | 19.31 words | 2.11 issues | 3.05 criteria |
| **Median** | 117.0 chars | 20.0 words | 2.00 issues | 3.00 criteria |
| **Standard Deviation** | 31.77 chars | 5.92 words | 0.82 issues | 0.44 criteria |

- **Single-Issue Queries:** 46 cases (23.0%) — clean, focused technical questions.
- **Multi-Issue Queries:** 154 cases (77.0%) — customer reports symptom combined with emotional grievance, attempted troubleshooting, or secondary failure.
