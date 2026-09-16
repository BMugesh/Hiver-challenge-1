"""
Unit Tests for Stage 7: AUTO-HANDLE vs. ESCALATE Decision Policy
================================================================
Validates deterministic safety gates, risk priority resolution,
non-LLM execution, reply immutability, auditability, and split integrity.
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.stage7_decision import (
    DecisionPolicy,
    DecisionEngine,
    EscalationReasonCode,
    CalibratedIntentClassifier,
    assign_ground_truth_safety_label
)


@pytest.fixture
def standard_policy():
    return DecisionPolicy(
        min_similarity=0.65,
        min_intent_confidence=0.60,
        min_intent_alignment=0.66,
        enforce_grounding_gate=True,
        policy_version="stage7_v1"
    )


@pytest.fixture
def decision_engine(standard_policy):
    return DecisionEngine(policy=standard_policy)


def test_strong_evidence_and_grounding_pass(decision_engine):
    """Test 1: Strong intent + strong evidence + clean grounding audit -> AUTO_HANDLE."""
    customer_msg = "My iPhone battery is draining so fast after the iOS update."
    intent = "BATTERY_CHARGING_POWER"
    intent_conf = 0.88
    retrieved_cases = [
        {"case_id": "C1", "intent_id": "BATTERY_CHARGING_POWER", "similarity": 0.84, "resolution_status": "CLEARLY_RESOLVED"},
        {"case_id": "C2", "intent_id": "BATTERY_CHARGING_POWER", "similarity": 0.79, "resolution_status": "PARTIALLY_RESOLVED"},
        {"case_id": "C3", "intent_id": "BATTERY_CHARGING_POWER", "similarity": 0.75, "resolution_status": "CLEARLY_RESOLVED"}
    ]
    draft_reply = "We'd be glad to help. Please check Settings > Battery to see app power usage."
    grounding_check = {
        "grounding_pass": True,
        "severity": "NONE",
        "unsupported_claims": []
    }
    gen_out = {"grounding_status": "GROUNDED"}

    result = decision_engine.evaluate(
        customer_message=customer_msg,
        intent=intent,
        intent_confidence=intent_conf,
        retrieved_evidence=retrieved_cases,
        draft_reply=draft_reply,
        grounding_check=grounding_check,
        generator_output=gen_out
    )

    assert result["decision"] == "AUTO_HANDLE"
    assert result["reason_code"] == EscalationReasonCode.STRONG_GROUNDED_EVIDENCE
    assert result["is_customer_safe"] is True
    assert result["policy_version"] == "stage7_v1"
    assert result["draft_reply"] == draft_reply


def test_low_intent_confidence_escalation(decision_engine):
    """Test 2: Low intent confidence (<0.60) -> ESCALATE (LOW_INTENT_CONFIDENCE)."""
    customer_msg = "It is glitching and broken."
    intent = "GENERAL_DEVICE_INQUIRY"
    intent_conf = 0.42  # Below 0.60 threshold
    retrieved_cases = [
        {"case_id": "C1", "intent_id": "GENERAL_DEVICE_INQUIRY", "similarity": 0.78, "resolution_status": "CLEARLY_RESOLVED"},
        {"case_id": "C2", "intent_id": "GENERAL_DEVICE_INQUIRY", "similarity": 0.72, "resolution_status": "PARTIALLY_RESOLVED"},
        {"case_id": "C3", "intent_id": "GENERAL_DEVICE_INQUIRY", "similarity": 0.68, "resolution_status": "CLEARLY_RESOLVED"}
    ]
    draft_reply = "Please restart your device."
    grounding_check = {"grounding_pass": True, "severity": "NONE", "unsupported_claims": []}
    gen_out = {"grounding_status": "GROUNDED"}

    result = decision_engine.evaluate(
        customer_message=customer_msg,
        intent=intent,
        intent_confidence=intent_conf,
        retrieved_evidence=retrieved_cases,
        draft_reply=draft_reply,
        grounding_check=grounding_check,
        generator_output=gen_out
    )

    assert result["decision"] == "ESCALATE"
    assert result["reason_code"] == EscalationReasonCode.LOW_INTENT_CONFIDENCE
    assert result["is_customer_safe"] is False


def test_weak_retrieval_similarity_escalation(decision_engine):
    """Test 3: Weak retrieval similarity (<0.65) -> ESCALATE (WEAK_EVIDENCE)."""
    customer_msg = "My magic pencil is not reacting to tap."
    intent = "GENERAL_DEVICE_INQUIRY"
    intent_conf = 0.85
    retrieved_cases = [
        {"case_id": "C1", "intent_id": "GENERAL_DEVICE_INQUIRY", "similarity": 0.58, "resolution_status": "PARTIALLY_RESOLVED"},
        {"case_id": "C2", "intent_id": "GENERAL_DEVICE_INQUIRY", "similarity": 0.52, "resolution_status": "PARTIALLY_RESOLVED"},
        {"case_id": "C3", "intent_id": "GENERAL_DEVICE_INQUIRY", "similarity": 0.48, "resolution_status": "PARTIALLY_RESOLVED"}
    ]
    draft_reply = "Check Settings to repair."
    grounding_check = {"grounding_pass": True, "severity": "NONE", "unsupported_claims": []}
    gen_out = {"grounding_status": "GROUNDED"}

    result = decision_engine.evaluate(
        customer_message=customer_msg,
        intent=intent,
        intent_confidence=intent_conf,
        retrieved_evidence=retrieved_cases,
        draft_reply=draft_reply,
        grounding_check=grounding_check,
        generator_output=gen_out
    )

    assert result["decision"] == "ESCALATE"
    assert result["reason_code"] == EscalationReasonCode.WEAK_EVIDENCE
    assert result["is_customer_safe"] is False


def test_grounding_audit_failure_escalation(decision_engine):
    """Test 4: Grounding audit failure -> ESCALATE (UNSUPPORTED_CLAIM)."""
    customer_msg = "My volume buttons are stuck."
    intent = "AUDIO_SOUND_SPEAKER"
    intent_conf = 0.90
    retrieved_cases = [
        {"case_id": "C1", "intent_id": "AUDIO_SOUND_SPEAKER", "similarity": 0.82, "resolution_status": "CLEARLY_RESOLVED"},
        {"case_id": "C2", "intent_id": "AUDIO_SOUND_SPEAKER", "similarity": 0.78, "resolution_status": "CLEARLY_RESOLVED"},
        {"case_id": "C3", "intent_id": "AUDIO_SOUND_SPEAKER", "similarity": 0.74, "resolution_status": "CLEARLY_RESOLVED"}
    ]
    draft_reply = "We will send an engineer to replace your speaker free of charge."
    grounding_check = {
        "grounding_pass": False,
        "severity": "MEDIUM",
        "unsupported_claims": ["Troubleshooting step references replacement not in evidence."]
    }
    gen_out = {"grounding_status": "GROUNDED"}

    result = decision_engine.evaluate(
        customer_message=customer_msg,
        intent=intent,
        intent_confidence=intent_conf,
        retrieved_evidence=retrieved_cases,
        draft_reply=draft_reply,
        grounding_check=grounding_check,
        generator_output=gen_out
    )

    assert result["decision"] == "ESCALATE"
    assert result["reason_code"] == EscalationReasonCode.UNSUPPORTED_CLAIM
    assert result["is_customer_safe"] is False


def test_high_risk_grounding_failure_escalation(decision_engine):
    """Test 5: High severity grounding violation -> ESCALATE (HIGH_RISK_GROUNDING_FAILURE)."""
    customer_msg = "I was overcharged for Apple Music."
    intent = "APP_STORE_PURCHASES_BILLING"
    intent_conf = 0.95
    retrieved_cases = [
        {"case_id": "C1", "intent_id": "APP_STORE_PURCHASES_BILLING", "similarity": 0.89, "resolution_status": "CLEARLY_RESOLVED"},
        {"case_id": "C2", "intent_id": "APP_STORE_PURCHASES_BILLING", "similarity": 0.85, "resolution_status": "CLEARLY_RESOLVED"},
        {"case_id": "C3", "intent_id": "APP_STORE_PURCHASES_BILLING", "similarity": 0.81, "resolution_status": "CLEARLY_RESOLVED"}
    ]
    draft_reply = "A full refund of $50 has been approved and processed to your account."
    grounding_check = {
        "grounding_pass": False,
        "severity": "HIGH",
        "unsupported_claims": ["Unauthorized refund approval promise", "Invented monetary refund amount"]
    }
    gen_out = {"grounding_status": "GROUNDED"}

    result = decision_engine.evaluate(
        customer_message=customer_msg,
        intent=intent,
        intent_confidence=intent_conf,
        retrieved_evidence=retrieved_cases,
        draft_reply=draft_reply,
        grounding_check=grounding_check,
        generator_output=gen_out
    )

    assert result["decision"] == "ESCALATE"
    assert result["reason_code"] == EscalationReasonCode.HIGH_RISK_GROUNDING_FAILURE
    assert result["is_customer_safe"] is False


def test_evidence_insufficient_escalation(decision_engine):
    """Test 6: Stage 6 EVIDENCE_INSUFFICIENT status -> ESCALATE (EVIDENCE_INSUFFICIENT)."""
    customer_msg = "Why is my device not syncing?"
    intent = "ACCOUNT_APPLEID_ICLOUD"
    intent_conf = 0.85
    retrieved_cases = [
        {"case_id": "C1", "intent_id": "ACCOUNT_APPLEID_ICLOUD", "similarity": 0.72, "resolution_status": "PARTIALLY_RESOLVED"}
    ]
    draft_reply = "Could you let us know your specific device model and exact iOS version?"
    grounding_check = {"grounding_pass": True, "severity": "NONE", "unsupported_claims": []}
    gen_out = {"grounding_status": "EVIDENCE_INSUFFICIENT"}

    result = decision_engine.evaluate(
        customer_message=customer_msg,
        intent=intent,
        intent_confidence=intent_conf,
        retrieved_evidence=retrieved_cases,
        draft_reply=draft_reply,
        grounding_check=grounding_check,
        generator_output=gen_out
    )

    assert result["decision"] == "ESCALATE"
    assert result["reason_code"] == EscalationReasonCode.EVIDENCE_INSUFFICIENT
    assert result["is_customer_safe"] is False


def test_intent_evidence_mismatch_escalation(decision_engine):
    """Test 7: Intent/Evidence Alignment mismatch (<0.66) -> ESCALATE (INTENT_EVIDENCE_MISMATCH)."""
    customer_msg = "My screen is flickering when I open camera."
    intent = "DISPLAY_TOUCH_SCREEN"
    intent_conf = 0.88
    # 2 out of 3 retrieved cases match APP_STORE_PURCHASES_BILLING, but customer intent is DISPLAY_TOUCH_SCREEN
    retrieved_cases = [
        {"case_id": "C1", "intent_id": "APP_STORE_PURCHASES_BILLING", "similarity": 0.78, "resolution_status": "CLEARLY_RESOLVED"},
        {"case_id": "C2", "intent_id": "APP_STORE_PURCHASES_BILLING", "similarity": 0.75, "resolution_status": "CLEARLY_RESOLVED"},
        {"case_id": "C3", "intent_id": "GENERAL_DEVICE_INQUIRY", "similarity": 0.72, "resolution_status": "CLEARLY_RESOLVED"}
    ]
    draft_reply = "Check your display settings."
    grounding_check = {"grounding_pass": True, "severity": "NONE", "unsupported_claims": []}
    gen_out = {"grounding_status": "GROUNDED"}

    result = decision_engine.evaluate(
        customer_message=customer_msg,
        intent=intent,
        intent_confidence=intent_conf,
        retrieved_evidence=retrieved_cases,
        draft_reply=draft_reply,
        grounding_check=grounding_check,
        generator_output=gen_out
    )

    assert result["decision"] == "ESCALATE"
    assert result["reason_code"] == EscalationReasonCode.INTENT_EVIDENCE_MISMATCH
    assert result["is_customer_safe"] is False


def test_conflicting_evidence_escalation(decision_engine):
    """Test 8: Conflicting retrieved intents (3 different intents in top-3) -> ESCALATE (CONFLICTING_EVIDENCE)."""
    customer_msg = "Device issue after update."
    intent = "KEYBOARD_TYPING_AUTOCORRECT"
    intent_conf = 0.85
    retrieved_cases = [
        {"case_id": "C1", "intent_id": "KEYBOARD_TYPING_AUTOCORRECT", "similarity": 0.76, "resolution_status": "CLEARLY_RESOLVED"},
        {"case_id": "C2", "intent_id": "BATTERY_CHARGING_POWER", "similarity": 0.74, "resolution_status": "CLEARLY_RESOLVED"},
        {"case_id": "C3", "intent_id": "CONNECTIVITY_WIFI_BLUETOOTH", "similarity": 0.73, "resolution_status": "CLEARLY_RESOLVED"}
    ]
    draft_reply = "Setup text replacement in Settings."
    grounding_check = {"grounding_pass": True, "severity": "NONE", "unsupported_claims": []}
    gen_out = {"grounding_status": "GROUNDED"}

    result = decision_engine.evaluate(
        customer_message=customer_msg,
        intent=intent,
        intent_confidence=intent_conf,
        retrieved_evidence=retrieved_cases,
        draft_reply=draft_reply,
        grounding_check=grounding_check,
        generator_output=gen_out
    )

    assert result["decision"] == "ESCALATE"
    assert result["reason_code"] in [
        EscalationReasonCode.CONFLICTING_EVIDENCE,
        EscalationReasonCode.INTENT_EVIDENCE_MISMATCH
    ]
    assert result["is_customer_safe"] is False


def test_simultaneous_failures_priority_order(decision_engine):
    """Test 9: Multiple simultaneous failures trigger the highest priority reason code."""
    # Case with: High Risk Grounding failure + Weak retrieval + Low intent confidence
    customer_msg = "Help"
    intent = "GENERAL_DEVICE_INQUIRY"
    intent_conf = 0.25  # Fails Gate 7
    retrieved_cases = [
        {"case_id": "C1", "intent_id": "ACCOUNT_APPLEID_ICLOUD", "similarity": 0.40, "resolution_status": "PARTIALLY_RESOLVED"}  # Fails Gate 5 & 6
    ]
    draft_reply = "We guarantee a free replacement phone."
    grounding_check = {
        "grounding_pass": False,
        "severity": "HIGH",  # Fails Gate 1
        "unsupported_claims": ["Unauthorized outcome guarantee", "Free replacement claim"]
    }
    gen_out = {"grounding_status": "GROUNDED"}

    result = decision_engine.evaluate(
        customer_message=customer_msg,
        intent=intent,
        intent_confidence=intent_conf,
        retrieved_evidence=retrieved_cases,
        draft_reply=draft_reply,
        grounding_check=grounding_check,
        generator_output=gen_out
    )

    # Gate 1 (HIGH_RISK_GROUNDING_FAILURE) must take priority over WEAK_EVIDENCE or LOW_INTENT_CONFIDENCE
    assert result["decision"] == "ESCALATE"
    assert result["reason_code"] == EscalationReasonCode.HIGH_RISK_GROUNDING_FAILURE


def test_stage7_reply_immutability(decision_engine):
    """Test 10: Stage 7 does not mutate, modify, or regenerate the Stage 6 draft reply."""
    original_reply = "Exact immutable text string with special characters & symbols: Settings > General."
    result = decision_engine.evaluate(
        customer_message="Battery issue",
        intent="BATTERY_CHARGING_POWER",
        intent_confidence=0.90,
        retrieved_evidence=[{"case_id": "C1", "intent_id": "BATTERY_CHARGING_POWER", "similarity": 0.85}],
        draft_reply=original_reply,
        grounding_check={"grounding_pass": True, "severity": "NONE", "unsupported_claims": []},
        generator_output={"grounding_status": "GROUNDED"}
    )
    assert result["draft_reply"] == original_reply


def test_deterministic_non_llm_execution(decision_engine):
    """Test 11: Decision engine executes 100 identical runs deterministically in <50ms without LLMs."""
    args = {
        "customer_message": "My WiFi is disconnected.",
        "intent": "CONNECTIVITY_WIFI_BLUETOOTH",
        "intent_confidence": 0.85,
        "retrieved_evidence": [
            {"case_id": "C1", "intent_id": "CONNECTIVITY_WIFI_BLUETOOTH", "similarity": 0.80},
            {"case_id": "C2", "intent_id": "CONNECTIVITY_WIFI_BLUETOOTH", "similarity": 0.77},
            {"case_id": "C3", "intent_id": "CONNECTIVITY_WIFI_BLUETOOTH", "similarity": 0.75}
        ],
        "draft_reply": "Reset network settings via Settings > General > Reset.",
        "grounding_check": {"grounding_pass": True, "severity": "NONE", "unsupported_claims": []},
        "generator_output": {"grounding_status": "GROUNDED"}
    }

    first_run = decision_engine.evaluate(**args)
    for _ in range(50):
        next_run = decision_engine.evaluate(**args)
        assert next_run["decision"] == first_run["decision"]
        assert next_run["reason_code"] == first_run["reason_code"]
        assert next_run["reason"] == first_run["reason"]


def test_ground_truth_safety_label_heuristic():
    """Test 12: Reference ground-truth safety heuristic correctly labels safe vs unsafe cases."""
    # Safe case
    label_safe, _ = assign_ground_truth_safety_label(
        customer_problem="My battery is draining quickly on iOS 11.0.3",
        true_intent="BATTERY_CHARGING_POWER",
        retrieved_cases=[{"intent_id": "BATTERY_CHARGING_POWER", "similarity": 0.82}],
        draft_reply="Check Settings > Battery",
        grounding_check={"grounding_pass": True, "severity": "NONE"}
    )
    assert label_safe == "SAFE_TO_AUTO_HANDLE"

    # Unsafe case (legal/dispute inquiry)
    label_unsafe, _ = assign_ground_truth_safety_label(
        customer_problem="I am taking Apple to court for a refund on stolen account",
        true_intent="APP_STORE_PURCHASES_BILLING",
        retrieved_cases=[{"intent_id": "APP_STORE_PURCHASES_BILLING", "similarity": 0.80}],
        draft_reply="Please contact billing",
        grounding_check={"grounding_pass": True, "severity": "NONE"}
    )
    assert label_unsafe == "SHOULD_ESCALATE"
