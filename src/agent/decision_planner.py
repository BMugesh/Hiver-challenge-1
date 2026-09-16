"""
SupportDNA Agent — Step 5: Agent Decision & Action Planning
===========================================================
Synthesizes Query Understanding, Risk Analysis, Evidence Quality, Layer 3 Escalation
Triggers, and Layer 4 Safety Constraints into an explicit action plan.

Actions:
- ANSWER
- GUIDE
- CLARIFY
- ESCALATE
- SAFE_REFUSAL
- SAFE_REFUSAL_AND_ESCALATE

Response Modes:
- DIRECT_ANSWER
- EVIDENCE_BACKED_GUIDANCE
- CLARIFICATION
- SAFE_ESCALATION
- INSUFFICIENT_EVIDENCE
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import re


@dataclass
class AgentDecisionResult:
    action: str            # ANSWER | GUIDE | CLARIFY | ESCALATE | SAFE_REFUSAL | SAFE_REFUSAL_AND_ESCALATE
    response_mode: str     # DIRECT_ANSWER | EVIDENCE_BACKED_GUIDANCE | CLARIFICATION | SAFE_ESCALATION | INSUFFICIENT_EVIDENCE
    reason_code: str
    reason: str
    is_customer_safe: bool
    escalation_trigger: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "response_mode": self.response_mode,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "is_customer_safe": bool(self.is_customer_safe),
            "escalation_trigger": self.escalation_trigger
        }


def plan_agent_action(
    query: str,
    query_understanding: Any,
    risk_result: Any,
    evidence_judge_result: Any
) -> AgentDecisionResult:
    """
    Formulate the agent's explicit operational action and response mode.
    Balances evidence strength with customer safety.
    """
    lower = query.lower()

    # 1. Gate: Adversarial Prompt Injection or Forced Output Manipulation
    is_inj = getattr(risk_result, "is_prompt_injection", False) or risk_result.injection_status in ["BLOCKED", "SUSPICIOUS"]
    if is_inj:
        sec_intent = getattr(risk_result, "security_intent", "PROMPT_INJECTION")
        if sec_intent == "APPROVAL_MANIPULATION":
            reason_code = "APPROVAL_MANIPULATION_BLOCKED"
        elif sec_intent == "POLICY_OVERRIDE":
            reason_code = "POLICY_OVERRIDE_BLOCKED"
        elif sec_intent == "FAKE_AUTHORITY":
            reason_code = "FAKE_AUTHORITY_BLOCKED"
        elif sec_intent == "SYSTEM_PROMPT_EXTRACTION":
            reason_code = "PROMPT_EXTRACTION_BLOCKED"
        else:
            reason_code = "PROMPT_INJECTION_BLOCKED"

        return AgentDecisionResult(
            action="SAFE_REFUSAL_AND_ESCALATE",
            response_mode="SAFE_ESCALATION",
            reason_code=reason_code,
            reason=f"Adversarial security manipulation intercepted ({sec_intent}): {risk_result.reason}",
            is_customer_safe=False,
            escalation_trigger="SECURITY_OVERRIDE_ATTEMPT"
        )

    # 3. Gate: High-Risk Escalation Triggers (Layer 3 & 4 Knowledge)
    # Physical hardware damage, screen cracked, battery swelling
    if any(kw in lower for kw in ["screen is cracked", "cracked screen", "shattered screen", "battery swelling", "swollen battery", "melted", "burned"]):
        return AgentDecisionResult(
            action="ESCALATE",
            response_mode="SAFE_ESCALATION",
            reason_code="HARDWARE_REPAIR_ESCALATION",
            reason="Physical hardware damage requires authorized service inspection and repair triage.",
            is_customer_safe=True,
            escalation_trigger="PHYSICAL_HARDWARE_DAMAGE"
        )

    # Private credentials / account takeover
    if any(kw in lower for kw in ["apple id locked", "activation lock", "account hacked", "credit card charged twice"]):
        return AgentDecisionResult(
            action="ESCALATE",
            response_mode="SAFE_ESCALATION",
            reason_code="ACCOUNT_SECURITY_ESCALATION",
            reason="Account security, identity verification, and financial disputes must be handled via secure human support channels.",
            is_customer_safe=True,
            escalation_trigger="PRIVATE_CREDENTIALS_AND_BILLING"
        )

    # 4. Gate: Unanswerable / Vague Query (Clarification Required)
    if not query_understanding.can_answer_now:
        return AgentDecisionResult(
            action="CLARIFY",
            response_mode="CLARIFICATION",
            reason_code="VAGUE_QUERY_NEEDS_CLARIFICATION",
            reason="Customer inquiry lacks actionable symptoms. A targeted diagnostic question is required.",
            is_customer_safe=True,
            escalation_trigger=None
        )

    # 5. Gate: Multi-Issue Queries
    if query_understanding.is_multi_issue:
        # If one issue is hardware damage or severe, escalate
        if any(kw in lower for kw in ["cracked", "broken", "dropped in water"]):
            return AgentDecisionResult(
                action="ESCALATE",
                response_mode="SAFE_ESCALATION",
                reason_code="MULTI_ISSUE_HARDWARE_ESCALATE",
                reason="Customer states multiple issues including physical hardware damage.",
                is_customer_safe=True,
                escalation_trigger="MULTI_ISSUE_COMPLEXITY"
            )
        # Otherwise, if evidence is strong for primary symptom, guide with triage
        if evidence_judge_result.overall_quality in ["STRONG", "MODERATE"]:
            return AgentDecisionResult(
                action="GUIDE",
                response_mode="EVIDENCE_BACKED_GUIDANCE",
                reason_code="MULTI_ISSUE_TRIAGE_GUIDANCE",
                reason="Multiple issues detected. Providing grounded troubleshooting for primary symptom with sequential triage.",
                is_customer_safe=True,
                escalation_trigger=None
            )
        else:
            return AgentDecisionResult(
                action="ESCALATE",
                response_mode="SAFE_ESCALATION",
                reason_code="MULTI_ISSUE_UNCERTAIN_ESCALATE",
                reason="Multiple complex issues detected without unified resolution evidence.",
                is_customer_safe=False,
                escalation_trigger="MULTI_ISSUE_DIVERGENCE"
            )

    # 6. Gate: Evidence Quality Evaluation
    ev_qual = evidence_judge_result.overall_quality

    if ev_qual in ["INSUFFICIENT", "IRRELEVANT"]:
        return AgentDecisionResult(
            action="ESCALATE",
            response_mode="INSUFFICIENT_EVIDENCE",
            reason_code="EVIDENCE_IRRELEVANT_OR_INSUFFICIENT",
            reason="Knowledge retrieval produced no relevant or sufficient troubleshooting precedent.",
            is_customer_safe=False,
            escalation_trigger="INSUFFICIENT_RESOLUTION_EVIDENCE"
        )

    if ev_qual == "CONFLICTING":
        return AgentDecisionResult(
            action="ESCALATE",
            response_mode="SAFE_ESCALATION",
            reason_code="CONFLICTING_HISTORICAL_EVIDENCE",
            reason="Historical evidence presents contradictory guidance across divergent intents.",
            is_customer_safe=False,
            escalation_trigger="CONFLICTING_EVIDENCE"
        )

    if ev_qual == "WEAK":
        return AgentDecisionResult(
            action="ESCALATE",
            response_mode="INSUFFICIENT_EVIDENCE",
            reason_code="WEAK_HISTORICAL_EVIDENCE",
            reason=f"Retrieved evidence similarity ({evidence_judge_result.top_similarity:.4f}) is too weak to ensure safe autonomous resolution.",
            is_customer_safe=False,
            escalation_trigger="WEAK_EVIDENCE_THRESHOLD"
        )

    # 7. Standard Strong / Moderate Grounded Guidance
    return AgentDecisionResult(
        action="GUIDE",
        response_mode="EVIDENCE_BACKED_GUIDANCE",
        reason_code="STRONG_GROUNDED_EVIDENCE",
        reason="Retrieved evidence strongly supports actionable, verified troubleshooting.",
        is_customer_safe=True,
        escalation_trigger=None
    )
