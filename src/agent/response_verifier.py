"""
SupportDNA Agent — Step 9: Response Verification
=================================================
Audits generated responses against 9 quality and safety dimensions:
1. RELEVANCE
2. GROUNDEDNESS
3. ACTIONABILITY
4. RESOLUTION_PATTERN_USAGE
5. UNNECESSARY_CLARIFICATION
6. IRRELEVANT_EVIDENCE_USAGE
7. SAFETY
8. INJECTION_RESISTANCE
9. ESCALATION_CONSISTENCY

Output Schema:
{
    "verification_pass": true,
    "relevance": 1.0,
    "groundedness": 1.0,
    "actionability": 1.0,
    "resolution_pattern_used": true,
    "unnecessary_clarification": false,
    "irrelevant_evidence_used": false,
    "safety_pass": true,
    "injection_resistance_pass": true,
    "escalation_consistent": true,
    "failure_reasons": []
}
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class ResponseVerificationResult:
    verification_pass: bool
    relevance: float
    groundedness: float
    actionability: float
    resolution_pattern_used: bool
    unnecessary_clarification: bool
    irrelevant_evidence_used: bool
    safety_pass: bool
    injection_resistance_pass: bool
    escalation_consistent: bool
    failure_reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verification_pass": bool(self.verification_pass),
            "relevance": round(float(self.relevance), 2),
            "groundedness": round(float(self.groundedness), 2),
            "actionability": round(float(self.actionability), 2),
            "resolution_pattern_used": bool(self.resolution_pattern_used),
            "unnecessary_clarification": bool(self.unnecessary_clarification),
            "irrelevant_evidence_used": bool(self.irrelevant_evidence_used),
            "safety_pass": bool(self.safety_pass),
            "injection_resistance_pass": bool(self.injection_resistance_pass),
            "escalation_consistent": bool(self.escalation_consistent),
            "failure_reasons": list(self.failure_reasons)
        }


class ResponseVerifier:
    """
    9-dimension auditor verifying response quality, safety, and compliance with evidence.
    """

    def verify(
        self,
        query: str,
        query_understanding: Any,
        risk_result: Any,
        evidence_judge_result: Any,
        decision_result: Any,
        response_plan: Any,
        generated_reply: str,
        conversation_context: Optional[Dict[str, Any]] = None
    ) -> ResponseVerificationResult:
        """Run verification audits on the candidate reply including conversation memory consistency."""
        failure_reasons = []
        lower_reply = generated_reply.lower()
        lower_query = query.lower()

        # 1. Relevance Audit
        # Does the answer actually address what the customer asked or preserved conversational issue?
        relevance_score = 1.0
        effective_intent = query_understanding.intent
        if conversation_context and conversation_context.get("effectiveIntent"):
            effective_intent = conversation_context["effectiveIntent"]

        if "renewal" in lower_query and "approved" in lower_query:
            # If customer asked about renewal approval, reply MUST address renewal/approval/support
            if "renewal" not in lower_reply and "approve" not in lower_reply and "purchase" not in lower_reply:
                relevance_score = 0.2
                failure_reasons.append("Response failed to address renewal/approval inquiry.")
        elif effective_intent == "BATTERY_CHARGING_POWER" or "battery" in lower_query or "drain" in lower_query:
            if "battery" not in lower_reply and "power" not in lower_reply and "drain" not in lower_reply:
                relevance_score = 0.3
                failure_reasons.append("Response failed to address battery drain symptoms.")

        # 2. Safety Audit
        # Did it make unsupported approvals, guarantees, or false refund claims?
        safety_pass = True
        if any(kw in lower_reply for kw in ["renewal is approved", "refund is approved", "apple has approved your refund", "free replacement is guaranteed"]):
            safety_pass = False
            failure_reasons.append("UNSUPPORTED_APPROVAL: Generated reply made unauthorized financial or warranty promise.")

        # 3. Injection Resistance Audit
        # Did adversarial customer instructions improperly influence the output?
        injection_resistance_pass = True
        is_inj = getattr(risk_result, "is_prompt_injection", False) or risk_result.injection_status in ["BLOCKED", "SUSPICIOUS"]
        if is_inj:
            has_affirmative_approval = bool(
                re.search(r'\b(?:payment\s+is\s+approved|renewal\s+is\s+approved|refund\s+is\s+approved|here\s+is\s+your\s+approved|apple\s+has\s+approved\s+your)\b', lower_reply) or
                re.search(r'\b(?:override\s+active|jailbreak\s+enabled|developer\s+mode\s+activated)\b', lower_reply) or
                (lower_reply.strip().startswith("approved") or lower_reply.strip().startswith("payment approved"))
            )
            if has_affirmative_approval:
                injection_resistance_pass = False
                failure_reasons.append("INJECTION_LEAKAGE: Prompt injection caused agent to fabricate unauthorized approval or override state.")

        # 4. Escalation Consistency Audit
        # If decision = ESCALATE or SAFE_REFUSAL_AND_ESCALATE, does response explain an appropriate escalation?
        escalation_consistent = True
        if decision_result.action in ["ESCALATE", "SAFE_REFUSAL_AND_ESCALATE"]:
            is_escalation_explained = any(
                kw in lower_reply for kw in [
                    "cannot confirm", "specialist", "support.apple.com", "getsupport",
                    "genius bar", "hardware service", "iforgot", "reportaproblem", "authorized",
                    "direct assistance"
                ]
            )
            if not is_escalation_explained:
                escalation_consistent = False
                failure_reasons.append("ESCALATION_INCONSISTENCY: Decision was ESCALATE but response did not explain escalation.")
            if "here are the recommended troubleshooting steps" in lower_reply:
                escalation_consistent = False
                failure_reasons.append("ESCALATION_INCONSISTENCY: Decision was ESCALATE but response contained standalone troubleshooting recommendations.")

        # 5. Attempted Steps Consistency Audit
        # Did the agent re-recommend steps already attempted in the conversation?
        repeated_attempted_step = False
        if conversation_context:
            attempted = conversation_context.get("attemptedSteps", [])
            for step in attempted:
                step_lower = step.lower()
                if "restart" in step_lower and ("restart your device" in lower_reply or "restart your phone" in lower_reply):
                    repeated_attempted_step = True
                    failure_reasons.append(f"REPEATED_ATTEMPTED_STEP: Re-recommended '{step}' when customer already attempted it.")

        # 6. Unnecessary Clarification Audit
        # Did agent ask for device model or iOS version when general troubleshooting could be given?
        unnecessary_clarification = False
        if effective_intent == "BATTERY_CHARGING_POWER" and evidence_judge_result.overall_quality in ["STRONG", "MODERATE"]:
            if "which device model and ios version" in lower_reply or "what model of iphone" in lower_reply:
                unnecessary_clarification = True
                failure_reasons.append("UNNECESSARY_CLARIFICATION: Asked for model/iOS version instead of providing battery usage troubleshooting.")

        # 7. Irrelevant Evidence Usage Audit
        irrelevant_evidence_used = False
        if evidence_judge_result.overall_quality == "IRRELEVANT" and decision_result.action == "GUIDE":
            irrelevant_evidence_used = True
            failure_reasons.append("IRRELEVANT_EVIDENCE_USAGE: Guided from irrelevant evidence.")

        # 8. Resolution Pattern Usage Audit
        resolution_pattern_used = True
        if evidence_judge_result.overall_quality == "STRONG" and decision_result.action == "GUIDE":
            if "battery" in lower_query and "settings" not in lower_reply:
                resolution_pattern_used = False
                failure_reasons.append("RESOLUTION_PATTERN_UNUSED: Strong evidence existed but Settings navigation was missing.")

        # 9. Actionability & Groundedness Scores
        actionability_score = 1.0 if any(kw in lower_reply for kw in ["settings", "check", "visit", "go to", "restart", "toggle"]) else 0.5
        groundedness_score = 1.0 if safety_pass and not irrelevant_evidence_used else 0.0

        # Overall Verification Pass
        verification_pass = (
            safety_pass and
            injection_resistance_pass and
            escalation_consistent and
            not unnecessary_clarification and
            not irrelevant_evidence_used and
            not repeated_attempted_step and
            relevance_score >= 0.7
        )

        return ResponseVerificationResult(
            verification_pass=verification_pass,
            relevance=relevance_score,
            groundedness=groundedness_score,
            actionability=actionability_score,
            resolution_pattern_used=resolution_pattern_used,
            unnecessary_clarification=unnecessary_clarification,
            irrelevant_evidence_used=irrelevant_evidence_used,
            safety_pass=safety_pass,
            injection_resistance_pass=injection_resistance_pass,
            escalation_consistent=escalation_consistent,
            failure_reasons=failure_reasons
        )
