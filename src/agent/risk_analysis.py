"""
SupportDNA Agent — Step 2: Risk & Security Intent Analysis
==========================================================
Audits incoming customer inquiries for adversarial manipulation, prompt injection,
policy overrides, and unauthorized claim extraction (e.g. refunds, renewal approvals).

Powered by the dedicated Semantic SecurityDetector, completely decoupled from
the 11-intent business problem taxonomy.
"""

from dataclasses import dataclass
from typing import Dict, Any, Union, List

from src.agent.security_detector import detect_security_intent, SecurityAnalysisResult


@dataclass
class RiskAnalysisResult:
    security_intent: str
    is_prompt_injection: bool
    security_confidence: float
    risk_level: str                  # "LOW" | "MEDIUM" | "HIGH"
    reason: str
    injection_status: str = "NONE"   # "NONE" | "SUSPICIOUS" | "BLOCKED" (backward-compatible)
    attack_type: str = "NONE"
    is_hard_negative: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "security_intent": self.security_intent,
            "is_prompt_injection": bool(self.is_prompt_injection),
            "security_confidence": round(float(self.security_confidence), 4),
            "risk_level": self.risk_level,
            "reason": self.reason,
            "injection_status": self.injection_status,
            "attack_type": self.attack_type
        }


def analyze_risk(customer_text: Union[str, List[Dict[str, str]]]) -> RiskAnalysisResult:
    """
    Evaluate customer inquiry or multi-turn conversation for prompt injection
    and security intent using the semantic SecurityDetector.
    """
    sec_res: SecurityAnalysisResult = detect_security_intent(customer_text)

    # Map injection_status for backward compatibility with existing tests/verifier
    if sec_res.is_prompt_injection:
        inj_status = "BLOCKED"
    elif sec_res.risk_level == "MEDIUM":
        inj_status = "SUSPICIOUS"
    else:
        inj_status = "NONE"

    return RiskAnalysisResult(
        security_intent=sec_res.security_intent,
        is_prompt_injection=sec_res.is_prompt_injection,
        security_confidence=sec_res.security_confidence,
        risk_level=sec_res.risk_level,
        reason=sec_res.reason,
        injection_status=inj_status,
        attack_type=sec_res.attack_type,
        is_hard_negative=sec_res.is_hard_negative
    )
