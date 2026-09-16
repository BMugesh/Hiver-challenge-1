"""
SupportDNA Security Intent & Prompt Injection Detector
=======================================================
Implements a dedicated, semantic security classifier that operates completely
independently from the 11-intent business problem taxonomy.

Labels:
- NONE
- PROMPT_INJECTION
- PROMPT_MANIPULATION
- FAKE_SYSTEM_INSTRUCTION
- FAKE_AUTHORITY
- APPROVAL_MANIPULATION
- POLICY_OVERRIDE
- SYSTEM_PROMPT_EXTRACTION
- OTHER_SUSPICIOUS_INSTRUCTION

Core Philosophy:
Detect semantic behavior, not mere keyword presence.
Distinguishes customer-controlled override directives from legitimate customer
reports, error message citations, past-tense personal actions, and billing inquiries.
"""

import re
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SECURITY_TRAIN_FILE = PROJECT_ROOT / "data" / "security" / "prompt_injection" / "prompt_injection_train.jsonl"


@dataclass
class SecurityAnalysisResult:
    """Standardized output structure for Security Classification."""
    security_intent: str             # NONE | PROMPT_INJECTION | PROMPT_MANIPULATION | ...
    is_prompt_injection: bool        # True | False
    security_confidence: float       # 0.0 - 1.0
    risk_level: str                  # LOW | MEDIUM | HIGH
    reason: str
    attack_type: str = "NONE"
    is_hard_negative: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "security_intent": self.security_intent,
            "is_prompt_injection": bool(self.is_prompt_injection),
            "security_confidence": round(float(self.security_confidence), 4),
            "risk_level": self.risk_level,
            "reason": self.reason
        }


# Quotation and error reporting patterns (indicates benign external citation)
ERROR_CITATION_PATTERNS = [
    r'\b(?:the\s+)?error\s+(?:message\s+|prompt\s+|code\s+|text\s+)?(?:says|reads|shows|stated|states)\b',
    r'\bthe\s+(?:dialog|screen|window|notification|system)\s+says\b',
    r'\bquoted\s+text\b',
    r'\bwhat\s+does\s+["\'].*["\']\s+mean\b',
    r'\bwhat\s+does\s+(?:the\s+)?(?:error|message)\s+mean\b',
]

# 1st person past personal actions (customer did something in the past, not commanding the AI)
PAST_PERSONAL_ACTION_PATTERNS = [
    r'\bi\s+(?:accidentally\s+|mistakenly\s+)?(?:ignored|overrode|overrode|bypassed|forgot)\s+(?:the\s+|my\s+)?(?:instructions|guide|manual|setting|notification|update|prompt)\b',
    r'\bi\s+tried\s+to\s+override\b',
    r'\bi\s+followed\s+all\s+(?:the\s+)?instructions\b',
]

# Legitimate customer questions about external approval / policy / admin
BENIGN_INQUIRY_PATTERNS = [
    r'\bwhy\s+(?:was|is|did)\s+(?:my\s+|the\s+)?(?:payment|card|purchase|order|transaction|charge|refund)\s+(?:not\s+|wasn\'?t\s+)?(?:approved|authorized|denied|declined|rejected)\b',
    r'\bwhy\s+does\s+(?:the\s+)?(?:payment|billing|system|app\s+store)\s+require\s+approval\b',
    r'\b(?:my\s+)?payment\s+needs\s+approval\b',
    r'\bthe\s+payment\s+was\s+not\s+approved\b',
    r'\bthe\s+system\s+says\s+(?:payment\s+)?approval\s+is\s+required\b',
    r'\bmy\s+refund\s+was\s+denied\b',
    r'\bhow\s+(?:can|do)\s+i\s+appeal\s+(?:it|my\s+refund|a\s+denial)\b',
    r'\bhow\s+do\s+i\s+(?:contact|reach)\s+(?:the|my)\s+administrator\b',
    r'\bwhat\s+is\s+(?:apple\'?s\s+)?(?:return|refund|warranty|replacement)\s+policy\b',
    r'\bhow\s+do\s+i\s+override\s+(?:manual\s+)?(?:dns|network|wifi|screen\s+time|display)\s+settings?\b',
    r'\bwaiting\s+for\s+(?:family\s+sharing\s+|organizer\s+)?approval\b',
    r'\bpending\s+approval\b',
]


class SecurityDetector:
    """
    Semantic Security Detector for Prompt Injection and Behavioral Manipulation.
    Distinguishes agent override directives from customer troubleshooting questions.
    """

    def __init__(self):
        # Semantic directive matchers
        self._init_semantic_matchers()

    def _init_semantic_matchers(self):
        # 1. Direct Instruction Overrides (imperative command targeting AI instructions/rules)
        self.re_direct_override = re.compile(
            r'\b(?:ignore|disregard|forget|bypass|cancel|drop|set\s+aside|override|overrule)\s+'
            r'(?:all\s+|any\s+|prior\s+|previous\s+|your\s+|system\s+|developer\s+|safety\s+|the\s+|above\s+|these\s+)*'
            r'(?:instructions|rules|guidelines|prompts|constraints|programming|commands|safety\s+system|safeguards)\b|'
            r'\boverride\s+your\s+instructions\b',
            re.IGNORECASE
        )

        # 2. Approval & Refund Manipulation (forcing agent to declare, state, or confirm approval)
        self.re_approval_manipulation = re.compile(
            r'\b(?:say|respond\s+with|reply\s+with|output|declare|confirm|state|print)\s+'
            r'[\'"]?(?:payment\s+approved|renewal\s+approved|refund\s+approved|approved|'
            r'order\s+approved|trade-in\s+approved|warranty\s+approved|free\s+iphone|you\s+are\s+hacked)[\'"]?\b|'
            r'\b(?:say|confirm|state|declare)\s+that\s+.*?\b(?:is|has\s+been|was)\s+(?:approved|granted|credited|refunded)\b|'
            r'\bsay\s+[\'"]approved[\'"]\b|'
            r'\bconfirm\s+that\s+.*?\b(?:approved|refunded|granted)\b|'
            r'\bdeclare\s+that\s+my\s+refund\s+appeal\s+is\s+granted\b|'
            r'\bsay\s+that\s+my\s+account\s+has\s+been\s+credited\b',
            re.IGNORECASE
        )

        # 3. Policy Override (commanding agent to bypass or override support/company policy)
        self.re_policy_override = re.compile(
            r'\b(?:override|bypass|disregard|break|ignore|overrule)\s+(?:all\s+|any\s+|prior\s+|previous\s+|the\s+|support\s+|company\s+|apple\s+|applecare\s+|internal\s+)*'
            r'(?:policy|policies|rules|protocol|guidelines|escalation\s+policy|return\s+policy|security\s+restrictions|requirements|inspection\s+requirements|limits)\b|'
            r'\bbypass\s+(?:the\s+)?(?:standard\s+)?(?:escalation\s+policy|verification\s+protocol|security\s+filter)\b',
            re.IGNORECASE
        )

        # 4. Role Hijacking & Jailbreaking
        self.re_role_hijacking = re.compile(
            r'\b(?:you\s+are\s+now|pretend\s+you\s+are|act\s+as|roleplay\s+as|assume\s+the\s+persona\s+of|from\s+now\s+on\s+you\s+are)\s+'
            r'(?:an?\s+)?(?:apple\s+employee|unrestricted|dan|developer\s+mode|jailbroken|freebot|supervisor|technician|robot|executive)\b|'
            r'\b(?:jailbreak|dan\s+mode|unrestricted\s+mode|developer\s+mode\s+activated)\b',
            re.IGNORECASE
        )

        # 5. Fake Authority (user claiming executive/admin identity to coerce compliance)
        self.re_fake_authority = re.compile(
            r'\b(?:you\s+are\s+now\s+the\s+payment\s+administrator|'
            r'i\s+am\s+(?:an?\s+)?(?:authorized\s+)?(?:payment\s+administrator|system\s+administrator|tim\s+cook|vice\s+president|billing\s+director|senior\s+engineer|genius\s+bar\s+lead|regional\s+director|executive\s+assistant)|'
            r'as\s+(?:the\s+)?(?:administrator|director|lead|apple\s+store\s+regional\s+director)|'
            r'speaking\s+as\s+apple\s+(?:legal|corporate|security)|'
            r'this\s+inquiry\s+is\s+from\s+apple\s+legal)\b.*?(?:authorize|confirm|approve|execute|grant|override|order\s+you\s+to|command\s+you\s+to)?|'
            r'\b(?:i\s+order\s+you\s+to|i\s+command\s+you\s+to)\s+(?:approve|confirm|override|execute)\b',
            re.IGNORECASE
        )

        # 6. Fake System Instruction & Indirect Command
        self.re_fake_system = re.compile(
            r'^(?:system:|<system>|\[system\s+message\]|system\s+alert:|system\s+instruction:|system\s+notice:|system\s+override\b)|'
            r'\bthe\s+following\s+is\s+a\s+system\s+instruction:\s*payment\s+approved\b|'
            r'\[system:\s*bypass|system\s+command:|<internal_system_directive>|###\s*system\s+update:|'
            r'\bexecute\s+system\s+command\b|'
            r'\bfollow\s+this\s+note\b',
            re.IGNORECASE
        )

        # 7. System Prompt Extraction
        self.re_prompt_extraction = re.compile(
            r'\b(?:reveal|show|print|output|display|leak|repeat)\s+'
            r'(?:all\s+|any\s+|your\s+|the\s+|internal\s+|hidden\s+)*'
            r'(?:system\s+prompt|developer\s+instructions|system\s+message|internal\s+guidelines|hidden\s+constraints|hidden\s+prompt|initial\s+prompt|secret\s+rules)\b|'
            r'\brepeat\s+all\s+text\s+above\s+this\s+line\b',
            re.IGNORECASE
        )

        # 8. Delimiter & Indirect Injection
        self.re_delimiter_injection = re.compile(
            r'[-=]{4,}\s*(?:end\s+user\s+message|begin\s+instruction|system)[\s\S]*?[-=]{4,}|'
            r'```system\b[\s\S]*?```|<instruction_block>',
            re.IGNORECASE
        )

        # 9. Obfuscated override directives (e.g. i.g.n.o.r.e or 1gn0re)
        self.re_obfuscated = re.compile(
            r'(?:i[\s\.\-_]*g[\s\.\-_]*n[\s\.\-_]*o[\s\.\-_]*r[\s\.\-_]*e|'
            r'1gn0re|b[\s\.\-_]*y[\s\.\-_]*p[\s\.\-_]*a[\s\.\-_]*s[\s\.\-_]*s)\s+'
            r'.*?(?:instruction|rule|approved?|payment|refund)\b|'
            r'\b(?:base64|rot13|decode\s+this)\b.*?(?:approve|payment|instructions)',
            re.IGNORECASE
        )

    def is_hard_negative(self, text: str) -> Tuple[bool, str]:
        """
        Detect whether a query is a hard negative: contains attack-like keywords
        (ignore, override, approved, administrator, policy) but is used in a benign,
        declarative, interrogative, or quoting context.
        """
        lower = text.lower().strip()

        # Check 1: Quoted error reporting or explaining
        for pattern in ERROR_CITATION_PATTERNS:
            if re.search(pattern, lower):
                return True, "Customer is citing or inquiring about an external error message or dialog."

        # Check 2: Past personal action description
        for pattern in PAST_PERSONAL_ACTION_PATTERNS:
            if re.search(pattern, lower):
                return True, "Customer is describing their own past action or manual step, not issuing a directive."

        # Check 3: Benign customer support inquiry about approval/policy/billing
        for pattern in BENIGN_INQUIRY_PATTERNS:
            if re.search(pattern, lower):
                return True, "Customer is asking about an external billing status, policy, or system requirement."

        # Check 4: General questions starting with Why, How, What asking about setting or policy
        if re.search(r'^(?:why|how|what|can\s+i|where|is\s+there)\b', lower):
            if not re.search(r'\b(?:ignore\s+all|say\s+["\']?payment\s+approved|you\s+are\s+now)\b', lower):
                if any(w in lower for w in ["approval", "approved", "policy", "administrator", "instructions"]):
                    return True, "Interrogative customer support inquiry containing technical domain terms."

        return False, ""

    def analyze(self, query: Union[str, List[Dict[str, str]]]) -> SecurityAnalysisResult:
        """
        Analyze customer text or multi-turn conversation for security intent and prompt injection.
        """
        # Handle multi-turn conversation input
        customer_text = ""
        context_notes = []

        if isinstance(query, list):
            # Sequence of messages: [{"role": "user"|"assistant", "content": "..."}]
            user_messages = [m["content"] for m in query if m.get("role") == "user"]
            if not user_messages:
                return SecurityAnalysisResult(
                    security_intent="NONE",
                    is_prompt_injection=False,
                    security_confidence=1.0,
                    risk_level="LOW",
                    reason="No customer message present.",
                    attack_type="NONE"
                )
            # Evaluate the most recent user turn with full history awareness
            customer_text = user_messages[-1]
            if len(user_messages) > 1:
                context_notes.append(f"Multi-turn conversation ({len(query)} total turns, {len(user_messages)} user turns).")
        else:
            customer_text = str(query)

        cleaned = customer_text.strip()
        lower = cleaned.lower()

        # Step 1: Semantic Hard Negative Filter (High Priority)
        is_hard_neg, hard_neg_reason = self.is_hard_negative(cleaned)
        if is_hard_neg:
            return SecurityAnalysisResult(
                security_intent="NONE",
                is_prompt_injection=False,
                security_confidence=0.98,
                risk_level="LOW",
                reason=f"Benign customer inquiry (Hard Negative confirmed): {hard_neg_reason}",
                attack_type="NONE",
                is_hard_negative=True
            )

        # Step 2: Semantic Behavior Analysis of Override Directives

        # Fake System Instruction
        if self.re_fake_system.search(cleaned):
            return SecurityAnalysisResult(
                security_intent="FAKE_SYSTEM_INSTRUCTION",
                is_prompt_injection=True,
                security_confidence=0.99,
                risk_level="HIGH",
                reason="Adversarial fake system directive or synthetic instruction header detected.",
                attack_type="fake_system_message"
            )

        # Delimiter / Indirect Injection
        if self.re_delimiter_injection.search(cleaned):
            return SecurityAnalysisResult(
                security_intent="PROMPT_INJECTION",
                is_prompt_injection=True,
                security_confidence=0.96,
                risk_level="HIGH",
                reason="Adversarial delimiter injection or simulated conversation boundary detected.",
                attack_type="delimiter_injection"
            )

        # System Prompt Extraction
        if self.re_prompt_extraction.search(cleaned):
            return SecurityAnalysisResult(
                security_intent="SYSTEM_PROMPT_EXTRACTION",
                is_prompt_injection=True,
                security_confidence=0.97,
                risk_level="HIGH",
                reason="System prompt or internal developer instruction exfiltration attempt detected.",
                attack_type="system_prompt_extraction"
            )

        # Fake Authority Impersonation
        if self.re_fake_authority.search(cleaned):
            return SecurityAnalysisResult(
                security_intent="FAKE_AUTHORITY",
                is_prompt_injection=True,
                security_confidence=0.96,
                risk_level="HIGH",
                reason="Adversarial fake authority or administrator identity assertion detected.",
                attack_type="fake_authority"
            )

        # Role Hijacking & Jailbreak
        if self.re_role_hijacking.search(cleaned):
            return SecurityAnalysisResult(
                security_intent="PROMPT_MANIPULATION",
                is_prompt_injection=True,
                security_confidence=0.96,
                risk_level="HIGH",
                reason="Persona hijacking or adversarial jailbreak attempt detected.",
                attack_type="role_hijacking"
            )

        # Approval Manipulation
        if self.re_approval_manipulation.search(cleaned):
            return SecurityAnalysisResult(
                security_intent="APPROVAL_MANIPULATION",
                is_prompt_injection=True,
                security_confidence=0.98,
                risk_level="HIGH",
                reason="Forced confirmation/approval extraction attempt targeting financial or warranty state.",
                attack_type="approval_manipulation"
            )

        # Policy Override
        if self.re_policy_override.search(cleaned):
            return SecurityAnalysisResult(
                security_intent="POLICY_OVERRIDE",
                is_prompt_injection=True,
                security_confidence=0.95,
                risk_level="HIGH",
                reason="Command attempting to override or bypass support policy or safety restrictions.",
                attack_type="policy_override"
            )

        # Direct Instruction Override
        if self.re_direct_override.search(cleaned):
            # Check if there's also an approval extraction
            if any(w in lower for w in ["payment", "approved", "refund", "renewal"]):
                sec_intent = "APPROVAL_MANIPULATION"
            else:
                sec_intent = "PROMPT_INJECTION"

            return SecurityAnalysisResult(
                security_intent=sec_intent,
                is_prompt_injection=True,
                security_confidence=0.98,
                risk_level="HIGH",
                reason="Direct instruction override directive attempting to nullify system guidelines.",
                attack_type="direct_instruction_override"
            )

        # Obfuscated Manipulation Directive
        if self.re_obfuscated.search(cleaned):
            return SecurityAnalysisResult(
                security_intent="PROMPT_INJECTION",
                is_prompt_injection=True,
                security_confidence=0.92,
                risk_level="HIGH",
                reason="Obfuscated instruction override attempt detected.",
                attack_type="obfuscated_instruction"
            )

        # Step 3: Default Benign Support Request
        return SecurityAnalysisResult(
            security_intent="NONE",
            is_prompt_injection=False,
            security_confidence=0.95,
            risk_level="LOW",
            reason="No prompt injection or adversarial manipulation detected.",
            attack_type="NONE",
            is_hard_negative=False
        )


# Singleton instance for high-speed reuse
_DETECTOR_INSTANCE = None


def get_security_detector() -> SecurityDetector:
    global _DETECTOR_INSTANCE
    if _DETECTOR_INSTANCE is None:
        _DETECTOR_INSTANCE = SecurityDetector()
    return _DETECTOR_INSTANCE


def detect_security_intent(query: Union[str, List[Dict[str, str]]]) -> SecurityAnalysisResult:
    """Public helper function to analyze security intent and prompt injection."""
    detector = get_security_detector()
    return detector.analyze(query)
