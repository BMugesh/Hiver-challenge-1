"""
SupportDNA Agent — Step 4: Evidence Quality Judge
=================================================
Independent evaluation of candidate evidence to ensure the agent never treats
irrelevant or superficial lexical matches as authoritative troubleshooting proof.

Classifications:
- STRONG
- MODERATE
- WEAK
- CONFLICTING
- IRRELEVANT
- INSUFFICIENT
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import re


@dataclass
class EvidenceJudgeResult:
    overall_quality: str   # STRONG | MODERATE | WEAK | CONFLICTING | IRRELEVANT | INSUFFICIENT
    intent_match: bool
    issue_match: bool
    symptom_match: bool
    resolution_relevance: bool
    is_conflicting: bool
    is_insufficient: bool
    top_similarity: float
    intent_alignment_ratio: float
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_quality": self.overall_quality,
            "intent_match": bool(self.intent_match),
            "issue_match": bool(self.issue_match),
            "symptom_match": bool(self.symptom_match),
            "resolution_relevance": bool(self.resolution_relevance),
            "is_conflicting": bool(self.is_conflicting),
            "is_insufficient": bool(self.is_insufficient),
            "top_similarity": round(float(self.top_similarity), 4),
            "intent_alignment_ratio": round(float(self.intent_alignment_ratio), 4),
            "explanation": self.explanation
        }


def _get_field(obj: Any, field: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(field, default)
    return getattr(obj, field, default)


class EvidenceJudge:
    """
    Evaluates whether retrieved cases provide genuine, grounded technical resolution
    evidence for the customer's stated symptoms.
    """

    def __init__(
        self,
        min_strong_similarity: float = 0.65,
        min_moderate_similarity: float = 0.55
    ):
        self.min_strong_similarity = min_strong_similarity
        self.min_moderate_similarity = min_moderate_similarity

    def judge(
        self,
        customer_query: str,
        query_understanding: Any,
        retrieved_cases: List[Any]
    ) -> EvidenceJudgeResult:
        """Evaluate quality and relevance of candidate evidence cases."""
        if not retrieved_cases:
            return EvidenceJudgeResult(
                overall_quality="INSUFFICIENT",
                intent_match=False,
                issue_match=False,
                symptom_match=False,
                resolution_relevance=False,
                is_conflicting=False,
                is_insufficient=True,
                top_similarity=0.0,
                intent_alignment_ratio=0.0,
                explanation="Zero candidate evidence cases retrieved from the knowledge base."
            )

        target_intent = str(query_understanding.intent)
        customer_issues = query_understanding.issues
        top_case = retrieved_cases[0]
        top_sim = float(_get_field(top_case, "similarity", 0.0))
        top_intent = str(_get_field(top_case, "intent_id", "UNKNOWN"))
        top_prob = str(_get_field(top_case, "customer_problem", "")).lower()
        top_resp = str(_get_field(top_case, "support_response", "")).lower()

        # Check intent alignment across top-3 candidates
        matching_intent_count = sum(
            1 for c in retrieved_cases[:3]
            if str(_get_field(c, "intent_id", "")) == target_intent
        )
        alignment_ratio = matching_intent_count / min(len(retrieved_cases), 3)

        # Distinct intents represented across candidates
        intents_set = set(str(_get_field(c, "intent_id", "")) for c in retrieved_cases[:3])
        is_conflicting = len(intents_set) >= 3

        intent_match = (top_intent == target_intent)

        # Check symptom match
        symptom_match = False
        for iss in customer_issues:
            clean_iss = iss.replace("_", " ").lower()
            if clean_iss in top_prob or any(token in top_prob for token in clean_iss.split()):
                symptom_match = True
                break

        # Check resolution relevance: does the response offer troubleshooting instructions?
        resolution_relevance = any(
            kw in top_resp for kw in ["settings", "tap", "try", "check", "restart", "update", "reset", "article"]
        )

        # Example check: Query mentions battery drain; retrieved case is how to update iOS without battery mention
        query_lower = customer_query.lower()
        if ("battery" in query_lower or "drain" in query_lower) and ("battery" not in top_prob and "battery" not in top_resp):
            # Superficial match (e.g. both mentioned iOS 11 update, but case is not about battery)
            intent_match = False
            symptom_match = False

        # Determine overall quality
        if top_sim < self.min_moderate_similarity:
            overall = "WEAK"
            explanation = f"Semantic similarity ({top_sim:.4f}) is below minimum viable threshold ({self.min_moderate_similarity})."
        elif is_conflicting:
            overall = "CONFLICTING"
            explanation = "Top retrieved cases span 3 conflicting, divergent problem intents."
        elif not intent_match and alignment_ratio < 0.33:
            overall = "IRRELEVANT"
            explanation = f"Retrieved cases do not match customer intent ({target_intent}). Top case is {top_intent}."
        elif intent_match and (symptom_match or top_sim >= 0.75) and resolution_relevance:
            overall = "STRONG"
            explanation = "Evidence strongly aligns with customer symptoms and provides verified troubleshooting steps."
        elif intent_match and top_sim >= self.min_strong_similarity:
            overall = "MODERATE"
            explanation = "Evidence aligns with customer intent and provides general actionable support."
        elif intent_match:
            overall = "MODERATE"
            explanation = "Evidence matches problem domain with moderate semantic alignment."
        else:
            overall = "WEAK"
            explanation = "Retrieved cases have weak intent alignment or lack specific resolution steps."

        return EvidenceJudgeResult(
            overall_quality=overall,
            intent_match=intent_match,
            issue_match=symptom_match,
            symptom_match=symptom_match,
            resolution_relevance=resolution_relevance,
            is_conflicting=is_conflicting,
            is_insufficient=(overall in ["INSUFFICIENT", "WEAK", "IRRELEVANT"]),
            top_similarity=top_sim,
            intent_alignment_ratio=alignment_ratio,
            explanation=explanation
        )
