"""
SupportDNA Agent — Step 3: Evidence Retrieval
==============================================
Retrieves candidate historical cases from the 5,545-case FAISS index and Layer 2
Resolution Knowledge using domain-aware ranking.

New Mindset:
"Find evidence that is useful for answering THIS customer's actual problem."

Output Schema per Case:
{
    "case_id": "...",
    "similarity": 0.0,
    "intent_match": true,
    "issue_match": true,
    "evidence_type": "...",
    "resolution_pattern": "...",
    "relevance": "..."
}
"""

import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np

from src.knowledge.common import RESOLUTION_DIR


@dataclass
class RetrievedEvidenceCase:
    case_id: str
    intent_id: str
    similarity: float
    intent_match: bool
    issue_match: bool
    evidence_type: str
    resolution_pattern: str
    relevance: str
    support_response: str
    customer_problem: str
    resolution_status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "intent_id": self.intent_id,
            "similarity": round(float(self.similarity), 4),
            "intent_match": bool(self.intent_match),
            "issue_match": bool(self.issue_match),
            "evidence_type": self.evidence_type,
            "resolution_pattern": self.resolution_pattern,
            "relevance": self.relevance,
            "support_response": self.support_response,
            "customer_problem": self.customer_problem,
            "resolution_status": self.resolution_status
        }


class EvidenceRetriever:
    """
    Evidence retrieval manager wrapping FAISS SemanticRetriever and DomainAwareIntentRanker.
    Packs candidates with Layer 2 resolution patterns.
    """

    def __init__(
        self,
        retriever: Any,
        domain_ranker: Any,
        embedding_model: Any,
        top_k: int = 3
    ):
        self.retriever = retriever
        self.domain_ranker = domain_ranker
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.patterns = self._load_resolution_patterns()

    def _load_resolution_patterns(self) -> Dict[str, Any]:
        pat_path = RESOLUTION_DIR / "resolution_patterns.json"
        if pat_path.exists():
            with open(pat_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def retrieve_evidence(
        self,
        query: str,
        query_understanding: Any,
        retrieval_query: Optional[str] = None
    ) -> List[RetrievedEvidenceCase]:
        """Retrieve and package candidate evidence cases for the customer's specific problem."""
        target_intent = query_understanding.intent
        customer_issues = query_understanding.issues

        search_query = retrieval_query.strip() if retrieval_query and retrieval_query.strip() else query

        # 1. FAISS Top-10 semantic candidate retrieval
        raw_candidates = self.retriever.retrieve(search_query, top_k=10)

        # 2. Compute query embedding for domain ranker
        query_emb = self.embedding_model.encode(
            [search_query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )[0].astype(np.float32)

        # 3. Domain-Aware Intent Ranking
        ranking_res = self.domain_ranker.rank_candidates(
            query_text=search_query,
            query_embedding=query_emb,
            candidate_cases=raw_candidates
        )
        ranked_cases = ranking_res["ranked_cases"]

        # Select Top-K evidence cases
        # Prioritize cases matching target intent if available
        intent_aligned = [c for c in ranked_cases if c.get("intent_id") == target_intent]
        selected = intent_aligned[:self.top_k]
        if len(selected) < self.top_k:
            for c in ranked_cases:
                if c not in selected:
                    selected.append(c)
                if len(selected) >= self.top_k:
                    break

        # 4. Enrich cases with match assessments and resolution patterns
        results = []
        for c in selected:
            case_id = c.get("case_id", "UNKNOWN")
            case_intent = c.get("intent_id", "UNKNOWN")
            sim = float(c.get("similarity", 0.0))
            prob = c.get("customer_problem", "")
            resp = c.get("support_response", "")
            status = c.get("resolution_status", "PARTIALLY_RESOLVED")

            intent_match = (case_intent == target_intent)

            # Issue match: does the candidate case address any of the customer's stated issues?
            issue_match = False
            prob_lower = prob.lower()
            for iss in customer_issues:
                iss_clean = iss.replace("_", " ")
                if iss_clean in prob_lower:
                    issue_match = True
                    break

            # Layer 2 resolution pattern for this intent
            pat_info = self.patterns.get(case_intent, {})
            wf = pat_info.get("canonical_resolution_workflow", [])
            pattern_str = " → ".join(wf) if wf else "Diagnostic Check → Targeted Troubleshooting"

            # Relevance classification for this candidate
            if intent_match and issue_match and sim >= 0.65:
                relevance = "HIGHLY_RELEVANT"
                ev_type = "DIRECT_RESOLUTION"
            elif intent_match and sim >= 0.55:
                relevance = "MODERATELY_RELEVANT"
                ev_type = "INTENT_ALIGNED_GUIDANCE"
            else:
                relevance = "MARGINAL_OR_GENERIC"
                ev_type = "BACKGROUND_SUPPORT"

            results.append(RetrievedEvidenceCase(
                case_id=case_id,
                intent_id=case_intent,
                similarity=sim,
                intent_match=intent_match,
                issue_match=issue_match,
                evidence_type=ev_type,
                resolution_pattern=pattern_str,
                relevance=relevance,
                support_response=resp,
                customer_problem=prob,
                resolution_status=status
            ))

        return results
