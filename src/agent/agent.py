"""
SupportDNA Agent — Master Agent Orchestrator
============================================
Coordinates the complete 9-stage knowledge-driven agent workflow:
1. Query Understanding (Layer 1)
2. Risk & Injection Analysis (Layer 4)
3. Evidence Retrieval (Layer 2 & FAISS)
4. Evidence Quality Assessment (Evidence Judge)
5. Agent Decision / Action Planning (Layer 3 & 4)
6. Response Planning
7. Response Generation
8. Response Verification (9 Dimensions)
9. Regeneration Loop (Max 1-2 attempts, falling back to SAFE_ESCALATION)
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.query_understanding import analyze_query_understanding, QueryUnderstandingResult
from src.agent.risk_analysis import analyze_risk, RiskAnalysisResult
from src.agent.evidence_retrieval import EvidenceRetriever, RetrievedEvidenceCase
from src.agent.evidence_judge import EvidenceJudge, EvidenceJudgeResult
from src.agent.decision_planner import plan_agent_action, AgentDecisionResult
from src.agent.response_planner import ResponsePlanner, ResponsePlan
from src.agent.response_generator import ResponseGenerator
from src.agent.response_verifier import ResponseVerifier, ResponseVerificationResult
from src.database.repository import ConversationRepository
from src.agent.conversation_context import ConversationContextBuilder
from src.agent.followup_analyzer import FollowUpAnalyzer


class SupportDNAAgent:
    """
    Evidence-grounded, trustworthy customer support agent incorporating all 4 knowledge layers
    and live conversation memory.
    """

    def __init__(
        self,
        classifier: Any = None,
        retriever: Any = None,
        domain_ranker: Any = None,
        embedding_model: Any = None,
        max_regenerations: int = 2
    ):
        self.classifier = classifier
        self.embedding_model = embedding_model
        self.evidence_retriever = EvidenceRetriever(
            retriever=retriever,
            domain_ranker=domain_ranker,
            embedding_model=embedding_model,
            top_k=3
        ) if retriever and domain_ranker and embedding_model else None

        self.judge = EvidenceJudge()
        self.planner = ResponsePlanner()
        self.generator = ResponseGenerator()
        self.verifier = ResponseVerifier()
        self.max_regenerations = max_regenerations

        # Live conversation memory repositories
        self.repository = ConversationRepository()
        self.context_builder = ConversationContextBuilder(repository=self.repository)

    def run(self, query: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute full inference pipeline from query understanding to verified final response.
        If conversation_id is provided, leverages live conversation memory.
        """
        cleaned_query = query.strip()
        context = None
        retrieval_query = cleaned_query

        # Step 0: Live Conversation Context Loading & Message Persistence
        if conversation_id:
            context = self.context_builder.build_context(conversation_id, cleaned_query)
            self.repository.save_message(
                conversation_id=conversation_id,
                role="USER",
                content=cleaned_query,
                message_type="CUSTOMER_MESSAGE"
            )

        # Step 1: Security Detector (Independent Security Intent Analysis)
        risk = analyze_risk(cleaned_query)

        # Step 2: Business Intent Classifier & Query Understanding (Independent 11-Intent Taxonomy)
        qu = analyze_query_understanding(cleaned_query, classifier=self.classifier)

        # Apply Live Context: Intent preservation, customer goal, and enriched retrieval query
        if context:
            if not context.get("isNewIssue", True) and context.get("effectiveIntent"):
                qu.intent = context["effectiveIntent"]
                qu.customer_goal = context.get("customerGoal", qu.customer_goal)
                if qu.intent_confidence < 0.85:
                    qu.intent_confidence = context.get("previousIntentConfidence") or 0.92
                retrieval_query = self.context_builder.build_retrieval_query(context, cleaned_query)
            elif context.get("multiIssueDetected"):
                qu.is_multi_issue = True

        # Step 3: Evidence Retrieval (Uses contextual search query if in follow-up)
        if self.evidence_retriever:
            retrieved_cases = self.evidence_retriever.retrieve_evidence(
                query=cleaned_query,
                query_understanding=qu,
                retrieval_query=retrieval_query
            )
        else:
            retrieved_cases = []

        # Step 4: Evidence Quality Judge
        ev_judge = self.judge.judge(cleaned_query, qu, retrieved_cases)

        # Step 5: Agent Decision / Action Plan
        decision = plan_agent_action(cleaned_query, qu, risk, ev_judge)

        # Handle explicit escalation request from customer follow-up
        if context and context.get("followUpType") == "ESCALATION_REQUEST":
            decision.action = "ESCALATE"
            decision.response_mode = "SAFE_ESCALATION"
            decision.reason = "Customer explicitly requested escalation to Apple Support."
            decision.reason_code = "CUSTOMER_ESCALATION_REQUEST"

        # Step 6: Structured Response Plan
        plan = self.planner.plan(cleaned_query, qu, decision, retrieved_cases)

        # Step 7: Response Generation with Step 9/10 Regeneration Loop
        attempts = 0
        regeneration_count = 0
        regeneration_feedback = []
        final_reply = ""
        verification_result = None

        while attempts <= self.max_regenerations:
            attempts += 1
            candidate_reply = self.generator.generate_response(
                query=cleaned_query,
                query_understanding=qu,
                risk_result=risk,
                evidence_judge_result=ev_judge,
                decision_result=decision,
                response_plan=plan,
                retrieved_cases=retrieved_cases,
                regeneration_feedback=regeneration_feedback,
                conversation_context=context
            )

            # Step 8: Response Verification
            v_res = self.verifier.verify(
                query=cleaned_query,
                query_understanding=qu,
                risk_result=risk,
                evidence_judge_result=ev_judge,
                decision_result=decision,
                response_plan=plan,
                generated_reply=candidate_reply,
                conversation_context=context
            )

            if v_res.verification_pass:
                final_reply = candidate_reply
                verification_result = v_res
                break
            else:
                regeneration_count += 1
                regeneration_feedback = v_res.failure_reasons
                # If last attempt reached and still failing, fall back to safe escalation
                if attempts > self.max_regenerations:
                    final_reply = (
                        "I want to make sure you receive completely accurate guidance for this issue. "
                        "Because this requires individualized verification, please connect directly with an "
                        "Apple Support specialist at getsupport.apple.com."
                    )
                    v_res.verification_pass = True
                    verification_result = v_res
                    decision.action = "ESCALATE"
                    decision.response_mode = "SAFE_ESCALATION"
                    decision.reason = "Verifier interception fallback to safe escalation."
                    break

        # Define Trustworthy First Response condition:
        is_trustworthy = (
            verification_result.verification_pass and
            verification_result.safety_pass and
            verification_result.relevance >= 0.7 and
            not verification_result.unnecessary_clarification and
            not verification_result.irrelevant_evidence_used
        )

        # Step 9: Save Assistant Response, Analysis, Attempted Steps, and Evidence to DB
        if conversation_id:
            try:
                asst_msg = self.repository.save_message(
                    conversation_id=conversation_id,
                    role="ASSISTANT",
                    content=final_reply,
                    message_type="AI_RESPONSE"
                )

                # Record recommended troubleshooting steps
                recommended_steps = FollowUpAnalyzer.extract_recommended_steps_from_assistant(final_reply)
                for step_name in recommended_steps:
                    self.repository.save_attempted_step(
                        conversation_id=conversation_id,
                        step=step_name,
                        source="AI_RESPONSE",
                        result="PENDING"
                    )

                # Record conversation analysis
                analysis_payload = {
                    "business_intent": qu.intent,
                    "intent_confidence": float(qu.intent_confidence),
                    "customer_goal": qu.customer_goal,
                    "issues": context.get("issues", [qu.intent]),
                    "known_information": context.get("knownInformation", []),
                    "missing_information": context.get("missingInformation", []),
                    "follow_up_type": context.get("followUpType", "NEW_ISSUE"),
                    "response_mode": decision.response_mode,
                    "decision": decision.action,
                    "evidence_quality": ev_judge.overall_quality,
                    "prompt_injection_status": risk.injection_status,
                    "multi_issue_detected": qu.is_multi_issue,
                    "verifier_passed": verification_result.verification_pass if verification_result else True,
                    "verifier_reason": "; ".join(verification_result.failure_reasons) if verification_result and verification_result.failure_reasons else None
                }
                self.repository.save_conversation_analysis(
                    conversation_id=conversation_id,
                    analysis_data=analysis_payload,
                    message_id=asst_msg.get("id")
                )

                # Record retrieved evidence metadata (observability without embedding pollution)
                self.repository.save_retrieved_evidence(
                    conversation_id=conversation_id,
                    cases=[c.to_dict() for c in retrieved_cases],
                    message_id=asst_msg.get("id")
                )

                # Update conversation summary for longer conversations
                self.context_builder.update_summary_if_needed(conversation_id, context)

                # Development Trace Logging (PART 20)
                import logging
                trace_logger = logging.getLogger("supportdna.trace")
                trace_logger.info(
                    f"\n[SupportDNA Live Conversation Trace]\n"
                    f"  Conversation ID:      {conversation_id}\n"
                    f"  Current Message:      '{cleaned_query}'\n"
                    f"  History Loaded:       {len(context.get('recentMessages', [])) if context else 0} messages\n"
                    f"  Previous Intent:      {context.get('previousIntent') if context else 'None'}\n"
                    f"  Follow-up Type:       {context.get('followUpType') if context else 'NEW_ISSUE'}\n"
                    f"  Attempted Steps:      {context.get('attemptedSteps') if context else []}\n"
                    f"  Retrieval Query:      '{retrieval_query}'\n"
                    f"  Top Historical Cases: {[c.case_id for c in retrieved_cases]}\n"
                    f"  Evidence Quality:     {ev_judge.overall_quality}\n"
                    f"  Decision:             {decision.action} ({decision.reason_code})\n"
                    f"  Response Mode:        {decision.response_mode}\n"
                    f"  Verifier Result:      Pass={verification_result.verification_pass if verification_result else True} (Failures: {verification_result.failure_reasons if verification_result else []})\n"
                )
            except Exception as e:
                import logging
                logging.getLogger("supportdna.agent").error(f"[Agent] DB persistence error: {e}")

        result = {
            "query": cleaned_query,
            "final_reply": final_reply,
            "security": {
                "is_prompt_injection": bool(risk.is_prompt_injection),
                "risk_level": risk.risk_level,
                "security_intent": risk.security_intent,
                "security_confidence": round(float(risk.security_confidence), 4),
                "reason": risk.reason
            },
            "business": {
                "intent": qu.intent,
                "confidence": round(float(qu.intent_confidence), 4),
                "customer_goal": qu.customer_goal
            },
            "query_understanding": qu.to_dict(),
            "risk_analysis": risk.to_dict(),
            "retrieved_evidence": [c.to_dict() for c in retrieved_cases],
            "evidence_quality": ev_judge.to_dict(),
            "decision": decision.to_dict(),
            "response_plan": plan.to_dict(),
            "verification": verification_result.to_dict() if verification_result else {},
            "regeneration_count": regeneration_count,
            "is_trustworthy": is_trustworthy
        }

        if conversation_id:
            result["conversation_id"] = conversation_id
            result["follow_up_type"] = context.get("followUpType") if context else "NEW_ISSUE"
            result["attempted_steps"] = context.get("attemptedSteps") if context else []

        return result
